"""
Stage 3: Per-accept-event content evolution.

For selected cells, walk candidates in order and report what content evolved at each
ACCEPT event (every accepted candidate is one) and especially at each PARETO-BEST-UPDATE
event (when the running max val score changes).

For each event we extract:
  - which predictor(s) changed vs parent
  - parent prompt (chars / first line)
  - child prompt (chars / first line)
  - a categorization of the diff:
      * unchanged passthrough (Case-A merge with one parent's predictor verbatim)
      * pure inflation (child contains parent's text as a substring + adds more)
      * pure deflation (parent contains child's text as substring)
      * rewrite (significant chars overlap but neither contains the other)
      * structural additions (new bullets / conditional clauses / examples)
  - up to ~600 char excerpt of *what was newly added* (child − parent at line level)

Output: analysis/evolution/stage3_content_evolution.md
"""
import csv
import difflib
import json
import pickle
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "analysis/evolution/candidates.csv"
OUT_DIR = ROOT / "analysis/evolution/stage3_content_evolution"

# All 50 qwen Phase A cells; auto-discovered from candidates.csv. One markdown file per task.
TASKS = ["hotpotqa", "ifbench", "hover", "musique", "2wiki"]


def find_state(task: str, cell: str):
    for root in [
        ROOT / f"results/sec4_configuration_sweep_seed0/qwen3-8b/{task}/{cell}",
        ROOT / f"results/sec4_configuration_sweep_seed0/qwen3-8b/{task}/{cell}",
        ROOT / f"results/sec4_configuration_sweep_seed0/gpt-4.1-mini/{task}/{cell}",
        ROOT / f"results/sec4_configuration_sweep_seed0/results/sec4_configuration_sweep_seed0/qwen3-8b/musique/{cell}",
        ROOT / f"runs/phase_a_main_qwen/{task}/{cell}",
    ]:
        gs = root / "gepa_state.bin"
        if gs.exists():
            return root
    return None


def load_state(task: str, cell: str):
    root = find_state(task, cell)
    if root is None:
        return None
    s = pickle.load(open(root / "gepa_state.bin", "rb"))
    bc = root / "best_candidate.json"
    best_cand_dict = None
    if bc.exists():
        try:
            best_cand_dict = json.load(open(bc)).get("best_candidate")
        except Exception:
            pass
    return {"state": s, "root": root, "best_candidate_dict": best_cand_dict}


def categorize_diff(parent_text: str, child_text: str) -> str:
    p, c = parent_text or "", child_text or ""
    if p == c:
        return "UNCHANGED (passthrough)"
    if not p and c:
        return "FROM_EMPTY"
    if not c:
        return "TO_EMPTY"
    # substring tests
    if p in c:
        ratio = len(c) / max(1, len(p))
        return f"PURE_INFLATION (×{ratio:.1f}, +{len(c) - len(p):,} chars)"
    if c in p:
        ratio = len(p) / max(1, len(c))
        return f"PURE_DEFLATION (÷{ratio:.1f}, −{len(p) - len(c):,} chars)"
    # token-set similarity
    sm = difflib.SequenceMatcher(a=p, b=c, autojunk=False)
    r = sm.ratio()
    if r > 0.7:
        return f"REWRITE (overlap={r:.2f}, Δ={len(c) - len(p):+,d} chars)"
    return f"REPLACE (overlap={r:.2f}, Δ={len(c) - len(p):+,d} chars)"


BULLET_RE = re.compile(r"\n[\-\*]\s|\n\d+[\.\)]\s")
COND_RES = {
    "If": re.compile(r"\bIf\b"),
    "When": re.compile(r"\bWhen\b"),
    "Ensure": re.compile(r"\bensure\b", re.I),
    "must": re.compile(r"\bmust\b", re.I),
    "should": re.compile(r"\bshould\b", re.I),
}


def feature_counts(text: str) -> dict:
    text = text or ""
    out = {"bullets": len(BULLET_RE.findall(text))}
    for k, rx in COND_RES.items():
        out[k] = len(rx.findall(text))
    return out


def feature_delta(parent: str, child: str) -> dict:
    pf, cf = feature_counts(parent), feature_counts(child)
    return {k: cf[k] - pf[k] for k in cf}


def newly_added_lines(parent_text: str, child_text: str, max_chars: int = 600) -> str:
    """Return up to max_chars of lines present in child but not in parent (token-level diff)."""
    parent_lines = (parent_text or "").splitlines()
    child_lines = (child_text or "").splitlines()
    sm = difflib.SequenceMatcher(a=parent_lines, b=child_lines, autojunk=False)
    additions = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            for ln in child_lines[j1:j2]:
                if ln.strip():
                    additions.append(ln)
    out = "\n".join(additions)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def removed_lines(parent_text: str, child_text: str, max_chars: int = 400) -> str:
    parent_lines = (parent_text or "").splitlines()
    child_lines = (child_text or "").splitlines()
    sm = difflib.SequenceMatcher(a=parent_lines, b=child_lines, autojunk=False)
    drops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            for ln in parent_lines[i1:i2]:
                if ln.strip():
                    drops.append(ln)
    out = "\n".join(drops)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


def first_line(text: str, max_chars: int = 120) -> str:
    s = (text or "").strip().splitlines()
    if not s:
        return "(empty)"
    line = s[0]
    return (line[:max_chars] + "...") if len(line) > max_chars else line


def find_best_idx(state_obj, best_cand_dict):
    if best_cand_dict is None:
        return None
    pc = state_obj.get("program_candidates", [])
    for i, c in enumerate(pc):
        if c == best_cand_dict:
            return i
    return None


def load_cells_csv():
    rows = list(csv.DictReader(open(CSV_PATH)))
    for r in rows:
        r["idx"] = int(r["idx"])
        r["val"] = float(r["val"]) if r["val"] not in (None, "") else None
        r["val_delta"] = (
            float(r["val_delta"]) if r["val_delta"] not in (None, "", "None") else None
        )
        r["prompt_chars"] = int(r["prompt_chars"])
    return rows


def analyze_cell(task: str, cell: str, header: str, csv_rows) -> str:
    info = load_state(task, cell)
    if info is None:
        return f"### {header}\n\n_no gepa_state.bin found_\n"
    state_obj = info["state"]
    pc = state_obj.get("program_candidates", [])
    parents_arr = state_obj.get("parent_program_for_candidate", [])
    best_idx = find_best_idx(state_obj, info["best_candidate_dict"])

    cell_rows = sorted(
        [r for r in csv_rows if r["task"] == task and r["cell"] == cell],
        key=lambda r: r["idx"],
    )
    rows_by_idx = {r["idx"]: r for r in cell_rows}
    n = len(pc)

    out = [f"### {header}", f"\n{n} candidates total. ⭐ = overall best_candidate. ★ = new running-best.", ""]

    running_best = None
    n_pareto_updates = 0
    n_inert = 0
    op_counter = Counter()
    feature_totals = {"reflect": Counter(), "merge": Counter()}
    feature_n = {"reflect": 0, "merge": 0}

    for idx in range(n):
        row = rows_by_idx.get(idx)
        if row is None:
            continue
        origin = row["origin"]
        val = row["val"]
        if val is None:
            continue

        is_pareto_best_update = False
        if running_best is None or val > running_best + 1e-9:
            is_pareto_best_update = True
            n_pareto_updates += 1
            running_best = val

        is_overall_best = (idx == best_idx)
        marker = ""
        if is_overall_best:
            marker = "⭐ overall-best"
        elif is_pareto_best_update:
            marker = "★ new running-best"

        if origin == "seed":
            child = pc[idx]
            preds_summary = ", ".join(f"{k}({len(v)} chars)" for k, v in child.items())
            out.append(
                f"**iter {idx}** · seed · val={val:.4f} {marker}\n"
                f"- predictors: {preds_summary}\n"
            )
            continue

        # parent indices come from parent_program_for_candidate (could be list/None for seed)
        parent_ref = parents_arr[idx] if idx < len(parents_arr) else None
        parent_indices = []
        if isinstance(parent_ref, (list, tuple)):
            parent_indices = [p for p in parent_ref if p is not None]
        elif isinstance(parent_ref, int):
            parent_indices = [parent_ref]

        if not parent_indices:
            out.append(f"**iter {idx}** · {origin} · val={val:.4f} {marker}\n- _no parent reference_\n")
            continue

        child = pc[idx]
        # For each predictor: find which parent's text the child differs from / matches
        # We'll compare against the parent that "claimed" the predictor (closest match) so
        # the diff focuses on the channel that actually changed.
        per_pred_lines = []
        any_change = False
        for pred_name, child_text in child.items():
            # pick the parent with most similar text for this predictor, then diff against it
            cand_parents = []
            for p in parent_indices:
                if p < len(pc) and isinstance(pc[p], dict) and pred_name in pc[p]:
                    cand_parents.append((p, pc[p][pred_name]))
            if not cand_parents:
                continue
            # if exact match exists, mark passthrough
            exact = [pp for pp in cand_parents if pp[1] == child_text]
            if exact:
                p_idx = exact[0][0]
                per_pred_lines.append(f"  - `{pred_name}`: passthrough from parent[{p_idx}] ({len(child_text):,} chars)")
                continue
            any_change = True
            # pick most-similar parent
            cand_parents.sort(
                key=lambda pp: difflib.SequenceMatcher(a=pp[1], b=child_text, autojunk=False).quick_ratio(),
                reverse=True,
            )
            p_idx, p_text = cand_parents[0]
            cat = categorize_diff(p_text, child_text)
            fdelta = feature_delta(p_text, child_text)
            f_str = " ".join(f"Δ{k}={v:+d}" for k, v in fdelta.items() if v != 0) or "(no struct change)"
            per_pred_lines.append(
                f"  - `{pred_name}`: {cat} vs parent[{p_idx}] · {f_str}"
            )
            added = newly_added_lines(p_text, child_text, max_chars=350)
            removed_excerpt = removed_lines(p_text, child_text, max_chars=200)
            if added:
                per_pred_lines.append(f"    NEW:\n```\n{added}\n```")
            if removed_excerpt and "PURE_INFLATION" not in cat:
                per_pred_lines.append(f"    DROPPED:\n```\n{removed_excerpt}\n```")
            # contribute to feature totals (only for changed predictors)
            if origin in feature_totals:
                for k, v in fdelta.items():
                    feature_totals[origin][k] += v
                feature_n[origin] += 1

        op_counter[origin] += 1
        if not any_change:
            n_inert += 1

        parents_str = ",".join(str(p) for p in parent_indices)
        dval = row["val_delta"]
        dval_str = f"Δval={dval:+.4f}" if dval is not None else "Δval=?"
        flag = "INERT (no predictor changed)" if not any_change else ""
        out.append(
            f"**iter {idx}** · {origin}({parents_str}) · val={val:.4f} ({dval_str}) "
            f"chars={row['prompt_chars']:,} {marker} {flag}".rstrip()
        )
        out.extend(per_pred_lines)
        out.append("")

    # Cell summary
    summary_lines = [
        f"\n_Cell summary:_ {n_pareto_updates} running-best updates; "
        f"{n_inert} inert merges (no predictor changed); ops="
        + ", ".join(f"{k}={v}" for k, v in op_counter.items())
    ]
    for origin in ("reflect", "merge"):
        if feature_n[origin] > 0:
            avg = ", ".join(f"Δ{k}={feature_totals[origin][k] / feature_n[origin]:+.2f}/pred-step"
                            for k in ["bullets", "If", "When", "Ensure", "must", "should"])
            summary_lines.append(f"- avg per *changed* {origin} predictor: {avg}")
    out.extend(summary_lines)
    out.append("")
    return "\n".join(out)


def aggregate_overview(csv_rows) -> str:
    """Cross-cell overview: how many running-best updates per task come from each origin."""
    out = ["## Overview: Pareto-best-updates by origin (per task, qwen Phase A, all 50 cells)", ""]
    out.append("| Task | n cells | running-best updates total | reflect | merge | seed |")
    out.append("|------|--------:|--------------------------:|--------:|------:|-----:|")

    pa = [r for r in csv_rows if r["group"] == "phase_a_qwen"]
    for task in ["hotpotqa", "ifbench", "hover", "musique"]:
        cells = sorted({r["cell"] for r in pa if r["task"] == task})
        total_updates = 0
        origin_counts = Counter()
        for cell in cells:
            cell_rows = sorted(
                [r for r in pa if r["task"] == task and r["cell"] == cell],
                key=lambda r: r["idx"],
            )
            running = None
            for r in cell_rows:
                if r["val"] is None:
                    continue
                if running is None or r["val"] > running + 1e-9:
                    total_updates += 1
                    origin_counts[r["origin"]] += 1
                    running = r["val"]
        out.append(
            f"| {task} | {len(cells)} | {total_updates} | "
            f"{origin_counts['reflect']} | {origin_counts['merge']} | {origin_counts['seed']} |"
        )
    out.append("")
    return "\n".join(out)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = load_cells_csv()
    pa = [r for r in csv_rows if r["group"] == "phase_a_qwen"]

    # Master overview file
    overview = ["# Stage 3 — Per-accept-event content evolution",
                "",
                "Each accepted candidate is one accept event; the running-max val "
                "defines the Pareto-best frontier. We mark events where the running-max "
                "increases with **★** (new running-best) and where the cell's overall "
                "best_candidate is reached with **⭐**. For every accept event we report, "
                "per predictor, whether it was a pure passthrough, an inflation, a "
                "deflation, or a rewrite — and we extract the lines that were newly "
                "added (and dropped) by the operation.",
                "",
                "Per-task timelines: see `stage3_content_evolution/{task}.md` for the "
                "full chronology of every accept event in every qwen Phase A cell.",
                ""]
    overview.append(aggregate_overview(csv_rows))
    (OUT_DIR / "_overview.md").write_text("\n".join(overview))
    print(f"Wrote {OUT_DIR / '_overview.md'}")

    # Per-task files
    for task in TASKS:
        cells = sorted({r["cell"] for r in pa if r["task"] == task})
        if not cells:
            continue
        body = [f"# Stage 3 — {task} content evolution timelines",
                "",
                f"{len(cells)} cells · ★ = new running-best · ⭐ = overall best_candidate",
                "",
                "See `_overview.md` for cross-task summary.",
                "",
                "---",
                ""]
        for cell in cells:
            header = f"{task} · {cell}"
            body.append(analyze_cell(task, cell, header, csv_rows))
            body.append("---\n")
        out_path = OUT_DIR / f"{task}.md"
        out_path.write_text("\n".join(body))
        print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes, {len(cells)} cells)")


if __name__ == "__main__":
    main()
