"""
Stage 5 — Gate-fire instrumentation across all v2.x adaptive runs (qwen3-8b).

Parses [adaptive_start] / [layer1] / [layer2] log lines and produces a
quantified table of how often each gate fired. Quantifies the §4-§5 claim
that "G1, G3, A1, A2, A3 fire 0× on qwen" with actual counts.

Output: analysis/evolution/stage5_gate_fire.md
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "analysis/evolution/stage5_gate_fire.md"

VARIANTS = [
    "v1_g2on",
    "v2_5",
    "v2_6",
    "v2_7",
    "v2_8",
    "v2_9",
    "v2_10",
    "v2_11",
]
TASKS = ["hotpotqa", "ifbench", "hover", "musique"]

# Patterns
RE_START = re.compile(r"^\[adaptive_start\] skip reason=(\S+)\s+(.*)$")
RE_LAYER1 = re.compile(r"^\[layer1\]\s+(.*)$")
RE_LAYER2 = re.compile(
    r"^\[layer2\] decision pair=\((\d+),(\d+)\) ancestor=(\d+) algo=(\w+) reason=(\S+)"
)


def parse_log(log_path: Path):
    """Return per-event counters for a single cell log."""
    if not log_path.exists():
        return None
    counters = {
        "start_skip_reasons": Counter(),
        "layer1_skip_reasons": Counter(),
        "layer2_algo_chosen": Counter(),
        "layer2_reason": Counter(),
        "n_layer2_decisions": 0,
    }
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        m = RE_START.match(line)
        if m:
            counters["start_skip_reasons"][m.group(1)] += 1
            continue
        m = RE_LAYER1.match(line)
        if m:
            # parse skip_reason from format like:
            # "no_valid_adaptive_pair: <2 candidates"
            # "skip pair=(...) reason=G1_parent_strength"
            tail = m.group(1)
            if "skip" in tail and "reason=" in tail:
                rs = re.search(r"reason=(\S+)", tail)
                if rs:
                    counters["layer1_skip_reasons"][rs.group(1)] += 1
            elif tail.startswith("no_valid_adaptive_pair"):
                counters["layer1_skip_reasons"]["no_valid_adaptive_pair"] += 1
            elif "maturity_gini_unavailable" in tail:
                counters["layer1_skip_reasons"]["maturity_gini_unavailable"] += 1
            else:
                counters["layer1_skip_reasons"]["other"] += 1
            continue
        m = RE_LAYER2.match(line)
        if m:
            counters["n_layer2_decisions"] += 1
            counters["layer2_algo_chosen"][m.group(4)] += 1
            counters["layer2_reason"][m.group(5)] += 1
    return counters


def main():
    rows = []
    for variant in VARIANTS:
        log_dir = ROOT / f"adaptive_merge/logs_{variant}"
        if not log_dir.exists():
            continue
        for task in TASKS:
            log_path = log_dir / f"qwen_{task}_adaptive_s0.log"
            counts = parse_log(log_path)
            if counts is None:
                continue
            rows.append({"variant": variant, "task": task, "log": str(log_path), **counts})

    # ── Output ──
    out = ["# Stage 5 — Gate-fire instrumentation (qwen3-8b adaptive runs)",
           "",
           "Parses every `[adaptive_start]`, `[layer1]`, `[layer2]` log line "
           "across all v2.x adaptive runs on qwen3-8b. Quantifies §4-§5 claim "
           "that catastrophe-prevention gates (G1, G3, A1, A2, A3) fire 0× on qwen.",
           "",
           f"Variants surveyed: {', '.join([v for v in VARIANTS if (ROOT / f'adaptive_merge/logs_{v}').exists()])}",
           f"Cells parsed: {len(rows)}",
           ""]

    # ── Aggregate per gate (across all cells) ──
    out.append("## Aggregate gate-fire counts across all surveyed cells")
    out.append("")
    agg_start = Counter()
    agg_layer1 = Counter()
    agg_algo = Counter()
    agg_reason = Counter()
    n_layer2 = 0
    for r in rows:
        agg_start.update(r["start_skip_reasons"])
        agg_layer1.update(r["layer1_skip_reasons"])
        agg_algo.update(r["layer2_algo_chosen"])
        agg_reason.update(r["layer2_reason"])
        n_layer2 += r["n_layer2_decisions"]

    out.append("### Layer −1 (start gate) skip reasons")
    out.append("")
    out.append("| Skip reason | Count |")
    out.append("|-------------|------:|")
    for reason, count in agg_start.most_common():
        out.append(f"| `{reason}` | {count} |")
    out.append(f"| **TOTAL Layer-1 skips** | **{sum(agg_start.values())}** |")
    out.append("")

    out.append("### Layer 1 (skip gate) reasons")
    out.append("")
    out.append("| Skip reason | Count |")
    out.append("|-------------|------:|")
    if agg_layer1:
        for reason, count in agg_layer1.most_common():
            mark = ""
            if reason in ("G1_parent_strength", "G3_maturity_imbalance"):
                mark = " ⭐"
            out.append(f"| `{reason}`{mark} | {count} |")
    else:
        out.append("| _(no Layer 1 skip events logged)_ | 0 |")
    out.append(f"| **TOTAL Layer-1-pair skips** | **{sum(agg_layer1.values())}** |")
    out.append("")

    out.append("### Layer 2 (algorithm routing) — decisions made")
    out.append("")
    out.append(f"Total Layer 2 decisions logged: **{n_layer2}**")
    out.append("")
    out.append("| Algorithm chosen | Count |  | Reason | Count |")
    out.append("|------------------|------:|--|--------|------:|")
    algo_items = list(agg_algo.most_common())
    reason_items = list(agg_reason.most_common())
    max_rows = max(len(algo_items), len(reason_items))
    for i in range(max_rows):
        a, ac = (algo_items[i] if i < len(algo_items) else ("", ""))
        r, rc = (reason_items[i] if i < len(reason_items) else ("", ""))
        ac_s = str(ac) if ac != "" else ""
        rc_s = str(rc) if rc != "" else ""
        a_s = f"`{a}`" if a else ""
        r_s = f"`{r}`" if r else ""
        out.append(f"| {a_s} | {ac_s} |  | {r_s} | {rc_s} |")
    out.append("")

    # ── Per-rule fire-rate summary table ──
    out.append("## ⭐ Per-rule fire rate (the §4-§5 headline table)")
    out.append("")
    out.append("Concatenating all variants' qwen runs:")
    out.append("")
    out.append("| Layer | Rule | Fire count | Notes |")
    out.append("|-------|------|-----------:|-------|")
    out.append(f"| Layer −1 | warmup_not_passed | {agg_start.get('warmup_not_passed', 0)} | always fires until iter_frac ≥ 0.25 |")
    out.append(f"| Layer −1 | frontier_too_small | {agg_start.get('frontier_too_small', 0)} | task-specific (ifbench frontier slow to grow) |")
    out.append(f"| Layer −1 | not_plateaued_optional | {agg_start.get('not_plateaued_optional', 0)} | the dominant gate (~71% of all rejection events on qwen) |")
    out.append(f"| Layer −1 | other start skips | {sum(v for k,v in agg_start.items() if k not in ('warmup_not_passed','frontier_too_small','not_plateaued_optional'))} | misc |")
    out.append(f"| Layer 1 | **G1 parent_strength** | **{agg_layer1.get('G1_parent_strength', 0)}** | designed to skip merges with weak parents |")
    out.append(f"| Layer 1 | **G3 maturity_imbalance** | **{agg_layer1.get('G3_maturity_imbalance', 0)}** | catches §15.3 catastrophes (-28.66 / -35.67) |")
    out.append(f"| Layer 1 | parent_recent_winrate (G2 was deleted) | {agg_layer1.get('parent_recent_winrate_low_a', 0) + agg_layer1.get('parent_recent_winrate_low_b', 0)} | optional gate (off in default config) |")
    out.append(f"| Layer 2 | **A1 bloat (L_MAX)** | **{agg_reason.get('bloat_L_MAX', 0)}** | catches §7 IFBench combine_all catastrophe |")
    out.append(f"| Layer 2 | **A2 specialization split** | **{agg_reason.get('specialization_split', 0)}** | catches multi-domain heterogeneity (§15.2) |")
    out.append(f"| Layer 2 | **A3 near-duplicate** | **{agg_reason.get('near_duplicate', 0)}** | when parents are nearly identical |")
    out.append(f"| Layer 2 | **A4 default (=`original`)** | **{agg_reason.get('safe_complementary_original', 0) + agg_reason.get('layer2_routing_disabled', 0)}** | fallback — most merges land here |")
    out.append("")

    # Summary of A4-only routing
    a4_count = (
        agg_reason.get('safe_complementary_original', 0)
        + agg_reason.get('layer2_routing_disabled', 0)
    )
    a1_count = agg_reason.get('bloat_L_MAX', 0)
    a2_count = agg_reason.get('specialization_split', 0)
    a3_count = agg_reason.get('near_duplicate', 0)
    routed_count = a1_count + a2_count + a3_count + a4_count
    if routed_count > 0:
        out.append(f"→ **A1+A2+A3 routing fired {a1_count + a2_count + a3_count} / {routed_count} = "
                   f"{(a1_count+a2_count+a3_count)/routed_count*100:.1f}% of Layer 2 decisions.** "
                   f"A4 default catches {a4_count}/{routed_count} = "
                   f"{a4_count/routed_count*100:.1f}%. ")
    out.append("")

    # ── Per-variant breakdown ──
    out.append("## Per-variant per-task breakdown")
    out.append("")
    out.append("| Variant | Task | Start skips | Layer-1 skips | Layer-2 decisions | A1+A2+A3 fired? |")
    out.append("|---------|------|------------:|--------------:|------------------:|:---------------:|")
    for r in rows:
        n_start = sum(r["start_skip_reasons"].values())
        n_l1 = sum(r["layer1_skip_reasons"].values())
        n_l2 = r["n_layer2_decisions"]
        n_routing = (
            r["layer2_reason"].get("bloat_L_MAX", 0)
            + r["layer2_reason"].get("specialization_split", 0)
            + r["layer2_reason"].get("near_duplicate", 0)
        )
        marker = "✅" if n_routing > 0 else "❌"
        out.append(
            f"| {r['variant']} | {r['task']} | {n_start} | {n_l1} | {n_l2} | "
            f"{marker} ({n_routing} fired) |"
        )
    out.append("")

    # ── Conclusion ──
    out.append("## Conclusion")
    out.append("")
    g1_count = agg_layer1.get('G1_parent_strength', 0)
    g3_count = agg_layer1.get('G3_maturity_imbalance', 0)
    out.append(
        f"Across all {len(rows)} qwen3-8b adaptive runs surveyed, the catastrophe-prevention gates "
        f"fire as follows:"
    )
    out.append("")
    out.append(f"- **G1** (parent strength quantile): **{g1_count}× fires** {'❌ designed but never triggered on qwen' if g1_count == 0 else ''}")
    out.append(f"- **G3** (maturity imbalance): **{g3_count}× fires** {'❌ designed but never triggered on qwen' if g3_count == 0 else ''}")
    out.append(f"- **A1** (bloat L_MAX): **{a1_count}× fires** {'❌ designed but never triggered on qwen' if a1_count == 0 else ''}")
    out.append(f"- **A2** (specialization split): **{a2_count}× fires** {'❌ designed but never triggered on qwen' if a2_count == 0 else ''}")
    out.append(f"- **A3** (near-duplicate): **{a3_count}× fires** {'❌ designed but never triggered on qwen' if a3_count == 0 else ''}")
    out.append("")
    if g1_count + g3_count + a1_count + a2_count + a3_count == 0:
        out.append(
            "→ **All five catastrophe-prevention gates fired 0× across the entire qwen ablation chain.** "
            "On Qwen3-8B, the adaptive policy effectively reduces to: "
            "Layer −1 plateau gate + adaptive_diversity pair selection + always-`original` algorithm. "
            "The gates are silent infrastructure designed for catastrophe conditions that do not recur on Qwen3-8B. "
            "Cross-model validation (gpt × 5 task v2_first runs) will determine whether they fire on gpt-4.1-mini, "
            "where the documented catastrophes (§15.3 -28.66 cell) actually occurred."
        )
    out.append("")
    OUT_PATH.write_text("\n".join(out))
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes, {len(rows)} cells parsed)")


if __name__ == "__main__":
    main()
