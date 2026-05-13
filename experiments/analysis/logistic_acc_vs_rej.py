"""A6 — Five nested logistic regression models predicting accept/reject.

Builds 5 models with progressively more feature groups, each evaluated
under GroupKFold(5) by cell_id (= model+benchmark+algo+policy) to prevent
within-cell event correlation from inflating AUC.

Models:
  A: lexical only
  B: A + lineage
  C: B + semantic (SNR, SLC)
  D: C + judge (clarity, specificity, IC, coverage_vs_parents, contradiction)
  E: D + subsample profile  ← sanity check (NOT a research finding)

Reports for each: accuracy, balanced accuracy, ROC-AUC, PR-AUC (mean +/-
std across folds).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def F(x):
    try:
        return float(x) if x not in (None, "", "None") else None
    except Exception:
        return None


def B(x):
    if x in ("True", "true", True, 1, "1"):
        return 1
    if x in ("False", "false", False, 0, "0"):
        return 0
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accepted_csv", required=True, type=Path)
    ap.add_argument("--rejected_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_acc = list(csv.DictReader(open(args.accepted_csv)))
    rows_rej = list(csv.DictReader(open(args.rejected_csv)))
    print(f"Loaded {len(rows_acc)} accepted, {len(rows_rej)} rejected.",
          file=sys.stderr)

    all_rows = []
    for r in rows_acc:
        r["__y"] = 1
        all_rows.append(r)
    for r in rows_rej:
        r["__y"] = 0
        all_rows.append(r)

    # Feature groups
    LEXICAL = ["length_delta_vs_anc_total", "sentence_entropy", "predictor_entropy",
               "coverage_min", "coverage_p1", "coverage_p2",
               "novelty_fraction"]
    LINEAGE = ["noop_predictor_rate", "parent_gen_depth_max"]
    SEMANTIC = ["snr_semantic_novelty_rate", "slc_lost_count_max"]
    JUDGE = ["judge_clarity", "judge_specificity",
             "judge_internal_consistency", "judge_coverage_vs_parents"]
    JUDGE_BOOL = ["judge_contradiction_present"]  # bool; treat 1/0
    SUBSAMPLE = ["subsample_win", "subsample_tie", "subsample_loss"]

    MODELS = {
        "A_lexical_only":    LEXICAL,
        "B_+_lineage":       LEXICAL + LINEAGE,
        "C_+_semantic":      LEXICAL + LINEAGE + SEMANTIC,
        "D_+_judge":         LEXICAL + LINEAGE + SEMANTIC + JUDGE + JUDGE_BOOL,
        "E_sanity_+_sub":    LEXICAL + LINEAGE + SEMANTIC + JUDGE + JUDGE_BOOL + SUBSAMPLE,
    }

    def featurize(rows, fields):
        X, y, groups = [], [], []
        for r in rows:
            vec = []
            ok = True
            for f in fields:
                if f in JUDGE_BOOL:
                    v = B(r.get(f))
                else:
                    v = F(r.get(f))
                if v is None:
                    ok = False
                    break
                vec.append(v)
            if not ok:
                continue
            X.append(vec)
            y.append(r["__y"])
            cell_id = f"{r.get('model')}|{r.get('dataset')}|{r.get('algo')}|{r.get('policy')}"
            groups.append(cell_id)
        return np.array(X), np.array(y), np.array(groups)

    results = []

    for model_name, feat_set in MODELS.items():
        X, y, groups = featurize(all_rows, feat_set)
        n_groups = len(set(groups))
        n_splits = min(5, n_groups)
        if n_splits < 2:
            print(f"  [skip] {model_name}: not enough groups ({n_groups})",
                  file=sys.stderr)
            continue

        gkf = GroupKFold(n_splits=n_splits)
        accs, baccs, aucs, prs = [], [], [], []
        for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                continue
            scaler = StandardScaler().fit(X_train)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(scaler.transform(X_train), y_train)
            pred = clf.predict(scaler.transform(X_test))
            prob = clf.predict_proba(scaler.transform(X_test))[:, 1]
            accs.append(accuracy_score(y_test, pred))
            baccs.append(balanced_accuracy_score(y_test, pred))
            aucs.append(roc_auc_score(y_test, prob))
            prs.append(average_precision_score(y_test, prob))

        # Fit on full data for coefficient interpretability
        scaler_full = StandardScaler().fit(X)
        clf_full = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf_full.fit(scaler_full.transform(X), y)

        # majority baseline
        majority = max(y.mean(), 1 - y.mean())
        coef_str = " | ".join(
            f"{f}:{c:+.2f}"
            for f, c in sorted(zip(feat_set, clf_full.coef_[0]),
                               key=lambda x: -abs(x[1]))[:5]
        )

        results.append({
            "model": model_name,
            "n_features": len(feat_set),
            "n_examples": len(X),
            "n_folds": len(accs),
            "accuracy_mean": float(np.mean(accs)) if accs else float("nan"),
            "accuracy_std": float(np.std(accs)) if accs else float("nan"),
            "balanced_accuracy_mean": float(np.mean(baccs)) if baccs else float("nan"),
            "balanced_accuracy_std": float(np.std(baccs)) if baccs else float("nan"),
            "roc_auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "roc_auc_std": float(np.std(aucs)) if aucs else float("nan"),
            "pr_auc_mean": float(np.mean(prs)) if prs else float("nan"),
            "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
            "majority_baseline": float(majority),
            "top_5_coefs": coef_str,
        })

    # write
    if results:
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Wrote {args.out_csv}", file=sys.stderr)

    # Print summary
    print(f"\n{'='*100}")
    print(f"  LOGISTIC REGRESSION (GroupKFold by cell_id)")
    print(f"{'='*100}")
    print(f"\n{'model':22s} {'n':>5s} {'maj':>6s} {'acc':>10s} {'bal_acc':>11s} {'ROC-AUC':>11s} {'PR-AUC':>11s}")
    print("-" * 95)
    for r in results:
        print(f"{r['model']:22s} {r['n_examples']:>5d} {r['majority_baseline']:>6.3f}"
              f" {r['accuracy_mean']:>5.3f}±{r['accuracy_std']:.2f}"
              f" {r['balanced_accuracy_mean']:>6.3f}±{r['balanced_accuracy_std']:.2f}"
              f" {r['roc_auc_mean']:>6.3f}±{r['roc_auc_std']:.2f}"
              f" {r['pr_auc_mean']:>6.3f}±{r['pr_auc_std']:.2f}")

    print(f"\n  All metrics: GroupKFold(5) by cell_id, class_weight='balanced'.")
    print(f"  Model E is sanity check — confirms subsample profile dominates ")
    print(f"  the gate. Models A-D test prompt-level predictors.")

    print(f"\n  Top 5 coefficients per model (full-data fit, standardized):")
    for r in results:
        print(f"    {r['model']:22s}: {r['top_5_coefs']}")


if __name__ == "__main__":
    main()
