"""
Repeated subject-level split stability analysis.

The primary analysis reports a single held-out test AUC (0.932) from one
subject-level split (seed 42) of n = 61 test subjects. A reviewer's first
question is: was that a lucky draw? This experiment answers it directly.

For each of N random subject-level splits (same stratified logic as the primary
analysis, so seed 42 reproduces the primary split), we hold the tuned XGBoost
hyperparameters fixed and record:
  - mean 5-fold grouped CV AUC on the training partition
  - held-out test AUC, AUPRC, and Brier score

We then report the DISTRIBUTION of these metrics across splits, which is the
honest, split-robust summary of model performance. The percentile of the
primary-analysis value (0.932) within this distribution shows whether it was
a favourable draw.

Design note: hyperparameters were tuned once on the seed-42 training partition
and held fixed across all splits. This isolates test-draw variance for a fixed
model configuration (exactly the reviewer's concern about 0.932); it is a seed
sweep, not a full nested-CV re-tune, and is labelled as such in the paper.

Produces:
  results/repeated_splits.csv   -- per-split metrics
  results/repeated_splits.png   -- test-AUC distribution with primary marker

Usage:
    python -m src.experiments.repeated_splits --config configs/base.yaml --n-seeds 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.pipeline import make_pipeline
from src.splits import get_feature_cols, grouped_cv, subject_split
from src.train import augment_tabular

PRIMARY_SEED = 42


def _fit_predict(Xtr_feat, ytr, Xte_feat, params, seed):
    """Augment training data, fit the pipeline, return test-set probabilities."""
    Xtr_a, ytr_a = augment_tabular(Xtr_feat, ytr, seed=seed)
    pipe = make_pipeline(xgb.XGBClassifier(**params))
    pipe.fit(Xtr_a, ytr_a)
    return pipe.predict_proba(Xte_feat)[:, 1]


def _cv_auc(Xtr_feat, ytr, rids_tr, params, n_splits, seed):
    """Mean grouped-CV AUC on the training partition, fixed params."""
    cv = grouped_cv(n_splits=n_splits, seed=seed)
    aucs = []
    for tr, val in cv.split(Xtr_feat, ytr, groups=rids_tr):
        Xf = Xtr_feat.iloc[tr].reset_index(drop=True)
        yf = ytr.iloc[tr].reset_index(drop=True)
        Xv = Xtr_feat.iloc[val].reset_index(drop=True)
        yv = ytr.iloc[val].reset_index(drop=True)
        if yv.nunique() < 2:
            continue
        probs = _fit_predict(Xf, yf, Xv, params, seed)
        aucs.append(roc_auc_score(yv, probs))
    return float(np.mean(aucs)) if aucs else np.nan


def run(feature_csv: Path, results_dir: Path, n_seeds: int, n_splits: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the tuned hyperparameters from the primary run so the model
    # configuration is identical to the headline analysis.
    metrics_path = results_dir / "metrics.json"
    best_params = json.loads(metrics_path.read_text())["best_params"]
    print(f"Loaded tuned hyperparameters from {metrics_path}")

    df = pd.read_csv(feature_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)

    # X carries RID so we can recover training-partition groups for CV.
    X = df[[c for c in df.columns if c != "label_36mo"]]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]

    rows = []
    print(f"Running {n_seeds} repeated subject-level splits...")
    for seed in range(n_seeds):
        Xtr, Xte, ytr, yte = subject_split(X, y, rids, seed=seed)
        feat = get_feature_cols(Xtr)
        rids_tr = Xtr["RID"].reset_index(drop=True)

        Xtr_feat = Xtr[feat].reset_index(drop=True)
        Xte_feat = Xte[feat].reset_index(drop=True)
        ytr = ytr.reset_index(drop=True)
        yte = yte.reset_index(drop=True)

        if yte.nunique() < 2:
            continue

        cv_auc = _cv_auc(Xtr_feat, ytr, rids_tr, best_params, n_splits, seed)
        probs = _fit_predict(Xtr_feat, ytr, Xte_feat, best_params, seed)

        rows.append({
            "seed": seed,
            "n_test": int(len(yte)),
            "n_test_conv": int(yte.sum()),
            "cv_auc": cv_auc,
            "test_auc": roc_auc_score(yte, probs),
            "test_auprc": average_precision_score(yte, probs),
            "test_brier": brier_score_loss(yte, probs),
        })
        if (seed + 1) % 25 == 0:
            print(f"  {seed + 1}/{n_seeds} splits done")

    res = pd.DataFrame(rows)
    res.to_csv(results_dir / "repeated_splits.csv", index=False)

    # ---- Summary ----
    def summ(col):
        v = res[col].dropna()
        return v.mean(), v.std(), np.percentile(v, 2.5), np.percentile(v, 50), np.percentile(v, 97.5)

    print("\n" + "=" * 66)
    print(f"REPEATED-SPLIT STABILITY  (n = {len(res)} valid splits)")
    print("=" * 66)
    print(f"{'Metric':<12}{'mean':>8}{'SD':>8}{'2.5%':>8}{'median':>9}{'97.5%':>8}")
    for col, label in [("cv_auc", "CV AUC"), ("test_auc", "Test AUC"),
                       ("test_auprc", "Test AUPRC"), ("test_brier", "Test Brier")]:
        m, sd, lo, md, hi = summ(col)
        print(f"{label:<12}{m:>8.3f}{sd:>8.3f}{lo:>8.3f}{md:>9.3f}{hi:>8.3f}")

    primary = res[res["seed"] == PRIMARY_SEED]
    if len(primary):
        p_auc = float(primary["test_auc"].iloc[0])
        pct = float((res["test_auc"] < p_auc).mean() * 100)
        print("-" * 66)
        print(f"Primary split (seed {PRIMARY_SEED}) test AUC: {p_auc:.3f} "
              f"-> {pct:.0f}th percentile of the distribution")
    print("=" * 66)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(res["test_auc"].dropna(), bins=24, color="#1565C0", alpha=0.75, edgecolor="white")
    mean_auc = res["test_auc"].mean()
    lo, hi = np.percentile(res["test_auc"].dropna(), [2.5, 97.5])
    ax.axvline(mean_auc, color="#0D47A1", lw=2, label=f"Mean = {mean_auc:.3f}")
    ax.axvspan(lo, hi, color="#1565C0", alpha=0.08, label=f"95% range [{lo:.3f}, {hi:.3f}]")
    if len(primary):
        ax.axvline(p_auc, color="#C62828", lw=2, linestyle="--",
                   label=f"Primary split (seed 42) = {p_auc:.3f}")
    ax.set_xlabel("Held-out test AUC", fontsize=11)
    ax.set_ylabel("Number of splits", fontsize=11)
    ax.set_title(f"Test-AUC distribution across {len(res)} random subject-level splits\n"
                 "(tuned XGBoost held fixed; test n = 61 each)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / "repeated_splits.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_dir}/repeated_splits.csv, repeated_splits.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-seeds", type=int, default=200)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(
        feature_csv=Path(cfg["feature_csv"]),
        results_dir=Path(cfg.get("results_dir", "results")),
        n_seeds=args.n_seeds,
        n_splits=cfg.get("n_splits", 5),
    )
