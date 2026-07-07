"""
Regenerate the primary-analysis artifacts on the CORRECTED clean subject-level
split (seed 42), reusing the previously tuned XGBoost hyperparameters.

This exists because an index-alignment bug in the original train.py produced a
positional (non-stratified, seed-independent) tail-61 test set. The bug is now
fixed in train.py; this script refreshes the downstream artifacts without
re-running the 80-trial Optuna search (hyperparameters are held fixed).

Regenerates:
  results/metrics.json      -- corrected test metrics on the clean split
  results/calibration.png   -- reliability diagram on the clean test set
  results/dca.png, dca.csv  -- decision curve on the clean test set
  results/shap_summary.png  -- SHAP beeswarm on the clean test set
  results/shap_rankings.csv

Usage:
    python -m src.experiments.regen_primary --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

from src.pipeline import make_pipeline
from src.splits import get_feature_cols, subject_split
from src.train import augment_tabular
from src.evaluate import bootstrap_metric, evaluate, plot_calibration, decision_curve
from src.explain import run_shap


def run(feature_csv: Path, results_dir: Path, seed: int = 42) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    best_params = json.loads((results_dir / "metrics.json").read_text())["best_params"]

    df = pd.read_csv(feature_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)
    feat_cols = get_feature_cols(df)
    X = df[feat_cols + ["RID"]]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]

    X_train, X_test, y_train, y_test = subject_split(X, y, rids, seed=seed)
    X_train_feat = X_train[feat_cols].reset_index(drop=True)
    X_test_feat = X_test[feat_cols].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    Xa, ya = augment_tabular(X_train_feat, y_train, seed=seed)
    pipe = make_pipeline(xgb.XGBClassifier(**best_params))
    pipe.fit(Xa, ya)

    metrics = evaluate(pipe, X_test_feat, y_test, feat_cols)
    metrics["cv_auc"] = json.loads((results_dir / "metrics.json").read_text()).get("cv_auc")
    metrics["best_params"] = best_params
    metrics["split"] = "clean stratified subject-level split (seed 42), corrected"
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    probs = pipe.predict_proba(X_test_feat)[:, 1]
    plot_calibration(y_test, probs, metrics["brier"], results_dir)
    decision_curve(y_test, probs, results_dir)
    run_shap(pipe, X_test_feat, feat_cols, results_dir)
    print("\nRegenerated primary artifacts on corrected clean split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(
        feature_csv=Path(cfg["feature_csv"]),
        results_dir=Path(cfg.get("results_dir", "results")),
        seed=cfg.get("seed", 42),
    )
