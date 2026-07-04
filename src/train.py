"""
Optuna hyperparameter search with leak-free grouped CV + tabular augmentation.

Usage:
    python -m src.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score

from src.pipeline import make_pipeline
from src.splits import get_feature_cols, grouped_cv, subject_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
NOISE_SCALE = 0.015
MISS_RATE = 0.05
AUG_COPIES = 2


# ---------------------------------------------------------------------------
# Augmentation (training fold only)
# ---------------------------------------------------------------------------

def _augment_noise(X: pd.DataFrame, noise_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xa = X.copy()
    for col in Xa.columns:
        if Xa[col].nunique(dropna=True) <= 5:
            continue
        s = Xa[col].std(skipna=True)
        if s == 0 or np.isnan(s):
            continue
        noise = rng.normal(0.0, noise_scale * s, size=len(Xa))
        obs = Xa[col].notna()
        Xa.loc[obs, col] += noise[obs]
    return Xa


def _augment_missingness(X: pd.DataFrame, miss_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = X.values.astype(float)
    mask = rng.random(arr.shape) < miss_rate
    arr[mask & ~np.isnan(arr)] = np.nan
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def augment_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    n_copies: int = AUG_COPIES,
    noise_scale: float = NOISE_SCALE,
    miss_rate: float = MISS_RATE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Only ever called on the training partition."""
    Xs, ys = [X], [y]
    for i in range(n_copies):
        s = seed + i * 1000
        Xc = _augment_noise(X.copy(), noise_scale, s)
        Xc = _augment_missingness(Xc, miss_rate, s + 1)
        Xs.append(Xc)
        ys.append(y)
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rids_train: pd.Series,
    n_splits: int,
    seed: int,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "eval_metric": "logloss",
        "random_state": seed,
    }
    cv = grouped_cv(n_splits=n_splits, seed=seed)
    aucs = []
    feat_cols = get_feature_cols(X_train)
    X_feat = X_train[feat_cols]

    for tr, val in cv.split(X_feat, y_train, groups=rids_train):
        Xtr = X_feat.iloc[tr].reset_index(drop=True)
        ytr = y_train.iloc[tr].reset_index(drop=True)
        Xval = X_feat.iloc[val].reset_index(drop=True)
        yval = y_train.iloc[val].reset_index(drop=True)

        Xtr_a, ytr_a = augment_tabular(Xtr, ytr, seed=seed)

        pipe = make_pipeline(xgb.XGBClassifier(**params))
        pipe.fit(Xtr_a, ytr_a)
        aucs.append(roc_auc_score(yval, pipe.predict_proba(Xval)[:, 1]))

    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Main train entry point
# ---------------------------------------------------------------------------

def train(
    feature_csv: Path,
    n_trials: int = 80,
    n_splits: int = 5,
    seed: int = SEED,
    results_dir: Path = Path("results"),
) -> dict:
    """
    Load features, split, tune, train final model.
    Returns dict of test metrics and best params.
    """
    df = pd.read_csv(feature_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)

    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]

    X_train_df, X_test_df, y_train, y_test = subject_split(X, y, rids, seed=seed)
    rids_train = rids[rids.isin(
        set(rids) - set(rids[~rids.isin(X_train_df.index)])  # same partition
    )].reset_index(drop=True)

    # Rebuild rids_train aligned with the split
    train_mask = rids.isin(set(rids.unique()) - set(
        rids[~y.index.isin(y_train.index)]
    ))

    # Simpler: attach RID back to the split DataFrames before stripping it
    df_train = df[df["RID"].isin(set(rids) - set(
        df.loc[~df.index.isin(X_train_df.index), "RID"]
    ))].reset_index(drop=True)

    rids_train_s = df_train["RID"]
    X_train_feat = df_train[feat_cols]
    y_train_s = df_train["label_36mo"].astype(int)

    df_test = df[~df["RID"].isin(set(df_train["RID"]))].reset_index(drop=True)
    X_test_feat = df_test[feat_cols]
    y_test_s = df_test["label_36mo"].astype(int)

    print(f"Train: {len(y_train_s)} | Test: {len(y_test_s)}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective(t, X_train_feat, y_train_s, rids_train_s, n_splits, seed),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = {**study.best_params, "eval_metric": "logloss", "random_state": seed}
    print(f"CV AUC: {study.best_value:.4f}")

    # Final model on full augmented training set
    Xtr_a, ytr_a = augment_tabular(X_train_feat, y_train_s, seed=seed)
    final_pipe = make_pipeline(xgb.XGBClassifier(**best_params))
    final_pipe.fit(Xtr_a, ytr_a)

    from src.evaluate import evaluate
    metrics = evaluate(final_pipe, X_test_feat, y_test_s, feat_cols)
    metrics["cv_auc"] = study.best_value
    metrics["best_params"] = best_params

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test AUC: {metrics['auc']:.3f}  AUPRC: {metrics['auprc']:.3f}")

    return metrics, final_pipe, feat_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train(
        feature_csv=Path(cfg["feature_csv"]),
        n_trials=cfg.get("n_trials", 80),
        n_splits=cfg.get("n_splits", 5),
        seed=cfg.get("seed", 42),
        results_dir=Path(cfg.get("results_dir", "results")),
    )
