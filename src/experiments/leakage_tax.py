"""
Leakage-tax ablation — Priority 1 headline contribution.

Isolates the AUC inflation caused by each common leakage pattern,
using the same cohort and seed throughout so differences are causal.

Leak types tested:
  A) Augmentation before split  — noisy copies of test subjects appear in training
  B) Imputation on full data    — test statistics influence training imputation
  C) A + B combined             — both leaks active simultaneously

Clean baseline is the current pipeline (augment inside folds, impute inside folds,
subject-level split).

Each variant trains on the same X_train, evaluates on the same X_test.
The "tax" is (leaky CV AUC) − (clean test AUC): how much each leak flatters
the reported number.

Usage:
    python -m src.experiments.leakage_tax --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.evaluate import bootstrap_metric
from src.pipeline import make_pipeline
from src.splits import get_feature_cols, grouped_cv, subject_split
from src.train import augment_tabular

SEED = 42
N_SPLITS = 5

XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=SEED,
)


# ---------------------------------------------------------------------------
# Pipeline variants
# ---------------------------------------------------------------------------

def _clean_cv_and_test(X_train, y_train, rids_train, X_test, y_test, feat_cols):
    """
    Clean pipeline:
      - Augmentation inside each CV fold (train partition only)
      - Imputation inside each CV fold (fit on train, transform val)
      - Grouped CV keyed on RID
    """
    cv = grouped_cv(N_SPLITS, SEED)
    cv_aucs = []
    for tr, val in cv.split(X_train[feat_cols], y_train, groups=rids_train):
        Xtr = X_train[feat_cols].iloc[tr].reset_index(drop=True)
        ytr = y_train.iloc[tr].reset_index(drop=True)
        Xval = X_train[feat_cols].iloc[val].reset_index(drop=True)
        yval = y_train.iloc[val].reset_index(drop=True)
        Xtr_a, ytr_a = augment_tabular(Xtr, ytr, seed=SEED)
        pipe = make_pipeline(xgb.XGBClassifier(**XGB_PARAMS))
        pipe.fit(Xtr_a, ytr_a)
        cv_aucs.append(roc_auc_score(yval, pipe.predict_proba(Xval)[:, 1]))

    # Final model on full augmented train
    Xtr_full, ytr_full = augment_tabular(X_train[feat_cols], y_train, seed=SEED)
    pipe_final = make_pipeline(xgb.XGBClassifier(**XGB_PARAMS))
    pipe_final.fit(Xtr_full, ytr_full)
    test_probs = pipe_final.predict_proba(X_test[feat_cols])[:, 1]
    return np.mean(cv_aucs), test_probs


def _leak_augment_before_split(X_full, y_full, rids_full, feat_cols):
    """
    Leak A — Augment BEFORE split.
    Noisy copies of every subject (including future test subjects) are mixed
    into the pool before train/test separation, so the model sees corrupted
    versions of test subjects during training.
    """
    # Augment full dataset first
    X_aug, y_aug = augment_tabular(X_full[feat_cols], y_full, seed=SEED)
    # Append original RIDs (repeated for each aug copy)
    n_copies = len(X_aug) // len(X_full)
    rids_aug = pd.concat([rids_full] * n_copies, ignore_index=True)

    # Record-level split (random, ignoring subject grouping — the canonical leak)
    idx = np.arange(len(X_aug))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    Xtr = X_aug.iloc[train_idx].reset_index(drop=True)
    ytr = y_aug.iloc[train_idx].reset_index(drop=True)
    Xte = X_aug.iloc[test_idx].reset_index(drop=True)
    yte = y_aug.iloc[test_idx].reset_index(drop=True)

    # CV: StratifiedKFold (no group awareness — leaky)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_aucs = []
    for tr, val in cv.split(Xtr, ytr):
        imp = SimpleImputer(strategy="median")
        Xtr_i = imp.fit_transform(Xtr.iloc[tr])
        Xval_i = imp.transform(Xtr.iloc[val])
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr_i, ytr.iloc[tr])
        cv_aucs.append(roc_auc_score(ytr.iloc[val], m.predict_proba(Xval_i)[:, 1]))

    imp_final = SimpleImputer(strategy="median")
    Xtr_final = imp_final.fit_transform(Xtr)
    Xte_final = imp_final.transform(Xte)
    m_final = xgb.XGBClassifier(**XGB_PARAMS)
    m_final.fit(Xtr_final, ytr)
    test_probs = m_final.predict_proba(Xte_final)[:, 1]
    return np.mean(cv_aucs), test_probs, yte


def _leak_impute_on_full_data(X_train, y_train, rids_train, X_test, y_test, feat_cols):
    """
    Leak B — Imputer fit on ALL data (train + test) before CV.
    Test statistics contaminate training imputation.
    """
    X_all = pd.concat([X_train[feat_cols], X_test[feat_cols]], ignore_index=True)
    imp_global = SimpleImputer(strategy="median")
    imp_global.fit(X_all)  # fitted on train+test — the leak

    Xtr_imp = pd.DataFrame(imp_global.transform(X_train[feat_cols]), columns=feat_cols)
    Xte_imp = pd.DataFrame(imp_global.transform(X_test[feat_cols]), columns=feat_cols)

    # CV: grouped, but imputation already leaked
    cv = grouped_cv(N_SPLITS, SEED)
    cv_aucs = []
    for tr, val in cv.split(Xtr_imp, y_train, groups=rids_train):
        Xtr_f = Xtr_imp.iloc[tr].reset_index(drop=True)
        ytr = y_train.iloc[tr].reset_index(drop=True)
        Xval_f = Xtr_imp.iloc[val].reset_index(drop=True)
        yval = y_train.iloc[val].reset_index(drop=True)
        Xtr_a, ytr_a = augment_tabular(Xtr_f, ytr, seed=SEED)
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr_a.values, ytr_a.values)
        cv_aucs.append(roc_auc_score(yval, m.predict_proba(Xval_f.values)[:, 1]))

    Xtr_a, ytr_a = augment_tabular(Xtr_imp, y_train, seed=SEED)
    m_final = xgb.XGBClassifier(**XGB_PARAMS)
    m_final.fit(Xtr_a.values, ytr_a.values)
    test_probs = m_final.predict_proba(Xte_imp.values)[:, 1]
    return np.mean(cv_aucs), test_probs


def _leak_combined(X_full, y_full, rids_full, feat_cols):
    """
    Leak A + B — Augment before split AND impute on full data.
    Worst-case scenario, approximating the 'old notebook' behaviour.
    """
    # Impute on full data first
    imp_global = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imp_global.fit_transform(X_full[feat_cols]), columns=feat_cols
    )

    # Then augment full imputed dataset
    X_aug, y_aug = augment_tabular(X_imp, y_full, seed=SEED)
    n_orig = len(X_full)

    # Record-level random split
    idx = np.arange(len(X_aug))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_test = int(len(idx) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    Xtr = X_aug.iloc[train_idx].reset_index(drop=True)
    ytr = y_aug.iloc[train_idx].reset_index(drop=True)
    Xte = X_aug.iloc[test_idx].reset_index(drop=True)
    yte = y_aug.iloc[test_idx].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_aucs = []
    for tr, val in cv.split(Xtr, ytr):
        m = xgb.XGBClassifier(**XGB_PARAMS)
        m.fit(Xtr.iloc[tr].values, ytr.iloc[tr].values)
        cv_aucs.append(roc_auc_score(ytr.iloc[val], m.predict_proba(Xtr.iloc[val].values)[:, 1]))

    m_final = xgb.XGBClassifier(**XGB_PARAMS)
    m_final.fit(Xtr.values, ytr.values)
    test_probs = m_final.predict_proba(Xte.values)[:, 1]
    return np.mean(cv_aucs), test_probs, yte


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(feature_csv: Path, results_dir: Path, seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)
    feat_cols = get_feature_cols(df)
    # Carry RID inside X so training-partition groups can be recovered from the
    # (reset-index) split frames rather than the original df, which would give a
    # positional, non-stratified split.
    X = df[feat_cols + ["RID"]]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]

    # One fixed subject-level split — all variants use the same hold-out
    X_train, X_test, y_train, y_test = subject_split(X, y, rids, seed=seed)
    rids_train = X_train["RID"].reset_index(drop=True)
    X_train = X_train[feat_cols].reset_index(drop=True)
    X_test = X_test[feat_cols].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print("=" * 60)
    print("LEAKAGE-TAX ABLATION")
    print("=" * 60)

    rows = []

    # --- Clean ---------------------------------------------------------------
    print("\n[1/4] Clean pipeline...")
    cv_auc_clean, probs_clean = _clean_cv_and_test(
        X_train, y_train, rids_train, X_test, y_test, feat_cols
    )
    test_auc_clean = roc_auc_score(y_test, probs_clean)
    auprc_clean = average_precision_score(y_test, probs_clean)
    brier_clean = brier_score_loss(y_test, probs_clean)
    print(f"  CV AUC: {cv_auc_clean:.3f} | Test AUC: {test_auc_clean:.3f} | AUPRC: {auprc_clean:.3f} | Brier: {brier_clean:.3f}")
    rows.append({
        "Pipeline": "Clean (subject-split, augment inside folds, impute inside folds)",
        "CV AUC": cv_auc_clean,
        "Test AUC": test_auc_clean,
        "AUPRC": auprc_clean,
        "Brier": brier_clean,
        "AUC inflation (CV − test)": cv_auc_clean - test_auc_clean,
    })

    # --- Leak A: augment before split ----------------------------------------
    print("\n[2/4] Leak A — augment before split (record-level split)...")
    X_full_df = df[feat_cols]
    y_full = df["label_36mo"].astype(int)
    rids_full = df["RID"]
    cv_auc_a, probs_a, y_test_a = _leak_augment_before_split(X_full_df, y_full, rids_full, feat_cols)
    test_auc_a = roc_auc_score(y_test_a, probs_a)
    auprc_a = average_precision_score(y_test_a, probs_a)
    brier_a = brier_score_loss(y_test_a, probs_a)
    print(f"  CV AUC: {cv_auc_a:.3f} | Test AUC: {test_auc_a:.3f} | AUPRC: {auprc_a:.3f} | Brier: {brier_a:.3f}")
    rows.append({
        "Pipeline": "Leak A: augment before split + record-level split",
        "CV AUC": cv_auc_a,
        "Test AUC": test_auc_a,
        "AUPRC": auprc_a,
        "Brier": brier_a,
        "AUC inflation (CV − test)": cv_auc_a - test_auc_a,
    })

    # --- Leak B: impute on full data ------------------------------------------
    print("\n[3/4] Leak B — impute on full data...")
    cv_auc_b, probs_b = _leak_impute_on_full_data(
        X_train, y_train, rids_train, X_test, y_test, feat_cols
    )
    test_auc_b = roc_auc_score(y_test, probs_b)
    auprc_b = average_precision_score(y_test, probs_b)
    brier_b = brier_score_loss(y_test, probs_b)
    print(f"  CV AUC: {cv_auc_b:.3f} | Test AUC: {test_auc_b:.3f} | AUPRC: {auprc_b:.3f} | Brier: {brier_b:.3f}")
    rows.append({
        "Pipeline": "Leak B: impute on train+test before CV",
        "CV AUC": cv_auc_b,
        "Test AUC": test_auc_b,
        "AUPRC": auprc_b,
        "Brier": brier_b,
        "AUC inflation (CV − test)": cv_auc_b - test_auc_b,
    })

    # --- Leak A + B combined -------------------------------------------------
    print("\n[4/4] Leak A + B combined (worst case)...")
    cv_auc_ab, probs_ab, y_test_ab = _leak_combined(X_full_df, y_full, rids_full, feat_cols)
    test_auc_ab = roc_auc_score(y_test_ab, probs_ab)
    auprc_ab = average_precision_score(y_test_ab, probs_ab)
    brier_ab = brier_score_loss(y_test_ab, probs_ab)
    print(f"  CV AUC: {cv_auc_ab:.3f} | Test AUC: {test_auc_ab:.3f} | AUPRC: {auprc_ab:.3f} | Brier: {brier_ab:.3f}")
    rows.append({
        "Pipeline": "Leak A+B: augment before split + impute on full data",
        "CV AUC": cv_auc_ab,
        "Test AUC": test_auc_ab,
        "AUPRC": auprc_ab,
        "Brier": brier_ab,
        "AUC inflation (CV − test)": cv_auc_ab - test_auc_ab,
    })

    results = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "leakage_tax.csv", index=False)

    _print_table(results, test_auc_clean)
    _plot(results, results_dir)
    print(f"\nSaved: {results_dir}/leakage_tax.csv + leakage_tax.png")
    return results


def _print_table(results: pd.DataFrame, clean_test_auc: float) -> None:
    print("\n" + "=" * 60)
    print("LEAKAGE TAX SUMMARY")
    print("=" * 60)
    for _, row in results.iterrows():
        tax = row["Test AUC"] - clean_test_auc
        sign = f"+{tax:.3f}" if tax > 0 else f"{tax:.3f}"
        print(f"\n  {row['Pipeline']}")
        print(f"    CV AUC:  {row['CV AUC']:.3f}  |  Test AUC: {row['Test AUC']:.3f}  ({sign} vs clean)")
        print(f"    AUPRC:   {row['AUPRC']:.3f}   |  Brier:    {row['Brier']:.3f}")
    print()


def _plot(results: pd.DataFrame, results_dir: Path) -> None:
    labels = [r.split(":")[0] if ":" in r else r for r in results["Pipeline"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Leakage-Tax Ablation\n"
                 "(each bar = same cohort + seed; difference is causal)",
                 fontsize=13, fontweight="bold")

    for ax, col, title in [
        (axes[0], "CV AUC", "Reported CV AUC (what a leaky paper would claim)"),
        (axes[1], "Test AUC", "Honest Test AUC (unseen hold-out)"),
    ]:
        colors = ["#2E7D32" if i == 0 else "#C62828" for i in range(len(labels))]
        bars = ax.bar(x, results[col], color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("AUC")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ymin = max(0.5, results[col].min() - 0.05)
        ax.set_ylim(ymin, min(1.0, results[col].max() + 0.08))
        for bar, val in zip(bars, results[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if col == "Test AUC":
            clean_val = results[col].iloc[0]
            ax.axhline(clean_val, color="#2E7D32", linestyle="--", linewidth=1.5,
                       label=f"Clean baseline ({clean_val:.3f})")
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "leakage_tax.png", dpi=150, bbox_inches="tight")
    plt.close()


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
