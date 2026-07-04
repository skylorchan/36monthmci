"""
Missing-modality robustness benchmark.

Systematically ablates each expensive modality and compares:
  - Native-NaN XGBoost (no imputation, XGB handles NaN natively)
  - Imputed XGBoost (median imputation inside CV folds)

Answers the clinical question: "Do I need to order PET / CSF / MRI?"

Usage:
    python -m src.experiments.missing_modality --config configs/base.yaml
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
from sklearn.metrics import average_precision_score, roc_auc_score

from src.pipeline import make_pipeline
from src.splits import get_feature_cols, grouped_cv, subject_split
from src.train import augment_tabular

SEED = 42

# ---------------------------------------------------------------------------
# Modality definitions
# ---------------------------------------------------------------------------

MODALITY_FEATURES: dict[str, list[str]] = {
    "Cognition": [
        "CDRSB", "MMSCORE", "ADAS13",
        "DELTA_CDRSB", "DELTA_MMSCORE", "DELTA_ADAS",
    ],
    "MRI": [
        "HIPPO_L", "HIPPO_R", "ENTORH_L", "ENTORH_R",
        "VENTRICLES", "ICV",
        "HIPPO_L_norm", "HIPPO_R_norm", "VENTRICLES_norm",
        "DELTA_HIPPO_L_norm", "DELTA_HIPPO_R_norm",
    ],
    "Amyloid PET": ["SUMMARY_SUVR"],
    "CSF": ["ABETA42", "TAU", "PTAU"],
    "Demographics + APOE": ["PTGENDER", "PTEDUCAT", "AGE", "APOE4", "PTETHCAT", "PTRACCAT"],
}

# Ablation configs: name → set of modalities to KEEP
ABLATIONS: dict[str, list[str]] = {
    "Full model":          list(MODALITY_FEATURES.keys()),
    "Drop Amyloid PET":    ["Cognition", "MRI", "CSF", "Demographics + APOE"],
    "Drop CSF":            ["Cognition", "MRI", "Amyloid PET", "Demographics + APOE"],
    "Drop MRI":            ["Cognition", "Amyloid PET", "CSF", "Demographics + APOE"],
    "Drop MRI + PET + CSF (cognition + demo)": ["Cognition", "Demographics + APOE"],
    "Cognition only":      ["Cognition"],
}


def _cols_for_ablation(all_feat_cols: list[str], keep_modalities: list[str]) -> list[str]:
    """Return the feature columns (and their MISSING_ masks) for the kept modalities."""
    keep_base: set[str] = set()
    for mod in keep_modalities:
        keep_base.update(MODALITY_FEATURES.get(mod, []))

    result = []
    for col in all_feat_cols:
        if col.startswith("MISSING_"):
            base = col[len("MISSING_"):]
            if base in keep_base:
                result.append(col)
        elif col in keep_base:
            result.append(col)
    return result


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def _cv_auc(
    X: pd.DataFrame,
    y: pd.Series,
    rids: pd.Series,
    feat_cols: list[str],
    impute: bool,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple[float, float, float, float]:
    """
    5-fold grouped CV. Returns (mean_auc, std_auc, mean_auprc, std_auprc).
    impute=False uses XGBoost's native NaN handling (no imputer step).
    """
    cv = grouped_cv(n_splits=n_splits, seed=seed)
    aucs, auprcs = [], []

    for tr, val in cv.split(X[feat_cols], y, groups=rids):
        Xtr = X[feat_cols].iloc[tr].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        Xval = X[feat_cols].iloc[val].reset_index(drop=True)
        yval = y.iloc[val].reset_index(drop=True)

        Xtr_a, ytr_a = augment_tabular(Xtr, ytr, seed=seed)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed,
        )

        if impute:
            pipe = make_pipeline(model)
            pipe.fit(Xtr_a, ytr_a)
            probs = pipe.predict_proba(Xval)[:, 1]
        else:
            model.fit(Xtr_a.values, ytr_a.values)
            probs = model.predict_proba(Xval.values)[:, 1]

        aucs.append(roc_auc_score(yval, probs))
        auprcs.append(average_precision_score(yval, probs))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(auprcs)), float(np.std(auprcs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(feature_csv: Path, results_dir: Path, n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(feature_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)

    all_feat_cols = get_feature_cols(df)
    X = df[all_feat_cols]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]

    # Use full training set (no held-out test — this is an ablation study, not final eval)
    X_train, _, y_train, _ = subject_split(X, y, rids, seed=seed)
    rids_train = rids.iloc[: len(y_train)].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    rows = []
    for name, keep_mods in ABLATIONS.items():
        feat_cols = _cols_for_ablation(all_feat_cols, keep_mods)
        # Only keep columns that actually exist in the data
        feat_cols = [c for c in feat_cols if c in df.columns]
        n_feats = len([c for c in feat_cols if not c.startswith("MISSING_")])

        print(f"\n{name}  ({n_feats} base features)")

        # Native NaN (XGBoost handles missingness natively)
        auc_nat, std_nat, ap_nat, std_ap_nat = _cv_auc(
            X_train, y_train, rids_train, feat_cols, impute=False, n_splits=n_splits, seed=seed
        )
        print(f"  Native NaN : AUC {auc_nat:.3f}±{std_nat:.3f}  AUPRC {ap_nat:.3f}±{std_ap_nat:.3f}")

        # Imputed
        auc_imp, std_imp, ap_imp, std_ap_imp = _cv_auc(
            X_train, y_train, rids_train, feat_cols, impute=True, n_splits=n_splits, seed=seed
        )
        print(f"  Imputed    : AUC {auc_imp:.3f}±{std_imp:.3f}  AUPRC {ap_imp:.3f}±{std_ap_imp:.3f}")

        rows.append({
            "Ablation": name,
            "N features": n_feats,
            "AUC (native)": auc_nat,
            "AUC std (native)": std_nat,
            "AUC (imputed)": auc_imp,
            "AUC std (imputed)": std_imp,
            "AUPRC (native)": ap_nat,
            "AUPRC (imputed)": ap_imp,
        })

    results = pd.DataFrame(rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "missing_modality.csv", index=False)

    _plot(results, results_dir)
    print(f"\nSaved results to {results_dir}/missing_modality.csv + missing_modality.png")
    return results


def _plot(results: pd.DataFrame, results_dir: Path) -> None:
    labels = results["Ablation"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Missing-Modality Robustness Benchmark\n(5-fold grouped CV on training set)",
                 fontsize=13, fontweight="bold")

    for ax, metric, metric_std, title in [
        (axes[0], "AUC (native)", "AUC std (native)", "AUC"),
        (axes[1], "AUPRC (native)", None, "AUPRC"),
    ]:
        vals_nat = results[metric].values
        vals_imp = results[f"{metric.replace('(native)', '(imputed)').strip()}"].values

        bars1 = ax.bar(x - width / 2, vals_nat, width, label="Native NaN", color="#1565C0", alpha=0.85)
        bars2 = ax.bar(x + width / 2, vals_imp, width, label="Imputed", color="#EF6C00", alpha=0.85)

        if metric_std and metric_std in results.columns:
            ax.errorbar(x - width / 2, vals_nat, yerr=results[metric_std].values,
                        fmt="none", color="black", capsize=4, linewidth=1.2)
            ax.errorbar(x + width / 2, vals_imp, yerr=results["AUC std (imputed)"].values,
                        fmt="none", color="black", capsize=4, linewidth=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by modality ablation")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ymin = max(0.5, min(vals_nat.min(), vals_imp.min()) - 0.05)
        ax.set_ylim(ymin, 1.0)

    plt.tight_layout()
    plt.savefig(results_dir / "missing_modality.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(
        feature_csv=Path(cfg["feature_csv"]),
        results_dir=Path(cfg.get("results_dir", "results")),
        n_splits=cfg.get("n_splits", 5),
        seed=cfg.get("seed", 42),
    )
