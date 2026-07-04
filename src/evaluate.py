"""
Evaluation: AUC, AUPRC, Brier, calibration curves, decision curve analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def bootstrap_metric(y_true, y_score, metric_fn, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series, feat_cols: list[str]) -> dict:
    probs = pipeline.predict_proba(X_test[feat_cols])[:, 1]

    auc = roc_auc_score(y_test, probs)
    auc_m, auc_lo, auc_hi = bootstrap_metric(y_test, probs, roc_auc_score)
    ap_m, ap_lo, ap_hi = bootstrap_metric(y_test, probs, average_precision_score)
    brier = brier_score_loss(y_test, probs)

    print("=" * 58)
    print("TEST SET RESULTS")
    print("=" * 58)
    print(f"  AUC:   {auc:.3f}  [95% CI {auc_lo:.3f}–{auc_hi:.3f}]")
    print(f"  AUPRC: {ap_m:.3f}  [95% CI {ap_lo:.3f}–{ap_hi:.3f}]")
    print(f"  Brier: {brier:.3f}")
    print(f"  n = {len(y_test)} ({int(y_test.sum())} converters)")
    print("=" * 58)

    return {
        "auc": auc,
        "auc_ci": [auc_lo, auc_hi],
        "auprc": ap_m,
        "auprc_ci": [ap_lo, ap_hi],
        "brier": brier,
        "n_test": len(y_test),
        "n_converters": int(y_test.sum()),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def plot_calibration(y_true, probs, brier: float, results_dir: Path, n_bins: int = 8) -> None:
    """Reliability diagram with histogram of predicted probabilities."""
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k:", linewidth=1.5, label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, "o-", color="#1565C0", linewidth=2,
             markersize=7, label=f"XGBoost (Brier = {brier:.3f})")
    ax1.fill_between(mean_pred, frac_pos,
                     np.interp(mean_pred, [0, 1], [0, 1]),
                     alpha=0.08, color="#1565C0")
    ax1.set_ylabel("Observed conversion fraction", fontsize=11)
    ax1.set_title("Calibration — 36-month MCI→AD conversion", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    ax2.hist(probs, bins=20, color="#1565C0", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_dir}/calibration.png")


# ---------------------------------------------------------------------------
# Decision curve analysis
# ---------------------------------------------------------------------------

def _net_benefit(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    """Net benefit at a single decision threshold."""
    n = len(y_true)
    predicted_pos = probs >= threshold
    tp = int(((predicted_pos) & (y_true == 1)).sum())
    fp = int(((predicted_pos) & (y_true == 0)).sum())
    odds = threshold / (1 - threshold) if threshold < 1 else np.inf
    return tp / n - fp / n * odds


def decision_curve(
    y_true,
    probs,
    results_dir: Path,
    threshold_range: tuple[float, float] = (0.05, 0.85),
    n_points: int = 200,
) -> pd.DataFrame:
    """
    Decision curve analysis: net benefit of the model vs treat-all and treat-none
    across a range of decision thresholds.

    Net benefit = TP/n − FP/n × (threshold / (1 − threshold))

    A model is clinically useful at threshold t if its net benefit exceeds
    both treat-all (everyone gets intervention) and treat-none (zero).
    """
    y = np.asarray(y_true)
    p = np.asarray(probs)
    thresholds = np.linspace(*threshold_range, n_points)
    prevalence = y.mean()

    nb_model = np.array([_net_benefit(y, p, t) for t in thresholds])
    nb_all = np.array([prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds])
    nb_none = np.zeros(n_points)

    # Clip treat-all at 0 where it goes negative (no benefit)
    nb_all = np.clip(nb_all, -0.02, None)

    dca_df = pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": nb_none,
    })
    dca_df.to_csv(results_dir / "dca.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, nb_model, color="#1565C0", linewidth=2.5, label="XGBoost model")
    ax.plot(thresholds, nb_all, color="#EF6C00", linewidth=2, linestyle="--", label="Treat all")
    ax.plot(thresholds, nb_none, color="gray", linewidth=1.5, linestyle=":", label="Treat none")
    ax.fill_between(thresholds,
                    np.maximum(nb_model, np.maximum(nb_all, nb_none)),
                    nb_model,
                    where=(nb_model >= np.maximum(nb_all, nb_none)),
                    alpha=0.12, color="#1565C0", label="Model advantage")
    ax.set_xlabel("Decision threshold (predicted probability)", fontsize=11)
    ax.set_ylabel("Net benefit", fontsize=11)
    ax.set_title("Decision Curve Analysis — 36-month MCI→AD conversion\n"
                 "(model is useful where blue line exceeds orange and gray)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(*threshold_range)
    ax.set_ylim(-0.05, max(nb_model.max(), nb_all.max()) + 0.05)

    # Annotate clinical range
    ax.axvspan(0.15, 0.40, alpha=0.04, color="green",
               label="Typical clinical action range")
    ax.text(0.275, ax.get_ylim()[1] * 0.92, "Typical\nclinical range",
            ha="center", va="top", fontsize=8, color="darkgreen", style="italic")

    plt.tight_layout()
    plt.savefig(results_dir / "dca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {results_dir}/dca.png, dca.csv")

    # Report where model beats treat-all
    model_wins = dca_df[dca_df["net_benefit_model"] > dca_df["net_benefit_treat_all"]]
    if len(model_wins):
        print(f"  Model beats treat-all from threshold "
              f"{model_wins['threshold'].min():.2f} to {model_wins['threshold'].max():.2f}")
    else:
        print("  Model does not beat treat-all at any threshold — check calibration.")

    return dca_df
