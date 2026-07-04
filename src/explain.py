"""
SHAP explainability for the final XGBoost model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def run_shap(pipeline, X_test: pd.DataFrame, feat_cols: list[str], results_dir: Path) -> pd.DataFrame:
    # Extract XGBoost model from pipeline
    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]

    X_imp = pd.DataFrame(imputer.transform(X_test[feat_cols]), columns=feat_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    shap.summary_plot(shap_values, X_imp, show=False)
    plt.tight_layout()
    plt.savefig(results_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    ranking = pd.DataFrame({
        "feature": feat_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    ranking.to_csv(results_dir / "shap_rankings.csv", index=False)
    print("Top 10 SHAP features:")
    print(ranking.head(10).to_string(index=False))
    return ranking
