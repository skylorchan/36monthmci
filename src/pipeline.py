"""
sklearn / imblearn Pipeline factory.

Preprocessing order: Impute → (Scale) → (Resample) → Model.
All steps are fit on the training fold only. The Pipeline wrapper enforces this
structurally so it cannot be broken by manual mistakes.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def make_pipeline(model, *, scale: bool = False, resampler=None) -> Pipeline:
    """
    Parameters
    ----------
    model      : fitted sklearn estimator
    scale      : whether to add StandardScaler (XGBoost doesn't need it)
    resampler  : imblearn resampler to insert after imputation, or None
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if resampler is not None:
        steps.append(("resampler", resampler))
    steps.append(("model", model))
    return Pipeline(steps)
