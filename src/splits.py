"""
Subject-level train/test split and grouped CV factories.

INVARIANT: A subject's RID must never appear in both the train and test partitions.
This is enforced structurally here and verified in tests/test_splits.py.

Current dataset has one row per subject so StratifiedGroupKFold is equivalent
to StratifiedKFold, but we use the group-aware versions so the code is correct
if multi-visit rows are ever added.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def subject_split(
    X: pd.DataFrame,
    y: pd.Series,
    rids: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Subject-level train/test split.

    Returns X_train, X_test, y_train, y_test.
    The RID column is NOT included in the returned feature frames.
    """
    unique_rids = rids.unique()
    n_test = max(1, int(len(unique_rids) * test_size))

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_rids)

    # Stratify on subject-level label
    rid_label = pd.Series(
        y.values, index=rids.values
    ).groupby(level=0).first()

    converters = rid_label[rid_label == 1].index.tolist()
    non_converters = rid_label[rid_label == 0].index.tolist()

    n_test_conv = max(1, round(n_test * len(converters) / len(unique_rids)))
    n_test_non = n_test - n_test_conv

    rng2 = np.random.default_rng(seed)
    rng2.shuffle(converters)
    rng2.shuffle(non_converters)

    test_rids = set(converters[:n_test_conv] + non_converters[:n_test_non])
    train_rids = set(unique_rids) - test_rids

    assert train_rids.isdisjoint(test_rids), "RID overlap detected in split!"

    mask_train = rids.isin(train_rids)
    mask_test = rids.isin(test_rids)

    return (
        X[mask_train].reset_index(drop=True),
        X[mask_test].reset_index(drop=True),
        y[mask_train].reset_index(drop=True),
        y[mask_test].reset_index(drop=True),
    )


def grouped_cv(n_splits: int = 5, seed: int = 42) -> StratifiedGroupKFold:
    """Return a StratifiedGroupKFold splitter keyed on RID groups."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def get_feature_cols(X: pd.DataFrame) -> list[str]:
    """Return columns that are model inputs (exclude RID and label)."""
    return [c for c in X.columns if c not in ("RID", "label_36mo")]
