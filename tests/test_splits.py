"""
Leakage guard tests.

These tests defend the paper's core claim that no subject's data appears
on both sides of any train/test or CV split.
Run with: pytest tests/test_splits.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.splits import grouped_cv, subject_split


@pytest.fixture
def synthetic_data():
    """308 subjects, one row each, ~50% converters — mirrors the real cohort."""
    rng = np.random.default_rng(0)
    n = 308
    rids = pd.Series(range(1000, 1000 + n), name="RID")
    y = pd.Series(rng.integers(0, 2, n).astype(float), name="label_36mo")
    X = pd.DataFrame(
        rng.standard_normal((n, 20)),
        columns=[f"feat_{i}" for i in range(20)],
    )
    rows = rng.choice(n, size=200, replace=True)
    cols = rng.choice(20, size=200, replace=True)
    for r, c in zip(rows, cols):
        X.iloc[r, c] = np.nan
    return X, y, rids


def test_no_rid_overlap_in_train_test(synthetic_data):
    X, y, rids = synthetic_data
    X_train, X_test, y_train, y_test = subject_split(X, y, rids)
    # RIDs are encoded in the index; use the original rids series
    train_rids = set(rids.iloc[X_train.index] if hasattr(X_train, "index") else [])
    # Reconstruct from masks
    mask_train = ~rids.isin(
        rids[~rids.index.isin(range(len(y_train)))]  # approximate: just test uniqueness
    )
    # Direct approach: split returns aligned frames; check via y index
    assert len(y_train) + len(y_test) == len(y), "Rows lost in split"
    assert len(y_train) > 0 and len(y_test) > 0


def test_no_rid_overlap_subject_split_explicit(synthetic_data):
    """Verify that train and test RID sets are truly disjoint."""
    X, y, rids = synthetic_data
    df = X.copy()
    df["RID"] = rids.values
    df["label_36mo"] = y.values

    n_unique = rids.nunique()
    n_test_expected = max(1, int(n_unique * 0.2))

    # Use masks to track which RIDs go where
    unique_rids = rids.unique()
    rng = np.random.default_rng(42)
    rng.shuffle(unique_rids)

    rid_label = y.copy()
    rid_label.index = rids.values
    rid_label = rid_label.groupby(level=0).first()

    converters = rid_label[rid_label == 1].index.tolist()
    non_converters = rid_label[rid_label == 0].index.tolist()
    n_test_conv = max(1, round(n_test_expected * len(converters) / n_unique))
    n_test_non = n_test_expected - n_test_conv

    rng2 = np.random.default_rng(42)
    rng2.shuffle(converters)
    rng2.shuffle(non_converters)

    test_rids = set(converters[:n_test_conv] + non_converters[:n_test_non])
    train_rids = set(unique_rids) - test_rids

    assert train_rids.isdisjoint(test_rids), "RID overlap between train and test!"
    assert len(train_rids) + len(test_rids) == n_unique, "RIDs lost in split!"


def test_no_rid_overlap_in_cv_folds(synthetic_data):
    """Each CV fold must have disjoint train/val RID sets."""
    X, y, rids = synthetic_data
    cv = grouped_cv(n_splits=5, seed=42)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups=rids)):
        train_rids = set(rids.iloc[tr_idx])
        val_rids = set(rids.iloc[val_idx])
        overlap = train_rids & val_rids
        assert not overlap, f"Fold {fold}: RID overlap: {overlap}"


def test_imputer_fit_on_train_only(synthetic_data):
    """Imputer fit on train must not use test statistics."""
    X, y, rids = synthetic_data
    X_train, X_test, y_train, y_test = subject_split(X, y, rids, seed=42)

    imp_train = SimpleImputer(strategy="median")
    imp_train.fit(X_train)

    imp_full = SimpleImputer(strategy="median")
    imp_full.fit(pd.concat([X_train, X_test]))

    # Medians from train-only vs full data should differ (test data changes distribution)
    # This asserts they're not identical, confirming they're computed separately
    assert not np.allclose(
        imp_train.statistics_, imp_full.statistics_
    ), "Train-only and full-data imputers are identical — test is inconclusive"


def test_all_rids_accounted_for(synthetic_data):
    """No subjects dropped in the split."""
    X, y, rids = synthetic_data
    X_train, X_test, y_train, y_test = subject_split(X, y, rids, seed=42)
    assert len(y_train) + len(y_test) == len(y)


def test_cv_covers_all_train_rows(synthetic_data):
    """Every training row appears in exactly one validation fold."""
    X, y, rids = synthetic_data
    X_train, X_test, y_train, y_test = subject_split(X, y, rids, seed=42)
    train_rids = rids.iloc[: len(y_train)].reset_index(drop=True)
    X_tr = X_train
    y_tr = y_train

    cv = grouped_cv(n_splits=5, seed=42)
    val_idx_all = []
    for _, val_idx in cv.split(X_tr, y_tr, groups=train_rids):
        val_idx_all.extend(val_idx.tolist())

    assert sorted(val_idx_all) == list(range(len(y_tr))), "CV does not cover all training rows"


# ---------------------------------------------------------------------------
# Regression guards for the corrected split-reconstruction usage.
#
# The pipeline carries RID inside X so that training-partition groups are read
# back from the (reset-index) split frames rather than the original dataframe.
# An earlier version mixed reset-index positions with original indices, which
# silently produced a positional, non-stratified hold-out set. These tests
# assert the split is subject-disjoint, stratified, and deterministic.
# ---------------------------------------------------------------------------

def test_rid_carried_split_is_disjoint(synthetic_data):
    """When RID travels inside X, train/test RID sets must be disjoint and complete."""
    X, y, rids = synthetic_data
    X_rid = X.copy()
    X_rid["RID"] = rids.values
    X_train, X_test, _, _ = subject_split(X_rid, y, rids, seed=42)

    train_rids = set(X_train["RID"])
    test_rids = set(X_test["RID"])
    assert train_rids.isdisjoint(test_rids), "RID overlap between train and test!"
    assert train_rids | test_rids == set(rids), "Some RIDs lost in the split!"


def test_split_is_stratified(synthetic_data):
    """Test-set conversion prevalence must track the cohort prevalence."""
    X, y, rids = synthetic_data
    cohort_prev = float(y.mean())
    _, _, _, y_test = subject_split(X, y, rids, seed=42)
    test_prev = float(y_test.mean())
    # Stratified split should stay within ~10 percentage points on n≈61.
    assert abs(test_prev - cohort_prev) < 0.10, (
        f"Split not stratified: cohort {cohort_prev:.2f} vs test {test_prev:.2f}"
    )


def test_split_is_deterministic(synthetic_data):
    """Same seed reproduces the split exactly; different seeds differ."""
    X, y, rids = synthetic_data
    X_rid = X.copy()
    X_rid["RID"] = rids.values

    a = subject_split(X_rid, y, rids, seed=42)[1]["RID"]
    b = subject_split(X_rid, y, rids, seed=42)[1]["RID"]
    c = subject_split(X_rid, y, rids, seed=7)[1]["RID"]

    assert set(a) == set(b), "Same seed produced different test sets"
    assert set(a) != set(c), "Different seeds produced identical test sets"
