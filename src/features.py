"""
Feature engineering: baseline multimodal vectors + 12-month slope features.

All features are derived from index-visit (t=0) data or the 9–15 month window
for slopes. Nothing from after month 15 enters the feature vector.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import find_col, pick_file, read_csv, standardize_viscode, viscode_to_months

SLOPE_WINDOW = (9.0, 15.0)  # months


# ---------------------------------------------------------------------------
# Per-modality helpers
# ---------------------------------------------------------------------------

def _baseline(df: pd.DataFrame, rid_set: set, value_col: str) -> pd.DataFrame:
    df = df[df["RID"].isin(rid_set)].copy()
    df = standardize_viscode(df)
    df["months"] = df["VISCODE_STD"].apply(viscode_to_months)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return (
        df[df[value_col].notna()]
        .sort_values("months")
        .groupby("RID", as_index=False)
        .first()[["RID", value_col]]
    )


def _slope_delta(
    df: pd.DataFrame,
    rid_set: set,
    value_col: str,
    window: tuple[float, float] = SLOPE_WINDOW,
) -> pd.DataFrame:
    """12-month change: value at ~12 months minus baseline value."""
    df = df[df["RID"].isin(rid_set)].copy()
    df = standardize_viscode(df)
    df["months"] = df["VISCODE_STD"].apply(viscode_to_months)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    base = (
        df[df[value_col].notna()]
        .sort_values("months")
        .groupby("RID", as_index=False)
        .first()[["RID", value_col]]
        .rename(columns={value_col: f"{value_col}_bl"})
    )
    m12 = (
        df[df[value_col].notna() & df["months"].between(*window)]
        .sort_values("months")
        .groupby("RID", as_index=False)
        .first()[["RID", value_col]]
        .rename(columns={value_col: f"{value_col}_12"})
    )
    merged = base.merge(m12, on="RID", how="left")
    merged[f"DELTA_{value_col}"] = merged[f"{value_col}_12"] - merged[f"{value_col}_bl"]
    return merged[["RID", f"DELTA_{value_col}"]]


# ---------------------------------------------------------------------------
# Build full feature matrix
# ---------------------------------------------------------------------------

def build_features(data_dir: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_dir : Path  — directory of raw ADNI CSVs
    cohort   : DataFrame  — output of data.build_cohort (one row per RID)

    Returns
    -------
    DataFrame with columns: RID, label_36mo, <feature columns>, <MISSING_* masks>
    """
    rid_set = set(cohort["RID"])
    label_feat = cohort[["RID", "label_36mo"]].copy()

    # ---- CDR ----------------------------------------------------------------
    cdr = read_csv(pick_file(data_dir, "cdr"))
    cdr = standardize_viscode(cdr)
    cdr["CDRSB"] = pd.to_numeric(cdr.get("CDRSB", np.nan), errors="coerce")
    cdr_bl = _baseline(cdr, rid_set, "CDRSB")
    cdr_delta = _slope_delta(cdr, rid_set, "CDRSB")

    # ---- MMSE ---------------------------------------------------------------
    mmse = read_csv(pick_file(data_dir, "mmse"))
    mmse = standardize_viscode(mmse)
    mmse["MMSCORE"] = pd.to_numeric(mmse.get("MMSCORE", np.nan), errors="coerce")
    mmse_bl = _baseline(mmse, rid_set, "MMSCORE")
    mmse_delta = _slope_delta(mmse, rid_set, "MMSCORE")

    # ---- ADAS ---------------------------------------------------------------
    adas = read_csv(pick_file(data_dir, "adas"))
    adas = standardize_viscode(adas)
    adas_col = find_col(adas, ["TOTSCORE", "ADAS13", "TOTAL13"])
    adas[adas_col] = pd.to_numeric(adas[adas_col], errors="coerce")
    adas_bl = _baseline(adas, rid_set, adas_col).rename(columns={adas_col: "ADAS13"})
    adas_delta = _slope_delta(adas, rid_set, adas_col)
    adas_delta = adas_delta.rename(columns={f"DELTA_{adas_col}": "DELTA_ADAS"})

    # ---- MRI (UCSFFSX7) -----------------------------------------------------
    mri = read_csv(pick_file(data_dir, "ucsffsx7"))
    mri = standardize_viscode(mri)
    mri = mri[mri["RID"].isin(rid_set)].copy()
    mri["months"] = mri["VISCODE_STD"].apply(viscode_to_months)
    for col in ["ST29SV", "ST88SV", "ST24TA", "ST83TA", "ST10CV", "ST1SV"]:
        mri[col] = pd.to_numeric(mri[col], errors="coerce")
    mri["HIPPO_L_norm"] = mri["ST29SV"] / mri["ST1SV"]
    mri["HIPPO_R_norm"] = mri["ST88SV"] / mri["ST1SV"]
    mri["VENTRICLES_norm"] = mri["ST10CV"] / mri["ST1SV"]

    mri_bl = (
        mri[mri["ST29SV"].notna()]
        .sort_values("months")
        .groupby("RID", as_index=False)
        .first()[
            ["RID", "ST29SV", "ST88SV", "ST24TA", "ST83TA", "ST10CV", "ST1SV",
             "HIPPO_L_norm", "HIPPO_R_norm", "VENTRICLES_norm"]
        ]
        .rename(columns={
            "ST29SV": "HIPPO_L", "ST88SV": "HIPPO_R",
            "ST24TA": "ENTORH_L", "ST83TA": "ENTORH_R",
            "ST10CV": "VENTRICLES", "ST1SV": "ICV",
        })
    )
    hippo_l_delta = _slope_delta(mri, rid_set, "HIPPO_L_norm")
    hippo_r_delta = _slope_delta(mri, rid_set, "HIPPO_R_norm")

    # ---- Amyloid PET --------------------------------------------------------
    amy = read_csv(pick_file(data_dir, "ucberkeley_amy"))
    amy = standardize_viscode(amy)
    amy = amy[amy["RID"].isin(rid_set)].copy()
    amy["months"] = amy["VISCODE_STD"].apply(viscode_to_months)
    amy["SUMMARY_SUVR"] = pd.to_numeric(amy["SUMMARY_SUVR"], errors="coerce")
    amy_bl = (
        amy[amy["SUMMARY_SUVR"].notna()]
        .sort_values("months")
        .groupby("RID", as_index=False)
        .first()[["RID", "SUMMARY_SUVR", "AMYLOID_STATUS"]]
    )

    # ---- CSF ----------------------------------------------------------------
    csf = read_csv(pick_file(data_dir, "upennbiomk"))
    csf = standardize_viscode(csf)
    csf = csf[csf["RID"].isin(rid_set)].copy()
    csf["months"] = csf["VISCODE_STD"].apply(viscode_to_months)
    abeta_col = find_col(csf, ["ABETA42", "ABETA", "AB42"])
    tau_col = find_col(csf, ["TAU", "TTAU"])
    ptau_col = find_col(csf, ["PTAU", "PTAU181"])
    csf_cols = [c for c in [abeta_col, tau_col, ptau_col] if c]
    for c in csf_cols:
        csf[c] = pd.to_numeric(csf[c], errors="coerce")
    rename_map = {}
    if abeta_col:
        rename_map[abeta_col] = "ABETA42"
    if tau_col:
        rename_map[tau_col] = "TAU"
    if ptau_col:
        rename_map[ptau_col] = "PTAU"
    if abeta_col:
        csf_bl = (
            csf[csf[abeta_col].notna()]
            .sort_values("months")
            .groupby("RID", as_index=False)
            .first()[["RID"] + csf_cols]
            .rename(columns=rename_map)
        )
    else:
        csf_bl = pd.DataFrame({"RID": list(rid_set)})

    # ---- APOE ---------------------------------------------------------------
    apoe = read_csv(pick_file(data_dir, "apoeres"))
    if "APOE4" not in apoe.columns and "GENOTYPE" in apoe.columns:
        apoe["APOE4"] = apoe["GENOTYPE"].str.count("4").astype(float)
    else:
        apoe["APOE4"] = pd.to_numeric(apoe.get("APOE4", np.nan), errors="coerce")
    apoe_feat = (
        apoe[apoe["RID"].isin(rid_set)]
        .groupby("RID", as_index=False)["APOE4"]
        .first()
    )

    # ---- Demographics -------------------------------------------------------
    demo = read_csv(pick_file(data_dir, "ptdemog"))
    demo_cols = [c for c in ["PTGENDER", "PTEDUCAT", "AGE", "PTETHCAT", "PTRACCAT"] if c in demo.columns]
    demo_feat = (
        demo[demo["RID"].isin(rid_set)]
        .groupby("RID", as_index=False)[demo_cols]
        .first()
    )
    if "PTGENDER" in demo_feat.columns:
        demo_feat["PTGENDER"] = demo_feat["PTGENDER"].map({1.0: 0, 2.0: 1})

    # ---- Assemble -----------------------------------------------------------
    dfs = [
        label_feat,
        cdr_bl, mmse_bl, adas_bl,
        mri_bl[["RID", "HIPPO_L", "HIPPO_R", "ENTORH_L", "ENTORH_R",
                 "VENTRICLES", "HIPPO_L_norm", "HIPPO_R_norm", "VENTRICLES_norm"]],
        amy_bl, csf_bl, apoe_feat, demo_feat,
        cdr_delta, mmse_delta, adas_delta, hippo_l_delta, hippo_r_delta,
    ]
    X = reduce(lambda a, b: a.merge(b, on="RID", how="left"), dfs)

    # Missingness masks — one binary column per feature
    # Drop columns that provide no info (all-missing or constant)
    X = X.drop(columns=["CENTILOIDS", "AMYLOID_STATUS"], errors="ignore")

    feat_cols = [c for c in X.columns if c not in ("RID", "label_36mo")]
    for col in feat_cols:
        if X[col].isna().any():
            X[f"MISSING_{col}"] = X[col].isna().astype(int)

    print(f"Feature matrix: {X.shape[0]} subjects × {len(feat_cols)} features")
    _coverage(X, feat_cols)
    return X


def _coverage(X: pd.DataFrame, feat_cols: list[str]) -> None:
    n = len(X)
    for col in feat_cols:
        pct = X[col].notna().sum() / n * 100
        if pct < 100:
            print(f"  {col:30s}: {pct:5.1f}% coverage")
