"""
Cohort construction and labeling from raw ADNI CSVs.

Produces one row per subject with columns: RID, label_36mo, censored, max_follow_months.
Subject inclusion criteria:
  - ≥2 visits
  - ≥36 months follow-up OR converts within 36 months
  - ≥12 months follow-up OR converts within 36 months
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOW_MONTHS = 36.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_file(data_dir: Path, *fragments: str) -> Path:
    """Return first CSV in data_dir whose name (lowercased) contains all fragments."""
    for p in sorted(data_dir.glob("*.csv")):
        name = p.name.lower()
        if all(f.lower() in name for f in fragments):
            return p
    raise FileNotFoundError(f"No CSV matching {fragments} in {data_dir}")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", low_memory=False)


def standardize_viscode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["VISCODE", "VISCODE2", "VISCODE3"]:
        if col in df.columns:
            df["VISCODE_STD"] = df[col].astype(str).str.strip().str.lower()
            return df
    df["VISCODE_STD"] = np.nan
    return df


def viscode_to_months(v) -> float:
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ("bl", "baseline"):
        return 0.0
    m = re.fullmatch(r"m(\d+)", s)
    if m:
        return float(m.group(1))
    return np.nan


def coerce_examdate(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    df = df.copy()
    for col in ("EXAMDATE", "USERDATE", "USERDATE2", "SCANDATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df, col
    return df, None


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def build_cohort(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    cohort : DataFrame
        One row per subject: RID, label_36mo, censored, max_follow_months
    dx_core : DataFrame
        Visit-level DX table filtered to cohort subjects
    """
    dx = read_csv(pick_file(data_dir, "dxsum"))
    dx, date_col = coerce_examdate(dx)
    dx = standardize_viscode(dx)

    # months from baseline
    month_col = next(
        (c for c in dx.columns if c.lower() in ("month_bl", "monthbl")), None
    )
    if month_col:
        dx["months_from_bl"] = pd.to_numeric(dx[month_col], errors="coerce")
    else:
        dx["months_from_bl"] = dx["VISCODE_STD"].apply(viscode_to_months)

    for c in ["DXMCI", "DXAD"]:
        if c in dx.columns:
            dx[c] = pd.to_numeric(dx[c], errors="coerce")

    def _is_mci(row):
        if "DXMCI" in row and not pd.isna(row["DXMCI"]):
            return int(row["DXMCI"]) == 1
        d = str(row.get("DIAGNOSIS", "")).strip().upper()
        return any(t in d for t in ("EMCI", "LMCI", "MCI"))

    def _is_ad(row):
        if "DXAD" in row and not pd.isna(row["DXAD"]):
            return int(row["DXAD"]) == 1
        d = str(row.get("DIAGNOSIS", "")).strip().upper()
        return d == "AD" or "ALZ" in d or ("DEMENTIA" in d and "MCI" not in d)

    dx["is_MCI"] = dx.apply(_is_mci, axis=1)
    dx["is_AD"] = dx.apply(_is_ad, axis=1)

    sort_cols = ["RID"]
    if dx["months_from_bl"].notna().any():
        sort_cols += ["months_from_bl"]
    elif date_col:
        sort_cols += [date_col]
    sort_cols += ["VISCODE_STD"]
    dx = dx.sort_values(sort_cols)

    # First MCI visit per subject
    baseline = (
        dx[dx["is_MCI"]]
        .groupby("RID", as_index=False)
        .first()
        .rename(columns={"VISCODE_STD": "baseline_viscode"})
    )

    # Align all visits to baseline date
    dx1 = dx.merge(baseline[["RID", "baseline_viscode"]], on="RID", how="inner")
    if date_col is not None:
        base_dates = (
            dx[dx["is_MCI"]]
            .groupby("RID", as_index=False)
            .first()[["RID", date_col]]
            .rename(columns={date_col: "baseline_date"})
        )
        dx1 = dx1.merge(base_dates, on="RID", how="left")
        dx1["months_from_baseline"] = (
            (dx1[date_col] - dx1["baseline_date"]).dt.days / 30.44
        )
    else:
        dx1["months_from_baseline"] = dx1["months_from_bl"]

    miss = dx1["months_from_baseline"].isna()
    dx1.loc[miss, "months_from_baseline"] = dx1.loc[miss, "VISCODE_STD"].apply(
        viscode_to_months
    )
    dx1 = dx1[dx1["months_from_baseline"].notna() & (dx1["months_from_baseline"] >= -0.5)]
    dx1 = dx1.sort_values(["RID", "months_from_baseline"])

    # Label subjects
    def _label_subject(g):
        g = g.sort_values("months_from_baseline")
        within = g[g["months_from_baseline"] <= WINDOW_MONTHS]
        ever_ad = bool(within["is_AD"].any())
        max_follow = float(g["months_from_baseline"].max())
        if ever_ad:
            return pd.Series({"label_36mo": 1.0, "censored": 0, "max_follow_months": max_follow})
        if max_follow >= WINDOW_MONTHS:
            return pd.Series({"label_36mo": 0.0, "censored": 0, "max_follow_months": max_follow})
        return pd.Series({"label_36mo": np.nan, "censored": 1, "max_follow_months": max_follow})

    labels = dx1.groupby("RID").apply(_label_subject, include_groups=False).reset_index()
    visit_counts = dx1.groupby("RID").size().reset_index(name="n_visits")
    cohort = labels.merge(visit_counts, on="RID", how="left")

    converted = cohort["label_36mo"].eq(1)
    cohort = cohort[
        (cohort["n_visits"] >= 2)
        & ((cohort["max_follow_months"] >= WINDOW_MONTHS) | converted)
        & ((cohort["max_follow_months"] >= 12) | converted)
    ].copy()

    final_ids = set(cohort["RID"])
    dx_core = dx1[dx1["RID"].isin(final_ids)].merge(labels, on="RID", how="left")

    print(f"Cohort: {len(cohort)} subjects")
    print(f"  Converters:     {int((cohort['label_36mo']==1).sum())}")
    print(f"  Non-converters: {int((cohort['label_36mo']==0).sum())}")
    print(f"  Censored:       {int((cohort['censored']==1).sum())}")
    print(f"  Mean follow-up: {cohort['max_follow_months'].mean():.1f} months")

    return cohort, dx_core
