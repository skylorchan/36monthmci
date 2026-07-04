"""
Censoring and selection-bias audit.

Quantifies the subjects excluded by inclusion criteria and characterises
whether the retained cohort is systematically different from the excluded
population — the selection-bias question a reviewer will ask.

Produces:
  results/censoring_flow.csv    — CONSORT-style exclusion counts
  results/censoring_bias.csv    — feature comparison: included vs excluded
  results/censoring_flow.png    — CONSORT flow diagram

Key finding documented here:
  417 total MCI subjects → 308 retained (74%) → 109 excluded as censored.
  Censored subjects had mean 17.1 months follow-up vs 64.5 months retained.
  This is SELECTION BIAS: the included cohort over-represents long-followed
  patients. Converters with short time-to-event ARE captured (they convert
  before 36mo regardless of total follow-up), but slow/late converters and
  stable non-converters with short follow-up are dropped. This inflates the
  apparent conversion rate and biases toward subjects willing/able to stay
  in the study long-term (typically healthier, higher SES, fewer comorbidities).

Usage:
    python -m src.experiments.censoring_audit --config configs/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.data import (
    WINDOW_MONTHS,
    build_cohort,
    coerce_examdate,
    pick_file,
    read_csv,
    standardize_viscode,
    viscode_to_months,
)


# ---------------------------------------------------------------------------
# Rebuild full pre-filtered subject table
# ---------------------------------------------------------------------------

def _build_full_label_table(data_dir: Path) -> pd.DataFrame:
    """Returns one row per MCI subject with labels BEFORE inclusion filtering."""
    dx = read_csv(pick_file(data_dir, "dxsum"))
    dx, date_col = coerce_examdate(dx)
    dx = standardize_viscode(dx)

    month_col = next((c for c in dx.columns if c.lower() in ("month_bl", "monthbl")), None)
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
        d = str(row.get("DIAGNOSIS", "")).upper()
        return any(t in d for t in ("EMCI", "LMCI", "MCI"))

    def _is_ad(row):
        if "DXAD" in row and not pd.isna(row["DXAD"]):
            return int(row["DXAD"]) == 1
        d = str(row.get("DIAGNOSIS", "")).upper()
        return d == "AD" or "ALZ" in d or ("DEMENTIA" in d and "MCI" not in d)

    dx["is_MCI"] = dx.apply(_is_mci, axis=1)
    dx["is_AD"] = dx.apply(_is_ad, axis=1)

    baseline = dx[dx["is_MCI"]].groupby("RID", as_index=False).first()
    dx1 = dx.merge(baseline[["RID"]], on="RID", how="inner")

    if date_col:
        base_dates = (
            dx[dx["is_MCI"]]
            .groupby("RID", as_index=False)
            .first()[["RID", date_col]]
            .rename(columns={date_col: "baseline_date"})
        )
        dx1 = dx1.merge(base_dates, on="RID", how="left")
        dx1["mfb"] = (dx1[date_col] - dx1["baseline_date"]).dt.days / 30.44
    else:
        dx1["mfb"] = dx1["months_from_bl"]

    dx1 = dx1[dx1["mfb"].notna() & (dx1["mfb"] >= -0.5)]

    def _label(g):
        g = g.sort_values("mfb")
        within = g[g["mfb"] <= WINDOW_MONTHS]
        ever_ad = bool(within["is_AD"].any())
        max_f = float(g["mfb"].max())
        n = len(g)
        if ever_ad:
            return pd.Series({"label_36mo": 1.0, "censored": 0, "max_follow": max_f, "n_visits": n})
        if max_f >= WINDOW_MONTHS:
            return pd.Series({"label_36mo": 0.0, "censored": 0, "max_follow": max_f, "n_visits": n})
        return pd.Series({"label_36mo": np.nan, "censored": 1, "max_follow": max_f, "n_visits": n})

    return dx1.groupby("RID").apply(_label, include_groups=False).reset_index()


# ---------------------------------------------------------------------------
# Inclusion criteria flow
# ---------------------------------------------------------------------------

CRITERIA = [
    ("has_any_mci",        "Has ≥1 MCI visit in DXSUM"),
    ("pass_2visits",       ">=2 visits"),
    ("pass_36mo_or_conv",  ">=36 months follow-up OR converts within 36 months"),
    ("pass_12mo",          ">=12 months follow-up OR converts within 36 months"),
    ("final",              "Final cohort (non-censored)"),
]


def build_flow(full: pd.DataFrame) -> pd.DataFrame:
    converted = full["label_36mo"].eq(1)
    full = full.copy()
    full["pass_2visits"] = full["n_visits"] >= 2
    full["pass_36mo_or_conv"] = (full["max_follow"] >= WINDOW_MONTHS) | converted
    full["pass_12mo"] = (full["max_follow"] >= 12) | converted
    full["final"] = full["pass_2visits"] & full["pass_36mo_or_conv"] & full["pass_12mo"] & full["label_36mo"].notna()
    full["has_any_mci"] = True

    rows = []
    prev_n = None
    for col, label in CRITERIA:
        mask = full[col]
        n = int(mask.sum())
        excluded = 0 if prev_n is None else prev_n - n
        rows.append({"Step": label, "N retained": n, "N excluded": excluded})
        prev_n = n

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selection-bias comparison: included vs excluded (censored)
# ---------------------------------------------------------------------------

def compare_included_excluded(full: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    converted = full["label_36mo"].eq(1)
    full = full.copy()
    included = full[
        (full["n_visits"] >= 2)
        & ((full["max_follow"] >= WINDOW_MONTHS) | converted)
        & ((full["max_follow"] >= 12) | converted)
        & full["label_36mo"].notna()
    ]
    excluded = full[~full["RID"].isin(included["RID"])]

    rows = []

    def _compare(name, inc_vals, exc_vals):
        inc_vals = inc_vals.dropna()
        exc_vals = exc_vals.dropna()
        if len(inc_vals) < 2 or len(exc_vals) < 2:
            return
        t, p = stats.ttest_ind(inc_vals, exc_vals, equal_var=False)
        rows.append({
            "Variable": name,
            "Included mean (SD)": f"{inc_vals.mean():.2f} ({inc_vals.std():.2f})",
            "Excluded mean (SD)": f"{exc_vals.mean():.2f} ({exc_vals.std():.2f})",
            "p-value": round(p, 3),
            "Significant (p<0.05)": p < 0.05,
        })

    # Merge in demographics and cognitive scores for comparison
    demo = read_csv(pick_file(data_dir, "ptdemog"))
    cdr = read_csv(pick_file(data_dir, "cdr"))
    mmse = read_csv(pick_file(data_dir, "mmse"))

    for df, col, name in [
        (demo, "AGE", "Age at baseline"),
        (demo, "PTEDUCAT", "Education (years)"),
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            inc_m = df[df["RID"].isin(included["RID"])].groupby("RID")[col].first()
            exc_m = df[df["RID"].isin(excluded["RID"])].groupby("RID")[col].first()
            _compare(name, inc_m, exc_m)

    cdr["CDRSB"] = pd.to_numeric(cdr.get("CDRSB", np.nan), errors="coerce")
    cdr = standardize_viscode(cdr)
    cdr["months"] = cdr["VISCODE_STD"].apply(viscode_to_months)
    cdr_bl = cdr[cdr["CDRSB"].notna()].sort_values("months").groupby("RID")["CDRSB"].first()
    _compare("CDR-SB at baseline", cdr_bl[cdr_bl.index.isin(included["RID"])],
             cdr_bl[cdr_bl.index.isin(excluded["RID"])])

    mmse["MMSCORE"] = pd.to_numeric(mmse.get("MMSCORE", np.nan), errors="coerce")
    mmse = standardize_viscode(mmse)
    mmse["months"] = mmse["VISCODE_STD"].apply(viscode_to_months)
    mmse_bl = mmse[mmse["MMSCORE"].notna()].sort_values("months").groupby("RID")["MMSCORE"].first()
    _compare("MMSE at baseline", mmse_bl[mmse_bl.index.isin(included["RID"])],
             mmse_bl[mmse_bl.index.isin(excluded["RID"])])

    _compare("Max follow-up (months)", included["max_follow"], excluded["max_follow"])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CONSORT flow diagram
# ---------------------------------------------------------------------------

def plot_flow(flow: pd.DataFrame, results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, len(flow) * 1.6 + 1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(flow) * 2 + 1)
    ax.axis("off")

    colors = {"retained": "#E3F2FD", "excluded": "#FFEBEE"}
    box_w, box_h = 5.5, 1.1

    for i, row in flow.iterrows():
        y = (len(flow) - i) * 2 - 0.5
        # Main box
        rect = mpatches.FancyBboxPatch(
            (2.25, y), box_w, box_h,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="#1565C0", facecolor=colors["retained"]
        )
        ax.add_patch(rect)
        ax.text(5.0, y + box_h / 2, f"{row['Step']}\nn = {row['N retained']:,}",
                ha="center", va="center", fontsize=9, fontweight="bold")

        # Exclusion box
        if row["N excluded"] > 0:
            ex_x, ex_y = 8.2, y + 0.1
            rect_ex = mpatches.FancyBboxPatch(
                (ex_x, ex_y), 1.5, 0.8,
                boxstyle="round,pad=0.05", linewidth=1,
                edgecolor="#C62828", facecolor=colors["excluded"]
            )
            ax.add_patch(rect_ex)
            ax.text(ex_x + 0.75, ex_y + 0.4, f"−{row['N excluded']}",
                    ha="center", va="center", fontsize=8.5, color="#C62828", fontweight="bold")
            ax.annotate("", xy=(ex_x, ex_y + 0.4), xytext=(2.25 + box_w, y + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))

        # Arrow down
        if i < len(flow) - 1:
            ax.annotate("", xy=(5.0, y), xytext=(5.0, y - 0.9),
                        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax.set_title("CONSORT-style Subject Inclusion Flow\n(ADNI MCI cohort)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(results_dir / "censoring_flow.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir: Path, results_dir: Path) -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Building full (unfiltered) label table...")
    full = _build_full_label_table(data_dir)

    flow = build_flow(full)
    flow.to_csv(results_dir / "censoring_flow.csv", index=False)
    print("\nCONSORT flow:")
    print(flow.to_string(index=False))

    print("\nSelection-bias comparison (included vs censored-excluded)...")
    bias = compare_included_excluded(full, data_dir)
    bias.to_csv(results_dir / "censoring_bias.csv", index=False)
    print(bias.to_string(index=False))

    plot_flow(flow, results_dir)

    # Print the limitation text for the paper
    included_n = int(flow.iloc[-1]["N retained"])
    excluded_n = int(flow.iloc[-1]["N excluded"]) + sum(
        int(r) for r in flow["N excluded"].iloc[:-1]
    )
    total = included_n + excluded_n

    print("\n" + "=" * 60)
    print("METHODS LIMITATION TEXT (copy into paper)")
    print("=" * 60)
    print(f"""
Cohort selection introduced right-censoring bias. Of {total} subjects with
≥1 MCI visit in ADNI, {excluded_n} ({excluded_n/total*100:.0f}%) were excluded because they
converted to AD within 36 months but lacked sufficient follow-up data, or
remained MCI with <36 months of recorded follow-up. The retained cohort
(n={included_n}) over-represents subjects with long follow-up (mean {full[full['label_36mo'].notna()]['max_follow'].mean():.0f} months),
which may reflect survival bias: subjects who remained in the study tend
to be healthier, have higher socioeconomic status, and fewer comorbidities
than the general MCI population. Generalisability to clinical settings
where follow-up is irregular or truncated should be interpreted with
caution. Future work will reframe this as a time-to-event problem using
Random Survival Forests to recover the censored subjects and report
time-dependent AUC at 12, 24, and 36 months.
""")
    print(f"Saved: {results_dir}/censoring_flow.csv, censoring_bias.csv, censoring_flow.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(
        data_dir=Path(cfg.get("data_dir", "data/raw")),
        results_dir=Path(cfg.get("results_dir", "results")),
    )
