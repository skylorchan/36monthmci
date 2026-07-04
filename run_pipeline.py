"""
End-to-end pipeline: build cohort → extract features → train → evaluate → explain.

Usage:
    python run_pipeline.py --data-dir data/raw --config configs/base.yaml
"""

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip cohort/feature build; load existing feature CSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(args.data_dir)
    feat_csv = Path(cfg["feature_csv"])
    results_dir = Path(cfg.get("results_dir", "results"))
    seed = cfg.get("seed", 42)

    if not args.skip_build:
        from src.data import build_cohort
        from src.features import build_features

        print("=== Building cohort ===")
        cohort, _ = build_cohort(data_dir)

        print("\n=== Extracting features ===")
        X = build_features(data_dir, cohort)
        feat_csv.parent.mkdir(parents=True, exist_ok=True)
        X.to_csv(feat_csv, index=False)
        print(f"Saved: {feat_csv}")
    else:
        print(f"Skipping build; loading {feat_csv}")

    print("\n=== Training ===")
    from src.train import train
    metrics, final_pipe, feat_cols = train(
        feature_csv=feat_csv,
        n_trials=cfg.get("n_trials", 80),
        n_splits=cfg.get("n_splits", 5),
        seed=seed,
        results_dir=results_dir,
    )

    print("\n=== SHAP ===")
    import pandas as pd
    from src.explain import run_shap
    from src.splits import get_feature_cols, subject_split

    df = pd.read_csv(feat_csv)
    df = df[df["label_36mo"].notna()].reset_index(drop=True)
    fc = get_feature_cols(df)
    X = df[fc]
    y = df["label_36mo"].astype(int)
    rids = df["RID"]
    _, X_test, _, _ = subject_split(X, y, rids, seed=seed)
    run_shap(final_pipe, X_test, fc, results_dir)

    print("\nDone. Results in:", results_dir)


if __name__ == "__main__":
    main()
