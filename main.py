"""Orchestrates: data load, nested CV, feature stability, locked model,
PH assumptions, thresholds, comparator signatures, external validation,
reproducibility artifacts."""

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import data_io
import screening
import survival_models
import nested_cv
import evaluation
import thresholds
import comparator_signatures
import reproducibility


def main():
    merged, mirna_features, gene_features = data_io.load_merged_dataset()
    candidate_features = mirna_features + gene_features

    evaluation.diagnose_censoring_at_horizons(merged)

    dev_df, holdout_df = train_test_split(
        merged, test_size=0.15, random_state=config.RANDOM_SEED, stratify=merged[config.EVENT_COL]
    )
    print(f"\nDev cohort: {len(dev_df)} | Holdout: {len(holdout_df)}")

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results, oof_risk, selected_features_per_fold = nested_cv.nested_cv_survival(
        dev_df, mirna_features, gene_features,
        checkpoint_path=config.ARTIFACT_DIR / "nested_cv_checkpoint.csv"
    )
    print("\n=== Nested CV summary ===")
    print(fold_results[["harrell_c_index", "ipcw_c_index", "mean_auc"]].agg(["mean", "std"]))

    all_nonzero = []
    for fold_data in selected_features_per_fold.values():
        all_nonzero.extend(fold_data["nonzero_this_fold"].keys())
    stability = Counter(all_nonzero)
    print("\n=== Feature selection stability across outer folds ===")
    for feat, count in stability.most_common(30):
        print(f"  {feat}: nonzero in {count}/{len(selected_features_per_fold)} folds")
    consensus_features = [f for f, c in stability.items() if c >= 2]
    print(f"\nConsensus panel (>=2/5 folds): {consensus_features}")

    most_common_l1 = Counter(fold_results["l1_ratio"]).most_common(1)[0][0]
    median_alpha = fold_results["alpha"].median()

    _, final_features = screening.screen_features_by_group(
        dev_df, feature_groups={"mirna": mirna_features, "gene": gene_features},
        max_features_per_group=config.MAX_SCREENED_FEATURES_BY_GROUP,
    )
    y_dev = survival_models.to_structured_y(dev_df)

    locked_model = survival_models.ScaledCoxnet(l1_ratio=most_common_l1, alphas=[median_alpha])
    try:
        locked_model.fit(dev_df[final_features], y_dev)
    except ArithmeticError:
        probe = survival_models.ScaledCoxnet(l1_ratio=most_common_l1)
        probe.fit(dev_df[final_features], y_dev)
        locked_model, median_alpha = locked_model.refit_at_alpha_with_backoff(
            dev_df[final_features], y_dev, median_alpha, probe.model.alphas_
        )
    print(f"\nLocked model: {len(final_features)} candidates, "
          f"{(locked_model.coefficients_at_best_alpha() != 0).sum()} nonzero.")

    nz_coefs = locked_model.coefficients_at_best_alpha()
    nz_coefs = nz_coefs[nz_coefs != 0]
    print("\n=== LOCKED MODEL SIGNATURE ===")
    print(nz_coefs.to_string())

    from lifelines import CoxPHFitter
    nz_features = nz_coefs.index.tolist()
    if nz_features:
        cph_check = CoxPHFitter(penalizer=0.01)
        cph_check.fit(dev_df[nz_features + [config.TIME_COL, config.EVENT_COL]],
                      duration_col=config.TIME_COL, event_col=config.EVENT_COL)
        try:
            evaluation.check_ph_assumptions(cph_check, dev_df[nz_features + [config.TIME_COL, config.EVENT_COL]])
        except Exception as e:
            print(f"  PH assumption check skipped ({e}).")

    risk_dev = locked_model.predict_risk(dev_df[final_features])
    locked_thresholds = thresholds.lock_thresholds(risk_dev)

    y_holdout = survival_models.to_structured_y(holdout_df)
    risk_holdout = locked_model.predict_risk(holdout_df[final_features])
    holdout_metrics = evaluation.evaluate_risk_scores(y_dev, y_holdout, risk_holdout)
    print("\n=== Locked-model holdout performance ===")
    print(holdout_metrics)

    if config.COMPARATOR_SIGNATURES:
        comp_results = comparator_signatures.evaluate_all_comparators(dev_df, holdout_df)
        print("\n=== Comparator signatures ===")
        print(comp_results)

    reproducibility.save_run_manifest(
        out_dir=config.ARTIFACT_DIR, fold_results=fold_results,
        selected_features_per_fold={k: v["nonzero_this_fold"] for k, v in selected_features_per_fold.items()},
        locked_model=locked_model, locked_thresholds=locked_thresholds,
        train_patient_ids=dev_df["patient_id"].tolist(), test_patient_ids=holdout_df["patient_id"].tolist(),
        notes="Nested-CV survival pipeline with grouped miRNA/gene FDR screening.",
    )
    print("\nDone. See repo_artifacts/.")


if __name__ == "__main__":
    main()
