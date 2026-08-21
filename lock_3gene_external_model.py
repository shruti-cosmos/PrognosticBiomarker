"""Fits and locks a 3-gene model (ANLN, LDLRAD3, ERO1A) for external
validation, since no GEO platform used here measures miRNA. A model and
threshold fit on the same 3 features used externally is internally
consistent; reusing the 4-feature model's threshold after dropping one
term produces a scale mismatch."""

from sklearn.model_selection import train_test_split

import config
import data_io
import consensus_model

THREE_GENE_FEATURES = [
    "ENSG00000011426.11",  # ANLN
    "ENSG00000179241.13",  # LDLRAD3
    "ENSG00000197930.13",  # ERO1A
]


def main():
    merged, mirna_features, gene_features = data_io.load_merged_dataset()
    dev_df, holdout_df = train_test_split(
        merged, test_size=0.15, random_state=config.RANDOM_SEED, stratify=merged[config.EVENT_COL]
    )

    print("Fitting 3-gene model (for GEO external validation)...")
    cph, feature_stats = consensus_model.fit_consensus_model(dev_df, consensus_features=THREE_GENE_FEATURES)
    threshold = consensus_model.lock_threshold(cph, dev_df, feature_stats, consensus_features=THREE_GENE_FEATURES)

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    consensus_model.save_coefficients(
        cph, config.ARTIFACT_DIR / "LOCKED_3gene_external_coefficients.csv",
        consensus_features=THREE_GENE_FEATURES,
    )
    consensus_model.save_scaler_and_threshold(
        feature_stats, threshold,
        config.ARTIFACT_DIR / "LOCKED_3gene_external_scaler.json",
        config.ARTIFACT_DIR / "LOCKED_3gene_external_thresholds.json",
    )

    metrics, _ = consensus_model.evaluate_model(cph, dev_df, holdout_df, feature_stats, consensus_features=THREE_GENE_FEATURES)
    print(f"\n3-gene model TCGA holdout: Harrell C={metrics['harrell_c_index']:.3f} "
          f"IPCW C={metrics['ipcw_c_index']:.3f} mean AUC={metrics['mean_auc']:.3f}")


if __name__ == "__main__":
    main()
