"""Fits and locks the 4-feature consensus model."""

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import data_io
import consensus_model


def main():
    print("\n" + "=" * 80 + "\nFINAL 4-FEATURE CONSENSUS MODEL\n" + "=" * 80)
    for i, feature in enumerate(consensus_model.CONSENSUS_FEATURES, start=1):
        print(f"  {i}. {feature}")

    merged, mirna_features, gene_features = data_io.load_merged_dataset()

    missing = [f for f in consensus_model.CONSENSUS_FEATURES if f not in merged.columns]
    if missing:
        raise ValueError(f"Missing consensus features: {missing}")

    dev_df, holdout_df = train_test_split(
        merged, test_size=0.15, random_state=config.RANDOM_SEED, stratify=merged[config.EVENT_COL]
    )
    print(f"\nDevelopment: {dev_df.shape} | Holdout: {holdout_df.shape}")

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    cph, feature_stats = consensus_model.fit_consensus_model(dev_df)
    threshold = consensus_model.lock_threshold(cph, dev_df, feature_stats)

    consensus_model.save_coefficients(cph, config.ARTIFACT_DIR / "LOCKED_consensus_4_feature_coefficients.csv")
    consensus_model.save_scaler_and_threshold(
        feature_stats, threshold,
        config.ARTIFACT_DIR / "LOCKED_consensus_4_feature_scaler.json",
        config.ARTIFACT_DIR / "LOCKED_consensus_4_feature_thresholds.json",
    )
    consensus_model.save_model(cph, config.ARTIFACT_DIR / "locked_consensus_model.pkl")

    metrics, _ = consensus_model.evaluate_model(cph, dev_df, holdout_df, feature_stats)
    print(f"\nHoldout: Harrell C={metrics['harrell_c_index']:.3f} "
          f"IPCW C={metrics['ipcw_c_index']:.3f} mean AUC={metrics['mean_auc']:.3f}")

    pd.DataFrame([{
        "model": "LOCKED_4_FEATURE_CONSENSUS", "n_features": len(consensus_model.CONSENSUS_FEATURES),
        "features": ";".join(consensus_model.CONSENSUS_FEATURES),
        "harrell_c_index": metrics["harrell_c_index"], "ipcw_c_index": metrics["ipcw_c_index"],
        "mean_auc": metrics["mean_auc"],
    }]).to_csv(config.ARTIFACT_DIR / "LOCKED_consensus_4_feature_metrics.csv", index=False)

    print("\nLocking complete.")


if __name__ == "__main__":
    main()
