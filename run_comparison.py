"""Scores every comparator signature in resolved_comparator_signatures.json
alongside the study's own consensus model, on the same split and metrics."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
import data_io
import comparator_signatures
import consensus_model
import survival_models


def main():
    resolved_path = Path("resolved_comparator_signatures.json")
    if not resolved_path.exists():
        print("resolved_comparator_signatures.json not found. Run map_comparator_genes.py first.")
        return
    with open(resolved_path) as f:
        config.COMPARATOR_SIGNATURES = json.load(f)

    merged, mirna_features, gene_features = data_io.load_merged_dataset()
    dev_df, holdout_df = train_test_split(
        merged, test_size=0.15, random_state=config.RANDOM_SEED, stratify=merged[config.EVENT_COL]
    )

    comp_results = comparator_signatures.evaluate_all_comparators(dev_df, holdout_df)

    cph, feature_stats = consensus_model.fit_consensus_model(dev_df)
    our_metrics, _ = consensus_model.evaluate_model(cph, dev_df, holdout_df, feature_stats)

    our_row = pd.DataFrame([{
        "signature": "OUR_consensus_panel (this study)", "n_features": len(consensus_model.CONSENSUS_FEATURES),
        "weights_source": "fit_on_our_training_data",
        "harrell_c_index": our_metrics["harrell_c_index"], "ipcw_c_index": our_metrics["ipcw_c_index"],
        "mean_auc": our_metrics["mean_auc"], "reference": "This study",
    }])

    display_cols = ["signature", "n_features", "weights_source", "harrell_c_index", "ipcw_c_index", "mean_auc", "reference"]
    display_cols = [c for c in display_cols if c in comp_results.columns]
    full_table = pd.concat([our_row, comp_results[display_cols]], ignore_index=True)

    print("\n=== Full comparison table ===")
    print(full_table.to_string(index=False))

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    full_table.to_csv(config.ARTIFACT_DIR / "comparator_signature_results.csv", index=False)
    print(f"\nSaved to {config.ARTIFACT_DIR / 'comparator_signature_results.csv'}")


if __name__ == "__main__":
    main()
