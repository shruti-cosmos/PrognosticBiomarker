"""Random Survival Forest on the same 4 consensus features, same split,
same censoring-aware metrics -- an ML-vs-Cox comparison."""

from sklearn.model_selection import train_test_split

import config
import data_io
import survival_models
import evaluation
import consensus_model


def main():
    merged, _, _ = data_io.load_merged_dataset()
    dev_df, holdout_df = train_test_split(
        merged, test_size=0.15, random_state=config.RANDOM_SEED, stratify=merged[config.EVENT_COL]
    )

    features = consensus_model.CONSENSUS_FEATURES
    y_dev = survival_models.to_structured_y(dev_df)
    y_holdout = survival_models.to_structured_y(holdout_df)

    rsf = survival_models.fit_random_survival_forest(dev_df[features], y_dev, n_estimators=300)
    risk_holdout = rsf.predict(holdout_df[features].values)

    metrics = evaluation.evaluate_risk_scores(y_dev, y_holdout, risk_holdout)
    print("=== Random Survival Forest (4 consensus markers) ===")
    print(f"Harrell C-index : {metrics['harrell_c_index']:.3f}")
    print(f"IPCW C-index    : {metrics['ipcw_c_index']:.3f}")
    print(f"Mean AUC        : {metrics['mean_auc']:.3f}")


if __name__ == "__main__":
    main()
