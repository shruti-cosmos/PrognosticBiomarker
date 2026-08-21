"""Final consensus feature-set survival model: fit once on standardized
development-set features, with standardization stats and risk threshold
frozen and saved for external reuse (including cross-platform scoring)."""

import json
import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import config
import survival_models
import evaluation

CONSENSUS_FEATURES = [
    "ENSG00000179241.13",  # LDLRAD3
    "ENSG00000011426.11",  # ANLN
    "hsa-mir-4435-2",
    "ENSG00000197930.13",  # ERO1A
]


def fit_consensus_model(dev_df: pd.DataFrame, consensus_features=None, penalizer=0.01):
    if consensus_features is None:
        consensus_features = CONSENSUS_FEATURES
    consensus_features = list(consensus_features)

    required = consensus_features + [config.TIME_COL, config.EVENT_COL]
    missing = [c for c in required if c not in dev_df.columns]
    if missing:
        raise ValueError(f"Development data missing required columns: {missing}")

    model_df = dev_df[required].copy()
    feature_stats = {}
    for f in consensus_features:
        mean, std = float(model_df[f].mean()), float(model_df[f].std())
        feature_stats[f] = {"mean": mean, "std": std}
        model_df[f] = (model_df[f] - mean) / std

    print(f"\nFitting consensus model on {len(model_df)} samples, {len(consensus_features)} features.")
    for f in consensus_features:
        s = feature_stats[f]
        print(f"  {f}  (mean={s['mean']:.4f}, std={s['std']:.4f})")

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_df, duration_col=config.TIME_COL, event_col=config.EVENT_COL)
    print("\nFinal coefficients (per 1-SD change):")
    print(cph.summary[["coef", "exp(coef)", "p"]].to_string())

    return cph, feature_stats


def predict_risk(cph, df, feature_stats, consensus_features=None):
    if consensus_features is None:
        consensus_features = CONSENSUS_FEATURES
    missing = [f for f in consensus_features if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing consensus features: {missing}")

    scaled = df[consensus_features].copy()
    for f in consensus_features:
        s = feature_stats[f]
        scaled[f] = (scaled[f] - s["mean"]) / s["std"]

    # Linear predictor (log relative hazard) -- must match the convention
    # used everywhere else in the pipeline (e.g. ScaledCoxnet.predict_risk,
    # calculate_locked_risk in run_external_validation.py). Do NOT use
    # predict_partial_hazard here (exp-scale) or thresholds computed on
    # this function's output will not be comparable to scores computed
    # elsewhere on the linear scale.
    risk = cph.predict_log_partial_hazard(scaled).astype(float).values.ravel()
    return risk


def lock_threshold(cph, dev_df, feature_stats, consensus_features=None, quantiles=None):
    quantiles = quantiles or config.RISK_GROUP_QUANTILES
    risk_dev = predict_risk(cph, dev_df, feature_stats, consensus_features)
    cutpoints = np.quantile(risk_dev, quantiles).tolist()
    threshold = {"quantiles": quantiles, "cutpoints": cutpoints, "n_training_samples_used": len(dev_df)}
    print(f"\nLocked threshold (dev-set risk scores only): {threshold}")
    return threshold


def evaluate_model(cph, dev_df, eval_df, feature_stats, consensus_features=None):
    if consensus_features is None:
        consensus_features = CONSENSUS_FEATURES
    y_dev = survival_models.to_structured_y(dev_df)
    y_eval = survival_models.to_structured_y(eval_df)
    risk = predict_risk(cph, eval_df, feature_stats, consensus_features)
    metrics = evaluation.evaluate_risk_scores(y_dev, y_eval, risk)
    metrics["n_features"] = len(consensus_features)
    metrics["features"] = consensus_features
    return metrics, risk


def save_coefficients(cph, output_path, consensus_features=None):
    if consensus_features is None:
        consensus_features = CONSENSUS_FEATURES
    summary = cph.summary[["coef", "exp(coef)", "p"]].copy()
    summary = summary.rename(columns={"coef": "coefficient", "exp(coef)": "hazard_ratio", "p": "p_value"})
    summary["feature"] = summary.index
    summary = summary[["feature", "coefficient", "hazard_ratio", "p_value"]]
    summary.to_csv(output_path, index=False)
    print(f"\nSaved coefficients to: {output_path}")
    return summary


def save_scaler_and_threshold(feature_stats, threshold, scaler_path, threshold_path):
    with open(scaler_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    with open(threshold_path, "w") as f:
        json.dump(threshold, f, indent=2)
    print(f"Saved scaler stats to: {scaler_path}")
    print(f"Saved threshold to: {threshold_path}")


def save_model(cph, output_path):
    joblib.dump(cph, output_path)
    print(f"Saved model to: {output_path}")


def load_model(model_path):
    return joblib.load(model_path)
