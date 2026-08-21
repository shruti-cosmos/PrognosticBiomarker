"""Reconstructs published comparator gene signatures and scores them on
the same train/eval split and metrics as the study's own model."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

import config
import evaluation
import survival_models


def score_with_published_coefficients(df, features, coefficients):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Comparator features not present in dataset: {missing}")
    X = df[features].values
    beta = np.array([coefficients[f] for f in features])
    return X @ beta


def refit_signature_on_training_data(train_df, features, penalizer=0.1):
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Comparator features missing from data: {missing}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df[features + [config.TIME_COL, config.EVENT_COL]],
            duration_col=config.TIME_COL, event_col=config.EVENT_COL)
    return cph


def evaluate_all_comparators(train_df, eval_df):
    y_train = survival_models.to_structured_y(train_df)
    y_eval = survival_models.to_structured_y(eval_df)

    rows = []
    for name, spec in config.COMPARATOR_SIGNATURES.items():
        features = spec["features"]
        coefficients = spec.get("coefficients")
        try:
            if coefficients:
                risk_eval = score_with_published_coefficients(eval_df, features, coefficients)
            else:
                cph = refit_signature_on_training_data(train_df, features)
                risk_eval = cph.predict_partial_hazard(eval_df[features]).values.flatten()

            metrics = evaluation.evaluate_risk_scores(y_train, y_eval, risk_eval)
            metrics.update({
                "signature": name, "reference": spec.get("reference", ""),
                "n_features": len(features),
                "weights_source": "published" if coefficients else "refit_on_our_training_data",
            })
            rows.append(metrics)
        except Exception as e:
            rows.append({"signature": name, "error": str(e)})

    return pd.DataFrame(rows)
