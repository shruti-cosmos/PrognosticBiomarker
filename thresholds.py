"""Risk-group thresholds computed once from training data, frozen and
applied unchanged to test/external data."""

import json
import numpy as np
import pandas as pd

import config


def lock_thresholds(train_risk_scores, quantiles=None) -> dict:
    quantiles = quantiles or config.RISK_GROUP_QUANTILES
    cutpoints = np.quantile(train_risk_scores, quantiles).tolist()
    locked = {"quantiles": quantiles, "cutpoints": cutpoints,
              "n_training_samples_used": int(len(train_risk_scores))}
    print(f"  Locked risk thresholds (training only): {locked}")
    return locked


def apply_thresholds(risk_scores, locked: dict) -> pd.Series:
    cutpoints = locked["cutpoints"]
    groups = np.digitize(risk_scores, cutpoints)
    return pd.Series(groups, name="risk_group")


def save_thresholds(locked: dict, path):
    with open(path, "w") as f:
        json.dump(locked, f, indent=2)


def load_thresholds(path) -> dict:
    with open(path) as f:
        return json.load(f)
