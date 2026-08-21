"""Saves run manifest, fold results, feature lists, coefficients,
thresholds, and patient ID lists for the public repository."""

import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import sksurv
import lifelines

import config


def _versions():
    return {
        "python": sys.version, "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "scikit-learn": sklearn.__version__, "scikit-survival": sksurv.__version__,
        "lifelines": lifelines.__version__,
    }


def save_run_manifest(out_dir, fold_results, locked_model, locked_thresholds,
                       train_patient_ids, selected_features_per_fold=None,
                       test_patient_ids=None, external_patient_ids=None, notes=""):
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.RANDOM_SEED,
        "n_outer_folds": config.N_OUTER_FOLDS, "n_inner_folds": config.N_INNER_FOLDS,
        "screen_alpha_fdr": config.SCREEN_ALPHA_FDR, "screen_method": config.SCREEN_METHOD,
        "max_screened_features": config.MAX_SCREENED_FEATURES,
        "coxnet_l1_ratios_tried": config.COXNET_L1_RATIOS,
        "risk_group_quantiles": config.RISK_GROUP_QUANTILES,
        "eval_horizons_months": config.EVAL_HORIZONS_MONTHS,
        "software_versions": _versions(), "notes": notes,
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    fold_results.drop(columns=["selected_features"], errors="ignore").to_csv(
        out_dir / "nested_cv_fold_results.csv", index=False
    )
    if selected_features_per_fold:
        with open(out_dir / "nested_cv_selected_features_per_fold.json", "w") as f:
            json.dump({int(k): v for k, v in selected_features_per_fold.items()}, f, indent=2)

    coefs = locked_model.coefficients_at_best_alpha()
    coefs[coefs != 0].to_csv(out_dir / "locked_model_coefficients.csv", header=["coefficient"])

    with open(out_dir / "locked_risk_thresholds.json", "w") as f:
        json.dump(locked_thresholds, f, indent=2)

    pd.Series(sorted(train_patient_ids), name="patient_id").to_csv(out_dir / "train_patient_ids.csv", index=False)
    if test_patient_ids is not None:
        pd.Series(sorted(test_patient_ids), name="patient_id").to_csv(out_dir / "test_patient_ids.csv", index=False)
    if external_patient_ids is not None:
        pd.Series(sorted(external_patient_ids), name="patient_id").to_csv(out_dir / "external_patient_ids.csv", index=False)

    print(f"  Reproducibility artifacts written to: {out_dir}/")
    return manifest
