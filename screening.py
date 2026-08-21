"""Univariate Cox screening with Benjamini-Hochberg FDR correction.

Called only on training-fold data so it can be nested inside cross-
validation. screen_features_by_group applies a separate FDR correction
per feature type (e.g. miRNA vs. gene), preventing a numerically larger
feature space from diluting a smaller, comparably informative one.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from statsmodels.stats.multitest import multipletests

import config

warnings.filterwarnings("ignore", category=UserWarning, module="lifelines")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _univariate_cox_pvalue(feature, df, time_col, event_col):
    sub = df[[time_col, event_col, feature]].dropna()
    if sub[feature].std() == 0 or len(sub) < 20 or sub[feature].nunique() < 5:
        return feature, np.nan, np.nan, np.nan
    try:
        sub = sub.copy()
        sub[feature] = (sub[feature] - sub[feature].mean()) / sub[feature].std()
        cph = CoxPHFitter()
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        hr = cph.hazard_ratios_[feature]
        p = cph.summary.loc[feature, "p"]
        se = cph.summary.loc[feature, "se(coef)"]
        return feature, hr, p, se
    except (ConvergenceError, np.linalg.LinAlgError, Exception):
        return feature, np.nan, np.nan, np.nan


def screen_features(train_df, candidate_features, time_col=None, event_col=None,
                     alpha_fdr=None, max_features=None, n_jobs=-1):
    time_col = time_col or config.TIME_COL
    event_col = event_col or config.EVENT_COL
    alpha_fdr = alpha_fdr if alpha_fdr is not None else config.SCREEN_ALPHA_FDR
    max_features = max_features or config.MAX_SCREENED_FEATURES

    out = Parallel(n_jobs=n_jobs)(
        delayed(_univariate_cox_pvalue)(f, train_df, time_col, event_col)
        for f in candidate_features
    )
    results = pd.DataFrame(out, columns=["feature", "HR", "p", "se"]).dropna(subset=["p"]).reset_index(drop=True)

    reject, q, _, _ = multipletests(results["p"].values, alpha=alpha_fdr, method=config.SCREEN_METHOD)
    results["q"] = q
    results["significant"] = reject

    sig = results[results["significant"]].copy()
    sig["abs_log_hr"] = np.abs(np.log(sig["HR"].clip(lower=1e-8)))
    sig = sig.sort_values(["q", "abs_log_hr"], ascending=[True, False])

    selected = sig["feature"].head(max_features).tolist()
    print(f"  Screened {len(results)} features -> {len(sig)} significant at "
          f"FDR<{alpha_fdr} -> kept top {len(selected)}.")
    return results.sort_values("p"), selected


def screen_features_by_group(train_df, feature_groups, time_col=None, event_col=None,
                              alpha_fdr=None, max_features_per_group=None, n_jobs=-1):
    time_col = time_col or config.TIME_COL
    event_col = event_col or config.EVENT_COL
    alpha_fdr = alpha_fdr if alpha_fdr is not None else config.SCREEN_ALPHA_FDR
    max_features_per_group = max_features_per_group or {}

    pooled_selected = []
    for group_name, features in feature_groups.items():
        cap = max_features_per_group.get(group_name, config.MAX_SCREENED_FEATURES)
        print(f"  [{group_name}] screening {len(features)} candidates (own FDR universe)...")
        _, selected = screen_features(train_df, features, time_col, event_col, alpha_fdr, cap, n_jobs)
        pooled_selected.extend(selected)
        print(f"    {group_name}: {len(selected)} selected")

    return None, pooled_selected
