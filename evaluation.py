"""Censoring-aware evaluation: IPCW C-index, time-dependent AUC, Brier
score, PH assumption testing, calibration by risk group."""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sksurv.metrics import brier_score, concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc

import config


def diagnose_censoring_at_horizons(df, horizons=None, time_col=None, event_col=None):
    horizons = horizons or config.EVAL_HORIZONS_MONTHS
    time_col = time_col or config.TIME_COL
    event_col = event_col or config.EVENT_COL

    rows = []
    for h in horizons:
        event_before = ((df[time_col] <= h) & (df[event_col] == 1)).sum()
        censored_before = ((df[time_col] <= h) & (df[event_col] == 0)).sum()
        known_after = (df[time_col] > h).sum()
        rows.append({
            "horizon_months": h, "n_event_before_horizon": event_before,
            "n_censored_before_horizon_unknown_status": censored_before,
            "n_known_event_free_at_horizon": known_after,
            "pct_unknown_status": 100 * censored_before / len(df),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def evaluate_risk_scores(y_train_struct, y_eval_struct, risk_scores_eval, horizons=None):
    horizons = np.asarray(horizons or config.EVAL_HORIZONS_MONTHS, dtype=float)
    time_field_name = [n for n in y_train_struct.dtype.names if n != y_train_struct.dtype.names[0]][0]
    max_train_time = y_train_struct[time_field_name].max()
    horizons = horizons[horizons < max_train_time]

    results = {}
    event_field, time_field = y_eval_struct.dtype.names
    c_harrell = concordance_index_censored(
        y_eval_struct[event_field], y_eval_struct[time_field], risk_scores_eval
    )[0]
    results["harrell_c_index"] = c_harrell

    c_ipcw = concordance_index_ipcw(y_train_struct, y_eval_struct, risk_scores_eval)[0]
    results["ipcw_c_index"] = c_ipcw

    aucs, mean_auc = cumulative_dynamic_auc(y_train_struct, y_eval_struct, risk_scores_eval, horizons)
    results["time_dependent_auc"] = dict(zip(horizons.tolist(), aucs.tolist()))
    results["mean_auc"] = mean_auc
    results["horizons_used_months"] = horizons.tolist()
    return results


def brier_and_ibs(y_train_struct, y_eval_struct, surv_funcs_eval, horizons=None):
    horizons = np.asarray(horizons or config.EVAL_HORIZONS_MONTHS, dtype=float)
    times, scores = brier_score(y_train_struct, y_eval_struct, surv_funcs_eval, horizons)
    ibs = np.trapz(scores, times) / (times[-1] - times[0])
    return dict(zip(times.tolist(), scores.tolist())), ibs


def check_ph_assumptions(cph: CoxPHFitter, df: pd.DataFrame, time_col=None, event_col=None):
    return cph.check_assumptions(df, p_value_threshold=0.05, show_plots=False)


def calibration_by_risk_group(fitted_survival_fn, df, risk_groups, horizon_months, time_col=None, event_col=None):
    time_col = time_col or config.TIME_COL
    event_col = event_col or config.EVENT_COL

    rows = []
    for grp in sorted(risk_groups.unique()):
        mask = risk_groups == grp
        sub = df.loc[mask]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], sub[event_col])
        observed = kmf.predict(horizon_months)
        predicted = np.mean([fitted_survival_fn(i)(horizon_months) for i in sub.index]) \
            if callable(fitted_survival_fn) else np.nan
        rows.append({
            "risk_group": grp, "n": mask.sum(),
            "observed_KM_survival_at_horizon": observed,
            "mean_predicted_survival_at_horizon": predicted,
        })
    return pd.DataFrame(rows)
