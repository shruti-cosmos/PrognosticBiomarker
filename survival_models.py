"""Survival-specific models that correctly account for censoring.

ScaledCoxnet: elastic-net Cox with internal StandardScaler fit on
training data only. Includes automatic backoff for numerically unstable
low-alpha regions of the regularization path.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

import config


def to_structured_y(df: pd.DataFrame, time_col: str = None, event_col: str = None):
    time_col = time_col or config.TIME_COL
    event_col = event_col or config.EVENT_COL
    return Surv.from_dataframe(event_col, time_col, df)


class ScaledCoxnet:
    def __init__(self, l1_ratio=0.9, alphas=None, alpha_min_ratio=None):
        self.l1_ratio = l1_ratio
        self.alphas = alphas
        self.alpha_min_ratio = alpha_min_ratio if alpha_min_ratio is not None else config.COXNET_ALPHA_MIN_RATIO
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y_struct):
        self.feature_names_ = list(X.columns)
        Xs = self.scaler.fit_transform(X.values)

        if self.alphas is not None:
            self.model = CoxnetSurvivalAnalysis(
                l1_ratio=self.l1_ratio, alphas=self.alphas,
                fit_baseline_model=True, max_iter=100000,
            )
            try:
                self.model.fit(Xs, y_struct)
            except ArithmeticError as e:
                raise ArithmeticError(
                    f"Coxnet failed on supplied fixed alpha grid ({e})."
                ) from e
            return self

        backoff_schedule = [self.alpha_min_ratio, 0.05, 0.1, 0.2, 0.5]
        last_err = None
        for i, amr in enumerate(backoff_schedule):
            try:
                self.model = CoxnetSurvivalAnalysis(
                    l1_ratio=self.l1_ratio, alphas=None, alpha_min_ratio=amr,
                    n_alphas=config.COXNET_N_ALPHAS, fit_baseline_model=True,
                    max_iter=100000,
                )
                self.model.fit(Xs, y_struct)
                if i > 0:
                    print(f"    [ScaledCoxnet] converged at alpha_min_ratio={amr} "
                          f"(l1_ratio={self.l1_ratio}, n={Xs.shape[0]}, p={Xs.shape[1]}).")
                return self
            except ArithmeticError as e:
                last_err = e
                continue

        raise ArithmeticError(
            f"Coxnet did not converge for l1_ratio={self.l1_ratio} after "
            f"backoff through {backoff_schedule}. Last error: {last_err}."
        )

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.feature_names_].values)
        return self.model.predict(Xs)

    def coefficients_at_best_alpha(self, alpha_idx=-1) -> pd.Series:
        coefs = self.model.coef_[:, alpha_idx]
        return pd.Series(coefs, index=self.feature_names_).sort_values(key=np.abs, ascending=False)

    def refit_at_alpha_with_backoff(self, X, y_struct, target_alpha, alphas_path):
        alphas_sorted = np.sort(alphas_path)[::-1]
        start_idx = int(np.argmin(np.abs(alphas_sorted - target_alpha)))
        for a in alphas_sorted[start_idx:]:
            try:
                cand = ScaledCoxnet(l1_ratio=self.l1_ratio, alphas=[a])
                cand.fit(X, y_struct)
            except ArithmeticError:
                continue
            if (cand.coefficients_at_best_alpha() != 0).sum() > 0:
                return cand, a
        return None, None


def fit_random_survival_forest(X, y_struct, n_estimators=300, max_depth=None, seed=None):
    seed = seed if seed is not None else config.RANDOM_SEED
    rsf = RandomSurvivalForest(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=10, n_jobs=-1, random_state=seed,
    )
    rsf.fit(X.values, y_struct)
    return rsf
