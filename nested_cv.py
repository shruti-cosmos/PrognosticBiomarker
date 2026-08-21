"""Fully nested cross-validation for feature screening and Coxnet
hyperparameter selection. Screening runs once per outer fold on the
full outer-training data; l1_ratio and alpha are then tuned via inner
folds on that fixed feature set, and the outer test fold is scored once."""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_ipcw

import config
import evaluation
import screening
import survival_models


def _tune_l1_and_alpha(train_out: pd.DataFrame, final_selected: list, seed: int):
    y_train_out = survival_models.to_structured_y(train_out)
    inner_splitter = StratifiedKFold(n_splits=config.N_INNER_FOLDS, shuffle=True, random_state=seed)
    inner_splits = list(inner_splitter.split(train_out, train_out[config.EVENT_COL]))

    best = {"score": -np.inf}

    for l1_ratio in config.COXNET_L1_RATIOS:
        t0 = time.time()
        probe = survival_models.ScaledCoxnet(l1_ratio=l1_ratio)
        try:
            probe.fit(train_out[final_selected], y_train_out)
        except ArithmeticError as e:
            print(f"    l1_ratio={l1_ratio}: probe fit failed ({e}); skipping.", flush=True)
            continue
        shared_alphas = probe.model.alphas_

        inner_scores = np.zeros(len(shared_alphas))
        n_valid_folds = 0

        for tr_idx, val_idx in inner_splits:
            inner_train = train_out.iloc[tr_idx]
            inner_val = train_out.iloc[val_idx]
            y_inner_train = survival_models.to_structured_y(inner_train)
            y_inner_val = survival_models.to_structured_y(inner_val)

            m = survival_models.ScaledCoxnet(l1_ratio=l1_ratio, alphas=shared_alphas)
            try:
                m.fit(inner_train[final_selected], y_inner_train)
            except ArithmeticError:
                continue

            Xs_val = m.scaler.transform(inner_val[final_selected].values)
            risk_path = m.model.predict(Xs_val, alpha=None)
            if risk_path.ndim == 1:
                risk_path = risk_path[:, None]

            for j in range(risk_path.shape[1]):
                risk_j = risk_path[:, j]
                if np.allclose(risk_j, risk_j[0]):
                    continue
                try:
                    c = concordance_index_ipcw(y_inner_train, y_inner_val, risk_j)[0]
                except Exception:
                    c = np.nan
                if not np.isnan(c):
                    inner_scores[j] += c
            n_valid_folds += 1

        if n_valid_folds == 0:
            continue
        inner_scores /= n_valid_folds
        if not np.any(inner_scores > 0):
            print(f"    l1_ratio={l1_ratio}: no alpha scored above 0; skipping.", flush=True)
            continue

        best_alpha_idx = int(np.nanargmax(inner_scores))
        print(f"    l1_ratio={l1_ratio}: best inner IPCW-C={inner_scores[best_alpha_idx]:.3f} "
              f"at alpha={shared_alphas[best_alpha_idx]:.5g}  ({time.time()-t0:.0f}s)", flush=True)

        if inner_scores[best_alpha_idx] > best["score"]:
            best = {"score": inner_scores[best_alpha_idx], "l1_ratio": l1_ratio,
                    "alpha": shared_alphas[best_alpha_idx]}

    return best


def nested_cv_survival(df, mirna_features, gene_features, n_outer_folds=None, seed=None, checkpoint_path=None):
    n_outer_folds = n_outer_folds or config.N_OUTER_FOLDS
    seed = seed if seed is not None else config.RANDOM_SEED

    outer_splitter = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    splits = list(outer_splitter.split(df, df[config.EVENT_COL]))

    done_folds = set()
    fold_records = []
    if checkpoint_path is not None and checkpoint_path.exists():
        prev = pd.read_csv(checkpoint_path)
        fold_records = prev.to_dict("records")
        done_folds = set(prev["fold"].tolist())
        print(f"Resuming: folds already completed = {sorted(done_folds)}", flush=True)

    oof_risk = pd.Series(index=df.index, dtype=float)
    selected_features_per_fold = {}

    for fold_i, (tr_idx, te_idx) in enumerate(splits, start=1):
        if fold_i in done_folds:
            print(f"--- Outer fold {fold_i}/{n_outer_folds}: already done, skipping ---", flush=True)
            continue

        print(f"\n--- Outer fold {fold_i}/{n_outer_folds} ---", flush=True)
        t_fold = time.time()
        train_out = df.iloc[tr_idx].reset_index(drop=True)
        test_out = df.iloc[te_idx].reset_index(drop=True)

        t0 = time.time()
        _, final_selected = screening.screen_features_by_group(
            train_out, feature_groups={"mirna": mirna_features, "gene": gene_features},
            max_features_per_group=config.MAX_SCREENED_FEATURES_BY_GROUP,
        )
        print(f"  Screening took {time.time()-t0:.0f}s -> {len(final_selected)} features", flush=True)
        if len(final_selected) < 3:
            print("  WARNING: too few features survived screening; skipping fold.", flush=True)
            continue

        best = _tune_l1_and_alpha(train_out, final_selected, seed=seed + fold_i)
        if "l1_ratio" not in best:
            print("  WARNING: hyperparameter tuning failed on this fold; skipping.", flush=True)
            continue

        y_train_out = survival_models.to_structured_y(train_out)
        y_test_out = survival_models.to_structured_y(test_out)

        final_model = survival_models.ScaledCoxnet(l1_ratio=best["l1_ratio"], alphas=[best["alpha"]])
        final_model.fit(train_out[final_selected], y_train_out)
        n_nonzero = int((final_model.coefficients_at_best_alpha() != 0).sum())

        if n_nonzero == 0:
            print("  WARNING: 0 nonzero coefficients at selected alpha; backing off.", flush=True)
            probe = survival_models.ScaledCoxnet(l1_ratio=best["l1_ratio"])
            try:
                probe.fit(train_out[final_selected], y_train_out)
                cand, a = final_model.refit_at_alpha_with_backoff(
                    train_out[final_selected], y_train_out, best["alpha"], probe.model.alphas_
                )
                if cand is not None:
                    final_model = cand
                    best["alpha"] = a
                    print(f"    Recovered non-degenerate fit at alpha={a:.5g}.", flush=True)
            except ArithmeticError as e:
                print(f"    Probe refit failed ({e}); keeping the all-zero model.", flush=True)

        nz = final_model.coefficients_at_best_alpha()
        nz = nz[nz != 0]
        selected_features_per_fold[fold_i] = {
            "screened_pool": final_selected,
            "nonzero_this_fold": nz.to_dict(),
        }

        risk_test = final_model.predict_risk(test_out[final_selected])
        metrics = evaluation.evaluate_risk_scores(y_train_out, y_test_out, risk_test)
        metrics.update({
            "fold": fold_i, "l1_ratio": best["l1_ratio"], "alpha": best["alpha"],
            "n_selected_features": len(final_selected),
            "n_nonzero_coefficients": int((final_model.coefficients_at_best_alpha() != 0).sum()),
            "inner_cv_score": best["score"], "fold_seconds": time.time() - t_fold,
        })
        fold_records.append(metrics)
        oof_risk.iloc[te_idx] = risk_test

        print(f"  Fold {fold_i} done in {time.time()-t_fold:.0f}s | "
              f"features={len(final_selected)} nonzero={metrics['n_nonzero_coefficients']} "
              f"| Harrell C={metrics['harrell_c_index']:.3f} | IPCW C={metrics['ipcw_c_index']:.3f}", flush=True)

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(fold_records).drop(columns=["time_dependent_auc"], errors="ignore").to_csv(
                checkpoint_path, index=False
            )

    results = pd.DataFrame(fold_records)
    return results, oof_risk, selected_features_per_fold
