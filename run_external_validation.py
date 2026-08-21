"""External validation of the locked 3-gene Cox model on GEO cohorts.
No refitting occurs. Each cohort is self-standardized (own mean/std)
before scoring, since absolute expression scale is not comparable
across TCGA RNA-seq and GEO microarray platforms."""

import os
import json
import traceback

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "repo_artifacts")

COEFFICIENTS_FILE = os.path.join(ARTIFACT_DIR, "LOCKED_3gene_external_coefficients.csv")
THRESHOLD_FILE = os.path.join(ARTIFACT_DIR, "LOCKED_3gene_external_thresholds.json")

EXTERNAL_DIR = os.path.join(BASE_DIR, "external_validation")
OUTPUT_DIR = os.path.join(BASE_DIR, "external_validation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SYMBOL_TO_ENSEMBL = {
    "ANLN": "ENSG00000011426.11",
    "LDLRAD3": "ENSG00000179241.13",
    "ERO1L": "ENSG00000197930.13",
}
GENES = list(SYMBOL_TO_ENSEMBL.keys())

COHORTS = {
    "GSE31210": "GSE31210_external_validation.csv",
    "GSE30219": "GSE30219_external_validation.csv",
    "GSE72094": "GSE72094_external_validation.csv",
    "GSE50081": "GSE50081_external_validation.csv",
}


def load_locked_coefficients():
    if not os.path.exists(COEFFICIENTS_FILE):
        raise FileNotFoundError(f"{COEFFICIENTS_FILE} not found. Run lock_3gene_external_model.py first.")
    coeff_df = pd.read_csv(COEFFICIENTS_FILE)
    coeff_by_id = dict(zip(coeff_df["feature"], coeff_df["coefficient"]))
    coefficients = {}
    for symbol, ens_id in SYMBOL_TO_ENSEMBL.items():
        if ens_id not in coeff_by_id:
            raise KeyError(f"Coefficient for {ens_id} ({symbol}) not found. Available: {list(coeff_by_id.keys())}")
        coefficients[symbol] = coeff_by_id[ens_id]
    return coefficients


def load_locked_threshold():
    if not os.path.exists(THRESHOLD_FILE):
        print(f"WARNING: {THRESHOLD_FILE} not found. Risk groups will be 'Unclassified'.")
        return None
    with open(THRESHOLD_FILE) as f:
        return json.load(f)


def calculate_locked_risk(df, genes, coefficients):
    risk = np.zeros(len(df), dtype=float)
    for gene in genes:
        x = pd.to_numeric(df[gene], errors="coerce")
        std = x.std()
        if std == 0 or np.isnan(std):
            print(f"    WARNING: {gene} has zero/undefined variance in this cohort; contributing 0.")
            continue
        z = (x - x.mean()) / std
        risk += z.values * coefficients[gene]
    return risk


def assign_risk_groups(risk, thresholds):
    if thresholds is None or "cutpoints" not in thresholds:
        return pd.Series(["Unclassified"] * len(risk))
    cutpoints = thresholds["cutpoints"]
    group_idx = np.digitize(risk, cutpoints)
    if len(cutpoints) == 1:
        labels = {0: "Low", 1: "High"}
    elif len(cutpoints) == 2:
        labels = {0: "Low", 1: "Intermediate", 2: "High"}
    else:
        labels = {i: f"Group{i}" for i in range(len(cutpoints) + 1)}
    return pd.Series([labels[i] for i in group_idx])


def calculate_time_dependent_auc(df, risk):
    time = pd.to_numeric(df["survival_months"], errors="coerce").values
    event = pd.to_numeric(df["event"], errors="coerce").values
    valid = np.isfinite(time) & np.isfinite(event) & ((time >= 24) | ((time < 24) & (event == 1)))
    if valid.sum() == 0:
        return np.nan, 0
    y_24 = np.where((event == 1) & (time <= 24), 1, 0)
    y, r = y_24[valid], risk[valid]
    if len(np.unique(y)) < 2:
        return np.nan, int(valid.sum())
    return roc_auc_score(y, r), int(valid.sum())


def calculate_km_logrank(df, risk_groups):
    try:
        temp = df.copy()
        temp["risk_group"] = risk_groups.values
        temp["survival_months"] = pd.to_numeric(temp["survival_months"], errors="coerce")
        temp["event"] = pd.to_numeric(temp["event"], errors="coerce")
        temp = temp.dropna(subset=["survival_months", "event"])
        if temp["risk_group"].nunique() < 2:
            return np.nan
        result = multivariate_logrank_test(temp["survival_months"], temp["risk_group"], temp["event"])
        return result.p_value
    except Exception:
        return np.nan


def main():
    print("\n" + "=" * 70)
    print("LOCKED MODEL: GEO EXTERNAL VALIDATION")
    print("=" * 70)
    coefficients = load_locked_coefficients()
    for gene, coef in coefficients.items():
        print(f"{gene:<10} coefficient = {coef:.15f}")
    print("\nExcluded from GEO: hsa-mir-4435-2 -> not available on these platforms")
    print("Scoring: each cohort self-standardized before scoring.")

    print(f"\nLoading thresholds: {THRESHOLD_FILE}")
    thresholds = load_locked_threshold()
    if thresholds:
        print(json.dumps(thresholds, indent=2))

    all_metrics = []
    for cohort_name, filename in COHORTS.items():
        print("\n" + "=" * 70 + f"\n{cohort_name}\n" + "=" * 70)
        path = os.path.join(EXTERNAL_DIR, filename)
        try:
            external_df = pd.read_csv(path)
            print(f"Loaded: {path}\nOriginal samples: {len(external_df)}")

            required = ["patient_id"] + GENES + ["survival_months", "event"]
            missing_columns = [c for c in required if c not in external_df.columns]
            if missing_columns:
                print(f"SKIPPED: Missing columns: {missing_columns}")
                continue

            for gene in GENES:
                external_df[gene] = pd.to_numeric(external_df[gene], errors="coerce")
            external_df["survival_months"] = pd.to_numeric(external_df["survival_months"], errors="coerce")
            external_df["event"] = pd.to_numeric(external_df["event"], errors="coerce")

            external_df = external_df.dropna(subset=GENES + ["survival_months", "event"]).copy()
            print(f"Samples after dropping missing: {len(external_df)}")
            if len(external_df) == 0:
                print("SKIPPED: no complete samples.")
                continue

            print("Event distribution:", external_df["event"].value_counts().sort_index().to_dict())

            risk_external = calculate_locked_risk(external_df, GENES, coefficients)
            external_df["risk_score"] = risk_external
            print(f"\nRisk score: mean={np.mean(risk_external):.4f} sd={np.std(risk_external):.4f} "
                  f"min={np.min(risk_external):.4f} max={np.max(risk_external):.4f}")

            c_index = concordance_index(external_df["survival_months"], -external_df["risk_score"], external_df["event"])
            print(f"\nHarrell C-index = {c_index:.4f}")

            auc_24, n_auc = calculate_time_dependent_auc(external_df, risk_external)
            print(f"24-month AUC = {auc_24:.4f}" if np.isfinite(auc_24) else "24-month AUC = NA")

            risk_groups = assign_risk_groups(risk_external, thresholds)
            external_df["risk_group"] = risk_groups.values
            print("Risk group distribution:", external_df["risk_group"].value_counts().to_dict())

            logrank_p = calculate_km_logrank(external_df, risk_groups)
            print(f"Log-rank p = {logrank_p:.6g}" if np.isfinite(logrank_p) else "Log-rank p = NA")

            output_columns = ["patient_id"] + GENES + ["survival_months", "event", "risk_score", "risk_group"]
            cohort_output = os.path.join(OUTPUT_DIR, f"{cohort_name}_validation_results.csv")
            external_df[output_columns].to_csv(cohort_output, index=False)
            print(f"Saved: {cohort_output}")

            all_metrics.append({
                "cohort": cohort_name, "n_samples": len(external_df),
                "n_events": int(external_df["event"].sum()),
                "event_rate": float(external_df["event"].mean()),
                "harrell_c_index": float(c_index),
                "auc_24_month": float(auc_24) if np.isfinite(auc_24) else np.nan,
                "auc_24_n": n_auc,
                "logrank_p": float(logrank_p) if np.isfinite(logrank_p) else np.nan,
            })
        except Exception as e:
            print(f"ERROR in {cohort_name}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 70 + "\nFINAL EXTERNAL VALIDATION SUMMARY\n" + "=" * 70)
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_file = os.path.join(OUTPUT_DIR, "external_validation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(summary_df.to_string(index=False))
        print(f"\nSaved: {summary_file}")
    else:
        print("No cohorts were successfully validated.")


if __name__ == "__main__":
    main()
