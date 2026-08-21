"""Builds per-cohort external validation CSVs from GEO expression and
clinical files. Expression files are wide (probes x samples); multiple
probes per gene are averaged. Run --diagnose first to verify event/time
column coding before trusting the mapped output."""

import os
import sys

import numpy as np
import pandas as pd

GENES = ["ANLN", "LDLRAD3", "ERO1L"]

COHORTS = {
    "GSE31210": {
        "clinical_file": "GSE31210_clinical.csv", "id_col": "sample_id",
        "time_col": "months before death/censor", "time_is_days": False,
        "event_col": "death", "dead_values": {"1", "dead", "death", "yes", "true"},
    },
    "GSE30219": {
        "clinical_file": "GSE30219_clinical.csv", "id_col": "sample_id",
        "time_col": "follow-up time (months)", "time_is_days": False,
        "event_col": "status", "dead_values": {"1", "dead", "death", "deceased", "yes", "true"},
    },
    "GSE72094": {
        "clinical_file": "GSE72094_clinical.csv", "id_col": "sample_id",
        "time_col": "survival_time_in_days", "time_is_days": True,
        "event_col": "vital_status", "dead_values": {"1", "dead", "death", "deceased", "yes", "true"},
    },
    "GSE50081": {
        "clinical_file": "GSE50081_clinical.csv", "id_col": "sample_id",
        "time_col": "survival time", "time_is_days": False,  # verify via --diagnose
        "event_col": "status", "dead_values": {"1", "dead", "death", "deceased", "yes", "true"},
    },
}

EXPRESSION_FILE_PATTERN = "{gse}_{gene}_expression.csv"
OUT_DIR = "external_validation"


def load_gene_expression(gse: str, gene: str) -> pd.Series:
    path = EXPRESSION_FILE_PATTERN.format(gse=gse, gene=gene)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expression file: {path}")

    df = pd.read_csv(path, index_col=0)
    df = df[~df.index.astype(str).str.upper().eq("ID_REF")]
    df = df.apply(pd.to_numeric, errors="coerce")

    n_probes = df.shape[0]
    gene_series = df.mean(axis=0, skipna=True)
    gene_series.name = gene
    print(f"    {gene}: {n_probes} probe row(s) averaged -> {gene_series.notna().sum()} samples")
    return gene_series


def load_cohort_expression(gse: str) -> pd.DataFrame:
    print(f"  Loading expression for {gse}...")
    series_list = [load_gene_expression(gse, gene) for gene in GENES]
    expr_df = pd.concat(series_list, axis=1)
    expr_df.index.name = "sample_id"
    return expr_df


def _dedupe_columns(cols):
    seen, out = {}, []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def load_clinical(gse: str, cfg: dict) -> pd.DataFrame:
    path = cfg["clinical_file"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing clinical file: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = _dedupe_columns(df.columns.tolist())
    return df


def run_diagnostics(gse: str, clinical_df: pd.DataFrame, cfg: dict):
    print(f"\n  --- Diagnostics for {gse} ---")
    ev_col, tm_col = cfg["event_col"], cfg["time_col"]
    if ev_col in clinical_df.columns:
        print(f"  Event column '{ev_col}':\n{clinical_df[ev_col].value_counts(dropna=False).to_string()}")
    else:
        print(f"  WARNING: '{ev_col}' not found. Columns: {list(clinical_df.columns)}")
    if tm_col in clinical_df.columns:
        print(f"  Time column '{tm_col}' samples: {clinical_df[tm_col].dropna().head(10).tolist()}")
    else:
        print(f"  WARNING: '{tm_col}' not found. Columns: {list(clinical_df.columns)}")


def build_survival_columns(clinical_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = clinical_df.copy()
    time_raw = pd.to_numeric(out[cfg["time_col"]], errors="coerce")
    out["survival_months"] = time_raw / 30.44 if cfg["time_is_days"] else time_raw

    ev_raw = out[cfg["event_col"]].astype(str).str.strip().str.lower()
    out["event"] = ev_raw.isin({v.lower() for v in cfg["dead_values"]}).astype(int)
    out.loc[out[cfg["event_col"]].isna(), "event"] = np.nan

    out = out.rename(columns={cfg["id_col"]: "sample_id"})
    return out[["sample_id", "survival_months", "event"]]


def build_one_cohort(gse: str, cfg: dict, run_diagnostics_only: bool = False):
    print(f"\n{'='*70}\nProcessing {gse}\n{'='*70}")
    clinical_df = load_clinical(gse, cfg)
    run_diagnostics(gse, clinical_df, cfg)

    if run_diagnostics_only:
        return None

    surv_df = build_survival_columns(clinical_df, cfg)
    expr_df = load_cohort_expression(gse)
    merged = surv_df.merge(expr_df, on="sample_id", how="inner")

    n_before = len(merged)
    merged = merged.dropna(subset=["survival_months", "event"] + GENES)
    merged = merged[merged["survival_months"] > 0]
    n_after = len(merged)

    dup_ids = merged["sample_id"][merged["sample_id"].duplicated()].tolist()
    if dup_ids:
        print(f"  WARNING: {len(dup_ids)} duplicated sample_id(s): {dup_ids[:5]}...")
        merged = merged.drop_duplicates(subset="sample_id", keep="first")

    print(f"\n  Merged: {n_before} -> {n_after} samples after dropping missing values.")
    print(f"  Final: n={len(merged)}, events={int(merged['event'].sum())} "
          f"({merged['event'].mean()*100:.1f}% event rate)")

    merged = merged.rename(columns={"sample_id": "patient_id"})
    merged = merged[["patient_id"] + GENES + ["survival_months", "event"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{gse}_external_validation.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Wrote: {out_path}")
    return merged


def main():
    diagnostics_only = "--diagnose" in sys.argv
    summary_rows = []
    for gse, cfg in COHORTS.items():
        try:
            result = build_one_cohort(gse, cfg, run_diagnostics_only=diagnostics_only)
            if result is not None:
                summary_rows.append({
                    "cohort": gse, "n_patients": len(result),
                    "n_events": int(result["event"].sum()),
                    "event_rate_pct": round(result["event"].mean() * 100, 1),
                })
        except FileNotFoundError as e:
            print(f"\n  SKIPPING {gse}: {e}")

    if summary_rows and not diagnostics_only:
        summary_df = pd.DataFrame(summary_rows)
        print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(OUT_DIR, "cohort_summary.csv"), index=False)

    if diagnostics_only:
        print("\nDiagnostics-only run complete. Verify columns above, edit CONFIG if needed, then rerun without --diagnose.")


if __name__ == "__main__":
    main()
