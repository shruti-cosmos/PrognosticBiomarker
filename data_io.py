"""Loads clinical, miRNA, and gene expression data; builds the survival outcome."""

import numpy as np
import pandas as pd

import config


def _harmonize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"[A-Z]$", "", regex=True)


def build_survival_outcome(clinical: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical.copy()
    clinical["survival_months"] = np.where(
        clinical["vital_status"] == "Dead",
        clinical["days_to_death"] / 30.44,
        clinical["days_to_last_followup"] / 30.44,
    )
    clinical["event"] = (clinical["vital_status"] == "Dead").astype(int)

    n_before = len(clinical)
    clinical = clinical[clinical["survival_months"].notna() & (clinical["survival_months"] > 0)]
    print(f"  Dropped {n_before - len(clinical)} patients with missing/non-positive follow-up.")
    return clinical


def load_merged_dataset(
    clinical_file=None, mirna_file=None, gene_file=None,
    restrict_to_primary_tumor: bool = True, gene_usecols=None,
):
    clinical_file = clinical_file or config.CLINICAL_FILE
    mirna_file = mirna_file or config.MIRNA_FILE
    gene_file = gene_file or config.GENE_FILE

    print("=" * 80)
    print("LOADING RAW DATA")
    print("=" * 80)

    clinical = pd.read_csv(clinical_file)
    print(f"  Clinical: {clinical.shape}")

    mirna = pd.read_csv(mirna_file)
    print(f"  miRNA: {mirna.shape}")

    if gene_usecols is not None:
        cols = ["patient_id"] + [c for c in gene_usecols if c != "patient_id"]
        gene = pd.read_csv(gene_file, usecols=cols)
    else:
        gene = pd.read_csv(gene_file)
        if "Target" in gene.columns:
            gene = gene.drop(columns=["Target"])
    print(f"  Gene expression: {gene.shape}")

    if restrict_to_primary_tumor:
        for name, df in [("miRNA", mirna), ("gene", gene)]:
            code = df["patient_id"].astype(str).str[13:15]
            keep = code == "01"
            dropped = (~keep).sum()
            if dropped:
                print(f"  Dropping {dropped} non-primary-tumor rows from {name} file.")
            df.drop(df.index[~keep], inplace=True)

    clinical["patient_id"] = _harmonize_id(clinical["patient_id"])
    mirna["patient_id"] = _harmonize_id(mirna["patient_id"])
    gene["patient_id"] = _harmonize_id(gene["patient_id"])

    clinical = build_survival_outcome(clinical)

    merged = clinical.merge(mirna, on="patient_id", how="inner", suffixes=("", "_mirna"))
    merged = merged.merge(gene, on="patient_id", how="inner", suffixes=("", "_gene"))
    merged = merged.drop_duplicates(subset="patient_id").reset_index(drop=True)

    print(f"  Final merged cohort: {merged.shape[0]} patients, "
          f"{merged['event'].sum()} events ({merged['event'].mean()*100:.1f}% event rate)")

    mirna_features = [c for c in mirna.columns if c not in ("patient_id", "Recurrence_or_not")]
    gene_features = [c for c in gene.columns if c != "patient_id"]

    return merged, mirna_features, gene_features


if __name__ == "__main__":
    merged, mirna_feats, gene_feats = load_merged_dataset()
    print(merged[[config.TIME_COL, config.EVENT_COL]].describe())
