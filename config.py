"""Central configuration: paths, seeds, hyperparameter grids, thresholds."""

from pathlib import Path

RANDOM_SEED = 42

DATA_DIR = Path("data")
CLINICAL_FILE = DATA_DIR / "lung_clinical_cleaned_1.csv"
MIRNA_FILE = DATA_DIR / "TCGA-LUAD_mirna_sorted_f.csv"
GENE_FILE = DATA_DIR / "merged_output.csv"

EXTERNAL_COHORT_FILES = {
    "GSE30219": DATA_DIR / "external" / "GSE30219_external_validation.csv",
    "GSE31210": DATA_DIR / "external" / "GSE31210_external_validation.csv",
    "GSE50081": DATA_DIR / "external" / "GSE50081_external_validation.csv",
    "GSE72094": DATA_DIR / "external" / "GSE72094_external_validation.csv",
}
EXTERNAL_MODEL_FEATURE_TYPE = "gene_only"

OUTPUT_DIR = Path("outputs")
ARTIFACT_DIR = Path("repo_artifacts")

TIME_COL = "survival_months"
EVENT_COL = "event"
EVAL_HORIZONS_MONTHS = [12, 24, 36, 60]

SCREEN_ALPHA_FDR = 0.05
SCREEN_METHOD = "fdr_bh"
MAX_SCREENED_FEATURES = 200
MAX_SCREENED_FEATURES_BY_GROUP = {"mirna": 50, "gene": 150}

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_OUTER_REPEATS = 1

COXNET_L1_RATIOS = [0.5, 0.9, 1.0]
COXNET_N_ALPHAS = 50
COXNET_ALPHA_MIN_RATIO = 0.01

RISK_GROUP_QUANTILES = [0.5]

COMPARATOR_SIGNATURES = {}  # populated at runtime from resolved_comparator_signatures.json
