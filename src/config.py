"""
Central configuration for the NHS Procurement Anomaly Detection project (7005SCN).

All file paths, COVID period boundaries, and model hyperparameters live here so
every pipeline stage (data engineering -> EDA -> modelling -> validation) reads
from a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

for _p in (DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "bradford": DATA_RAW_DIR / "bradford_clean.csv",
    "lincolnshire": DATA_RAW_DIR / "lincolnshire_clean.csv",
    "nhs_england": DATA_RAW_DIR / "nhs_england_clean.csv",
    "contracts_finder": DATA_RAW_DIR / "contracts_clean.csv",
}

MASTER_PANEL_PATH = DATA_PROCESSED_DIR / "master_procurement_panel.csv"
ANOMALY_SCORES_PATH = DATA_PROCESSED_DIR / "anomaly_scores.csv"
SHAP_VALUES_PATH = DATA_PROCESSED_DIR / "shap_top_anomalies.csv"
VALIDATION_PATH = DATA_PROCESSED_DIR / "validation_redflags.csv"
HHI_PATH = DATA_PROCESSED_DIR / "hhi_monthly.csv"
STL_PATH = DATA_PROCESSED_DIR / "stl_shock_summary.csv"

# ---------------------------------------------------------------------------
# COVID-19 period boundaries (UK) — used to split pre / during / post COVID
# Pre-COVID:  2019-01-01 -> 2020-03-22  (day before first UK national lockdown)
# COVID:      2020-03-23 -> 2022-02-23  (day before "living with COVID" plan)
# Post-COVID: 2022-02-24 -> 2024-12-31
# ---------------------------------------------------------------------------
PRE_COVID_START = "2019-01-01"
PRE_COVID_END = "2020-03-22"
COVID_START = "2020-03-23"
COVID_END = "2022-02-23"
POST_COVID_START = "2022-02-24"
POST_COVID_END = "2024-12-31"

# STL baseline year used to quantify the COVID-19 shock (methodology stage 3)
STL_BASELINE_START = "2018-07-01"  # extended slightly to give STL enough seasonal cycles
STL_BASELINE_END = "2019-12-31"

# ---------------------------------------------------------------------------
# Modelling hyperparameters
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": 0.02,   # ~2% of pre-COVID transactions treated as the tail baseline
    "random_state": 42,
    "n_jobs": -1,
}

# Percentile threshold used to flag anomalies once scored on covid/post-covid data
ANOMALY_SCORE_PERCENTILE = 98

# Random seed used across the project for reproducibility
RANDOM_STATE = 42
