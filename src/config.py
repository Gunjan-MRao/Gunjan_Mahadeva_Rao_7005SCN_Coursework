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

# ---------------------------------------------------------------------------
# Multi-method anomaly detection comparison (Phase 3 extension)
# ---------------------------------------------------------------------------
LOF_PARAMS = {
    "n_neighbors": 35,
    "novelty": True,          # allows scoring records not used in fit()
    "contamination": 0.02,
    "n_jobs": -1,
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": 0.02,                # analogous to expected contamination
    "gamma": "scale",
}
# One-Class SVM training scales poorly (O(n^2)-O(n^3)) with sample size, so
# fitting is subsampled to a manageable training set for computational
# tractability -- standard practice for kernel-SVM methods on large datasets.
# All records (not just the subsample) are still scored/flagged.
OCSVM_TRAIN_SAMPLE_SIZE = 8000

# "Autoencoder" implemented as a bottleneck MLP trained for input
# reconstruction (scikit-learn MLPRegressor) rather than a full deep-learning
# framework, to keep the pipeline dependency-light, CPU-only, and fully
# reproducible without GPU requirements. Architecture: 7 -> 4 -> 2 -> 4 -> 7,
# i.e. a genuine bottleneck (compression to 2 latent units) as in a standard
# autoencoder; anomaly score = reconstruction error (MSE).
AUTOENCODER_PARAMS = {
    "hidden_layer_sizes": (4, 2, 4),
    "activation": "tanh",
    "solver": "adam",
    "max_iter": 500,
    "random_state": RANDOM_STATE,
    "early_stopping": True,
    "n_iter_no_change": 15,
}
AUTOENCODER_ANOMALY_PERCENTILE = 98  # reconstruction-error percentile used to flag

METHOD_COMPARISON_PATH = DATA_PROCESSED_DIR / "method_comparison_scores.csv"
METHOD_COMPARISON_SUMMARY_PATH = DATA_PROCESSED_DIR / "method_comparison_summary.csv"

# ---------------------------------------------------------------------------
# Synthetic anomaly injection (quantitative precision/recall/F1 evaluation)
# ---------------------------------------------------------------------------
# Confidential audit ground truth is unavailable (see validation/audit_validation.py),
# so detector performance is additionally benchmarked against synthetic anomalies
# with known, literature-motivated fraud/error signatures injected into a held-out
# evaluation sample. This gives genuine precision/recall/F1/PR-AUC numbers,
# standard practice for evaluating unsupervised anomaly detectors absent labels.
SYNTHETIC_INJECTION_RATE = 0.02       # fraction of eval sample replaced with synthetic anomalies
SYNTHETIC_EVAL_SAMPLE_SIZE = 20000    # size of the (post-COVID) evaluation sample
SYNTHETIC_AMOUNT_INFLATION_RANGE = (5.0, 15.0)   # x-normal-amount multiplier for inflated invoices
SYNTHETIC_RESULTS_PATH = DATA_PROCESSED_DIR / "synthetic_injection_evaluation.csv"

# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
STATS_RESULTS_PATH = DATA_PROCESSED_DIR / "statistical_tests_summary.csv"

# ---------------------------------------------------------------------------
# Supplier-buyer network analysis
# ---------------------------------------------------------------------------
NETWORK_MIN_TRANSACTIONS = 2   # minimum trust-supplier transactions to include an edge
NETWORK_HUB_PERCENTILE = 95    # betweenness-centrality percentile defining a "hub" supplier
NETWORK_NODE_METRICS_PATH = DATA_PROCESSED_DIR / "network_supplier_metrics.csv"
NETWORK_COMMUNITY_PATH = DATA_PROCESSED_DIR / "network_communities.csv"
