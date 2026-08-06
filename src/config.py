"""Central configuration for the NHS procurement anomaly-detection study.

Defines filesystem locations, study-period boundaries, reproducibility controls,
and model parameters shared across data engineering, analysis, and validation.
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

# ---------------------------------------------------------------------------
# Phase 1: raw-data acquisition
# ---------------------------------------------------------------------------
# Monthly FOI exports and Contracts Finder extracts are externally hosted.
# ``RAW_FILES`` identifies the consolidated Phase 2 inputs; ``RAW_STAGING_DIR``
# retains downloaded source artefacts and is excluded from version control.
GOOGLE_DRIVE_RAW_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1giZvxWA-uqWj8n8DcqOcOwKWh6ZFxVtS?usp=sharing"
)
RAW_STAGING_DIR = ROOT_DIR / "data" / "_raw_staging"

MASTER_PANEL_PATH = DATA_PROCESSED_DIR / "master_procurement_panel.csv"
ANOMALY_SCORES_PATH = DATA_PROCESSED_DIR / "anomaly_scores.csv"
SHAP_VALUES_PATH = DATA_PROCESSED_DIR / "shap_top_anomalies.csv"
VALIDATION_PATH = DATA_PROCESSED_DIR / "validation_redflags.csv"
HHI_PATH = DATA_PROCESSED_DIR / "hhi_monthly.csv"
STL_PATH = DATA_PROCESSED_DIR / "stl_shock_summary.csv"

# ---------------------------------------------------------------------------
# UK COVID-19 study-period boundaries
# Pre-COVID:  2019-01-01 to 2020-03-22 (before the first national lockdown)
# COVID:      2020-03-23 to 2022-02-23 (before the "Living with COVID" plan)
# Post-COVID: 2022-02-24 -> 2024-12-31
# ---------------------------------------------------------------------------
PRE_COVID_START = "2019-01-01"
PRE_COVID_END = "2020-03-22"
COVID_START = "2020-03-23"
COVID_END = "2022-02-23"
POST_COVID_START = "2022-02-24"
POST_COVID_END = "2024-12-31"

# STL baseline period for quantifying the COVID-19 expenditure shock
STL_BASELINE_START = "2018-07-01"  # Extends the baseline to support seasonal decomposition.
STL_BASELINE_END = "2019-12-31"

# ---------------------------------------------------------------------------
# Anomaly-detection model parameters
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": 0.02,   # Assumes 2% of pre-COVID transactions form the anomalous tail.
    "random_state": 42,
    "n_jobs": -1,
}

# Percentile threshold for flagging COVID and post-COVID anomaly scores
ANOMALY_SCORE_PERCENTILE = 98

# Shared pseudorandom seed for reproducible estimation
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Multi-method anomaly-detection comparison
# ---------------------------------------------------------------------------
LOF_PARAMS = {
    "n_neighbors": 35,
    "novelty": True,          # Permits scoring records not included in model fitting.
    "contamination": 0.02,
    "n_jobs": -1,
}

OCSVM_PARAMS = {
    "kernel": "rbf",
    "nu": 0.02,                # Sets an expected outlier proportion analogous to contamination.
    "gamma": "scale",
}
# One-Class SVM fitting has quadratic-to-cubic sample-size complexity; fitting
# is therefore subsampled, while all records are retained for scoring.
OCSVM_TRAIN_SAMPLE_SIZE = 8000

# The autoencoder is a symmetric bottleneck MLP (7→4→2→4→7) estimated with
# ``MLPRegressor``. Reconstruction mean-squared error is the anomaly measure.
AUTOENCODER_PARAMS = {
    "hidden_layer_sizes": (4, 2, 4),
    "activation": "tanh",
    "solver": "adam",
    "max_iter": 500,
    "random_state": RANDOM_STATE,
    "early_stopping": True,
    "n_iter_no_change": 15,
}
AUTOENCODER_ANOMALY_PERCENTILE = 98  # Reconstruction-error percentile for flagging.

METHOD_COMPARISON_PATH = DATA_PROCESSED_DIR / "method_comparison_scores.csv"
METHOD_COMPARISON_SUMMARY_PATH = DATA_PROCESSED_DIR / "method_comparison_summary.csv"

# ---------------------------------------------------------------------------
# Synthetic anomaly injection for precision, recall, and F1 evaluation
# ---------------------------------------------------------------------------
# In the absence of confidential audit labels, known synthetic fraud/error
# signatures support quantitative evaluation on a held-out sample.
SYNTHETIC_INJECTION_RATE = 0.02       # Evaluation-sample share replaced by synthetic anomalies.
SYNTHETIC_EVAL_SAMPLE_SIZE = 20000    # Post-COVID evaluation-sample size.
SYNTHETIC_AMOUNT_INFLATION_RANGE = (5.0, 15.0)   # Multiplier applied to inflated invoices.
SYNTHETIC_RESULTS_PATH = DATA_PROCESSED_DIR / "synthetic_injection_evaluation.csv"

# ---------------------------------------------------------------------------
# Statistical-inference settings
# ---------------------------------------------------------------------------
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
STATS_RESULTS_PATH = DATA_PROCESSED_DIR / "statistical_tests_summary.csv"

# ---------------------------------------------------------------------------
# Supplier–buyer network analysis
# ---------------------------------------------------------------------------
NETWORK_MIN_TRANSACTIONS = 2   # Minimum trust-supplier transactions required for an edge.
NETWORK_HUB_PERCENTILE = 95    # Betweenness-centrality percentile defining a hub supplier.
NETWORK_NODE_METRICS_PATH = DATA_PROCESSED_DIR / "network_supplier_metrics.csv"
NETWORK_COMMUNITY_PATH = DATA_PROCESSED_DIR / "network_communities.csv"

# ---------------------------------------------------------------------------
# Composite supplier-risk score
# ---------------------------------------------------------------------------
# Sparse supplier histories are excluded because one or two transactions do not
# support a stable supplier-level risk estimate.
SUPPLIER_RISK_MIN_TRANSACTIONS = 5
# Equally weights anomaly incidence, mean anomaly magnitude, and audit-rule
# incidence. Network centrality remains descriptive because its observed
# association with anomaly rates is inverse.
SUPPLIER_RISK_WEIGHTS = {
    "anomaly_rate": 1 / 3,
    "mean_anomaly_score": 1 / 3,
    "rule_flag_rate": 1 / 3,
}
SUPPLIER_RISK_TIER_THRESHOLDS = {"critical": 95, "high": 85, "medium": 60}  # Percentile cut points.
SUPPLIER_RISK_SCORE_PATH = DATA_PROCESSED_DIR / "supplier_risk_score.csv"

# ---------------------------------------------------------------------------
# Category-level analysis
# ---------------------------------------------------------------------------
CATEGORY_DEEP_DIVE_PATH = DATA_PROCESSED_DIR / "category_deep_dive.csv"
CATEGORY_SHOCK_RANKING_PATH = DATA_PROCESSED_DIR / "category_covid_shock_ranking.csv"

# ---------------------------------------------------------------------------
# Sensitivity analyses
# ---------------------------------------------------------------------------
ROBUSTNESS_THRESHOLD_PERCENTILES = [95, 98, 99]
ROBUSTNESS_PERIOD_SHIFT_MONTHS = 1  # Symmetric shift applied to each period boundary.
ROBUSTNESS_THRESHOLD_PATH = DATA_PROCESSED_DIR / "robustness_threshold_sensitivity.csv"
ROBUSTNESS_PERIOD_SHIFT_PATH = DATA_PROCESSED_DIR / "robustness_period_shift.csv"
ROBUSTNESS_FEATURE_ABLATION_PATH = DATA_PROCESSED_DIR / "robustness_feature_ablation.csv"

# ---------------------------------------------------------------------------
# Business-intelligence export
# ---------------------------------------------------------------------------
BI_EXPORT_DIR = DATA_PROCESSED_DIR / "bi_export"
BI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_HTML_PATH = REPORTS_DIR / "nhs_procurement_dashboard.html"
