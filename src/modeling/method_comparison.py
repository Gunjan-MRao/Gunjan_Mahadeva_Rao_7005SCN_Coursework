"""
Phase 4 extension: comparative unsupervised anomaly detection.

Isolation Forest, Local Outlier Factor, One-Class SVM, and an MLP autoencoder
are trained on the common pre-COVID feature baseline and scored on all records.
Pairwise Jaccard indices and Cohen's kappa quantify binary-flag agreement,
while two-or-more detector consensus summarises cross-detector corroboration.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src import config
from src.modeling.isolation_forest_shap import FEATURE_COLUMNS, engineer_features, load_trust_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

METHOD_NAMES = ["isolation_forest", "local_outlier_factor", "one_class_svm", "autoencoder"]


def _flag_top_percentile(scores: np.ndarray, percentile: float) -> np.ndarray:
    threshold = np.percentile(scores, percentile)
    return scores >= threshold


def fit_isolation_forest(X_train: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    model.fit(X_train)
    return -model.decision_function(X)  # Negation aligns larger scores with greater anomaly severity.


def fit_lof(X_train: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    model = LocalOutlierFactor(**config.LOF_PARAMS)
    model.fit(X_train)
    return -model.decision_function(X)  # Negation aligns larger scores with greater anomaly severity.


def fit_one_class_svm(X_train_scaled: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    model = OneClassSVM(**config.OCSVM_PARAMS)
    model.fit(X_train_scaled)
    return -model.decision_function(X_scaled)  # Negation aligns larger scores with greater anomaly severity.


def fit_autoencoder(X_train_scaled: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    """Fit an MLP autoencoder to reconstruct scaled feature vectors.

    Per-record mean squared reconstruction error is used as the anomaly score.
    """
    model = MLPRegressor(**config.AUTOENCODER_PARAMS)
    model.fit(X_train_scaled, X_train_scaled)
    reconstruction = model.predict(X_scaled)
    mse = np.mean((X_scaled - reconstruction) ** 2, axis=1)
    return mse


def run_method_comparison(panel: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel is None:
        panel = load_trust_panel()
    df = engineer_features(panel)
    X = df[FEATURE_COLUMNS].astype(float)
    train_mask = df["period"] == "pre_covid"
    X_train = X[train_mask]

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X)
    X_train_scaled = scaler.transform(X_train)

    logger.info("Fitting 4 anomaly detectors on %d pre-COVID records, scoring %d total records...", len(X_train), len(X))

    scores = {}
    timings = {}

    t0 = time.time()
    scores["isolation_forest"] = fit_isolation_forest(X_train, X)
    timings["isolation_forest"] = time.time() - t0

    t0 = time.time()
    scores["local_outlier_factor"] = fit_lof(X_train, X)
    timings["local_outlier_factor"] = time.time() - t0

    t0 = time.time()
    n_svm_train = min(config.OCSVM_TRAIN_SAMPLE_SIZE, X_train_scaled.shape[0])
    rng = np.random.RandomState(config.RANDOM_STATE)
    svm_idx = rng.choice(X_train_scaled.shape[0], size=n_svm_train, replace=False)
    logger.info("One-Class SVM: subsampling training set from %d to %d rows for tractability", X_train_scaled.shape[0], n_svm_train)
    scores["one_class_svm"] = fit_one_class_svm(X_train_scaled[svm_idx], X_scaled)
    timings["one_class_svm"] = time.time() - t0

    t0 = time.time()
    scores["autoencoder"] = fit_autoencoder(X_train_scaled, X_scaled)
    timings["autoencoder"] = time.time() - t0

    out = df[["record_id", "source", "entity", "supplier", "date", "amount", "category", "period"]].copy()
    flags = {}
    for method in METHOD_NAMES:
        pct = config.AUTOENCODER_ANOMALY_PERCENTILE if method == "autoencoder" else config.ANOMALY_SCORE_PERCENTILE
        out[f"{method}_score"] = scores[method]
        flag = _flag_top_percentile(scores[method], pct)
        out[f"{method}_flag"] = flag
        flags[method] = flag
        logger.info(
            "%-22s fit+score in %.1fs | flagged %d/%d (%.2f%%)",
            method, timings[method], flag.sum(), len(flag), flag.mean() * 100,
        )

    out["consensus_count"] = out[[f"{m}_flag" for m in METHOD_NAMES]].sum(axis=1)
    out["consensus_flag_majority"] = out["consensus_count"] >= 2  # Consensus threshold: at least two of four detectors flag the record.

    out.to_csv(config.METHOD_COMPARISON_PATH, index=False)
    logger.info("Saved per-record method comparison scores -> %s", config.METHOD_COMPARISON_PATH)

    # Summarise period-specific flag rates to assess detector stability across regimes.
    rate_rows = []
    for method in METHOD_NAMES:
        by_period = out.groupby("period")[f"{method}_flag"].mean().mul(100).round(2)
        for period, rate in by_period.items():
            rate_rows.append({"method": method, "period": period, "flag_rate_pct": rate})
    rate_df = pd.DataFrame(rate_rows)

    # Jaccard captures shared flags; kappa adjusts binary agreement for chance.
    from sklearn.metrics import cohen_kappa_score

    agreement_rows = []
    for i, m1 in enumerate(METHOD_NAMES):
        for m2 in METHOD_NAMES[i + 1:]:
            f1, f2 = flags[m1], flags[m2]
            intersection = np.logical_and(f1, f2).sum()
            union = np.logical_or(f1, f2).sum()
            jaccard = intersection / union if union else np.nan
            kappa = cohen_kappa_score(f1, f2)
            agreement_rows.append({
                "method_a": m1, "method_b": m2,
                "jaccard_index": round(jaccard, 4),
                "cohens_kappa": round(kappa, 4),
                "n_both_flag": int(intersection),
            })
    agreement_df = pd.DataFrame(agreement_rows)

    logger.info("Flag rate by period per method:\n%s", rate_df.pivot(index="period", columns="method", values="flag_rate_pct").to_string())
    logger.info("Pairwise method agreement:\n%s", agreement_df.to_string(index=False))
    logger.info(
        "Consensus (>=2/4 methods agree): %d/%d records (%.2f%%)",
        out["consensus_flag_majority"].sum(), len(out), out["consensus_flag_majority"].mean() * 100,
    )

    summary = {
        "flag_rate_by_period": rate_df,
        "pairwise_agreement": agreement_df,
        "timings_seconds": pd.DataFrame([timings]),
    }
    # Persist tabular summaries for reproducible reporting and downstream analysis.
    rate_df.to_csv(config.METHOD_COMPARISON_SUMMARY_PATH, index=False)
    agreement_df.to_csv(str(config.METHOD_COMPARISON_SUMMARY_PATH).replace(".csv", "_agreement.csv"), index=False)

    return out, summary


if __name__ == "__main__":
    run_method_comparison()
