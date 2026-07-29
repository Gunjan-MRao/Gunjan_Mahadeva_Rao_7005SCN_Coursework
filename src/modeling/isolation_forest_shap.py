"""
Phase 3 — Unsupervised Anomaly Detection (Isolation Forest) + SHAP Explainability.

Methodology (per proposal RQ2/RQ3):
  1. Engineer a feature set capturing amount magnitude, deviation from
     category/supplier norms, transaction recency/frequency and new-supplier
     status.
  2. Train Isolation Forest (Liu, Ting & Zhou, 2008) ONLY on pre-COVID
     trust_spend records, so the model learns "normal" pre-pandemic
     procurement behaviour.
  3. Score every record (pre/COVID/post) with the trained model.
  4. Flag the top `ANOMALY_SCORE_PERCENTILE` percent of (most anomalous)
     records as anomalies.
  5. Use SHAP TreeExplainer (Lundberg & Lee, 2017) to attribute each anomaly
     score to individual features, giving human-interpretable explanations
     of why a transaction was flagged (addresses RQ3 on explainability).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# NOTE: raw source/category identity are deliberately excluded from the feature
# set. An earlier iteration included ordinal-encoded source/category directly
# and found (via SHAP) that ~60% of top-ranked anomalies were driven almost
# entirely by `source_enc` -- i.e. the model was mostly learning "this record
# came from a small trust dataset (Bradford/Lincolnshire), not NHS England"
# rather than detecting genuine spend anomalies, because those two sources are
# a small minority of the pre-COVID training rows. `amount_zscore_category`
# already normalises amount within each source+category group, which captures
# the useful signal without letting sheer sample-size imbalance across sources
# dominate the anomaly ranking.
FEATURE_COLUMNS = [
    "log_amount",
    "amount_zscore_category",
    "days_since_last_txn",
    "supplier_txn_seq",
    "is_new_supplier",
    "month",
    "day_of_week",
]


def load_trust_panel() -> pd.DataFrame:
    panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
    panel = panel[panel["record_type"] == "trust_spend"].copy()
    return panel


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["is_new_supplier"] = df["is_new_supplier"].astype(bool).astype(int)
    df["days_since_last_txn"] = df["days_since_last_txn"].fillna(df["days_since_last_txn"].median())
    df["amount_zscore_category"] = df["amount_zscore_category"].fillna(0.0)
    return df


def train_and_score(panel: pd.DataFrame | None = None):
    if panel is None:
        panel = load_trust_panel()

    df = engineer_features(panel)
    X = df[FEATURE_COLUMNS]

    train_mask = df["period"] == "pre_covid"
    X_train = X[train_mask]
    logger.info("Training Isolation Forest on %d pre-COVID records with %d features", len(X_train), X.shape[1])

    model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    model.fit(X_train)

    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip sign so higher = more anomalous, which is more intuitive to report.
    df["anomaly_score"] = -model.decision_function(X)
    threshold = np.percentile(df["anomaly_score"], config.ANOMALY_SCORE_PERCENTILE)
    df["is_anomaly"] = df["anomaly_score"] >= threshold

    logger.info(
        "Flagged %d / %d records (%.2f%%) as anomalies at the %sth percentile threshold (%.4f)",
        df["is_anomaly"].sum(), len(df), df["is_anomaly"].mean() * 100,
        config.ANOMALY_SCORE_PERCENTILE, threshold,
    )
    logger.info("Anomaly rate by period:\n%s", df.groupby("period")["is_anomaly"].mean().mul(100).round(2).to_string())
    logger.info("Anomaly rate by source:\n%s", df.groupby("source")["is_anomaly"].mean().mul(100).round(2).to_string())

    anomaly_cols = [
        "record_id", "source", "entity", "supplier", "date", "amount", "category",
        "period", "is_new_supplier", "anomaly_score", "is_anomaly",
    ]
    df[anomaly_cols].to_csv(config.ANOMALY_SCORES_PATH, index=False)
    logger.info("Saved anomaly scores -> %s", config.ANOMALY_SCORES_PATH)

    return df, model, X


def explain_with_shap(df: pd.DataFrame, model: IsolationForest, X: pd.DataFrame, top_n: int = 200):
    logger.info("Computing SHAP values with TreeExplainer (this can take a while on large samples)...")

    # SHAP on the full scored set is expensive; explain the flagged anomalies plus
    # a random normal-record sample for contrast, which is standard practice for
    # explaining unsupervised outlier models at manageable computational cost.
    anomaly_idx = df.index[df["is_anomaly"]]
    sample_normal_idx = df.index[~df["is_anomaly"]].to_series().sample(
        n=min(2000, (~df["is_anomaly"]).sum()), random_state=config.RANDOM_STATE
    ).index
    explain_idx = anomaly_idx.union(sample_normal_idx)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.loc[explain_idx])

    shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS, index=explain_idx)
    shap_df["record_id"] = df.loc[explain_idx, "record_id"].values
    shap_df["is_anomaly"] = df.loc[explain_idx, "is_anomaly"].values

    # top anomalies ranked by score, with their dominant SHAP feature attribution
    top_anomalies = df.loc[anomaly_idx].sort_values("anomaly_score", ascending=False).head(top_n).copy()
    feature_shap = shap_df.loc[top_anomalies.index, FEATURE_COLUMNS]
    top_anomalies["top_shap_feature"] = feature_shap.abs().idxmax(axis=1)
    top_anomalies["top_shap_value"] = feature_shap.abs().max(axis=1)

    out_cols = [
        "record_id", "source", "entity", "supplier", "date", "amount", "category",
        "period", "is_new_supplier", "anomaly_score", "top_shap_feature", "top_shap_value",
    ]
    top_anomalies[out_cols].to_csv(config.SHAP_VALUES_PATH, index=False)
    logger.info("Saved top %d SHAP-explained anomalies -> %s", len(top_anomalies), config.SHAP_VALUES_PATH)

    logger.info(
        "Most common SHAP driver feature among top anomalies:\n%s",
        top_anomalies["top_shap_feature"].value_counts().to_string(),
    )
    return shap_df, top_anomalies


def run_modeling_pipeline():
    df, model, X = train_and_score()
    shap_df, top_anomalies = explain_with_shap(df, model, X)
    return df, shap_df, top_anomalies


if __name__ == "__main__":
    run_modeling_pipeline()
