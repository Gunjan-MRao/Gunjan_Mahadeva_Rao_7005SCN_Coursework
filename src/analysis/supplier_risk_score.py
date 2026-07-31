"""
Phase 7A — Composite supplier risk score.

Combines three independent signals into a single, ranked supplier-level risk
index, following the "composite risk index" approach used in the public
procurement corruption-risk literature (Fazekas, Toth & King, 2016, "An
Objective Corruption Risk Index Using Public Procurement Data", European
Journal on Criminal Policy and Research, 22(3); IMF Working Paper 2022/094,
"Assessing Vulnerabilities to Corruption in Public Procurement", which builds
a Composite Risk Indicator as a simple/weighted average of normalised red-flag
indicators). Each raw signal is percentile-ranked (0-100) across suppliers
before blending, so the three signals -- which live on very different natural
scales -- contribute equally regardless of their underlying distribution
(a standard normalisation step in composite-indicator construction; see also
OECD/JRC Handbook on Constructing Composite Indicators, 2008).

Signals blended (equal weights, see config.SUPPLIER_RISK_WEIGHTS):
  1. anomaly_rate       - share of the supplier's own trust_spend transactions
                          flagged anomalous by the Phase 4 Isolation Forest.
  2. mean_anomaly_score - supplier's mean raw (continuous) anomaly score,
                          capturing magnitude of anomalousness, not just the
                          binary flag (a supplier just over the 98th-percentile
                          cut and one far beyond it both score 1 on signal #1).
  3. rule_flag_rate     - share of the supplier's transactions matching >=1
                          literature-derived audit red-flag rule (Phase 5).

Network centrality (hub status, Phase 6D) is deliberately EXCLUDED from the
composite score itself and reported only as a separate descriptive column.
Stage 6D found hub suppliers have a significantly LOWER anomaly rate than
non-hub suppliers (Mann-Whitney p=5.4e-22) -- large, established multi-trust
suppliers are, if anything, under-represented among anomalies, likely because
they have long transaction histories the model has learned as "normal".
Folding centrality into the risk score would therefore bias it in the wrong
direction; it is retained for analysts to see, not scored on.

Suppliers below config.SUPPLIER_RISK_MIN_TRANSACTIONS are excluded from
ranking: a single anomalous transaction out of 1-2 total is too thin a basis
for a supplier-level judgement (small-sample noise), a standard exclusion
criterion in supplier scorecarding.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_supplier_signals() -> pd.DataFrame:
    """Aggregate the three raw per-supplier signals from existing pipeline
    artefacts (anomaly_scores.csv from Phase 4, validation_redflags.csv from
    Phase 5)."""
    scores = pd.read_csv(
        config.ANOMALY_SCORES_PATH, low_memory=False,
        usecols=["supplier", "amount", "anomaly_score", "is_anomaly"],
    )
    supplier_agg = scores.groupby("supplier").agg(
        n_transactions=("amount", "size"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        anomaly_rate=("is_anomaly", "mean"),
        mean_anomaly_score=("anomaly_score", "mean"),
    ).reset_index()

    validation = pd.read_csv(
        config.VALIDATION_PATH, low_memory=False, usecols=["supplier", "rule_flagged"],
    )
    rule_rate = validation.groupby("supplier")["rule_flagged"].mean().rename("rule_flag_rate").reset_index()

    merged = supplier_agg.merge(rule_rate, on="supplier", how="left")
    merged["rule_flag_rate"] = merged["rule_flag_rate"].fillna(0.0)
    return merged


def attach_hub_status(supplier_df: pd.DataFrame) -> pd.DataFrame:
    """Join network hub-status as a descriptive (non-scoring) column."""
    try:
        network = pd.read_csv(
            config.NETWORK_NODE_METRICS_PATH, low_memory=False,
            usecols=["node", "node_type", "is_hub_supplier"],
        )
        network = network[network["node_type"] == "supplier"][["node", "is_hub_supplier"]]
        network = network.rename(columns={"node": "supplier"})
        merged = supplier_df.merge(network, on="supplier", how="left")
        merged["is_hub_supplier"] = merged["is_hub_supplier"].fillna(False).astype(bool)
    except FileNotFoundError:
        logger.warning("Network node metrics not found -- risk score will not include hub-status context column.")
        merged = supplier_df.copy()
        merged["is_hub_supplier"] = np.nan
    return merged


def compute_composite_score(supplier_df: pd.DataFrame) -> pd.DataFrame:
    df = supplier_df[supplier_df["n_transactions"] >= config.SUPPLIER_RISK_MIN_TRANSACTIONS].copy()
    logger.info(
        "Scoring %d / %d suppliers with >= %d transactions",
        len(df), len(supplier_df), config.SUPPLIER_RISK_MIN_TRANSACTIONS,
    )

    weights = config.SUPPLIER_RISK_WEIGHTS
    for signal in weights:
        df[f"{signal}_pctrank"] = df[signal].rank(pct=True) * 100

    df["composite_risk_score"] = sum(df[f"{signal}_pctrank"] * w for signal, w in weights.items())

    thresholds = config.SUPPLIER_RISK_TIER_THRESHOLDS
    crit_cut = np.percentile(df["composite_risk_score"], thresholds["critical"])
    high_cut = np.percentile(df["composite_risk_score"], thresholds["high"])
    med_cut = np.percentile(df["composite_risk_score"], thresholds["medium"])

    def tier(score: float) -> str:
        if score >= crit_cut:
            return "Critical"
        if score >= high_cut:
            return "High"
        if score >= med_cut:
            return "Medium"
        return "Low"

    df["risk_tier"] = df["composite_risk_score"].apply(tier)

    logger.info(
        "Risk tier cut points: Medium>=%.2f, High>=%.2f, Critical>=%.2f (composite score, 0-100 scale)",
        med_cut, high_cut, crit_cut,
    )
    logger.info("Risk tier counts:\n%s", df["risk_tier"].value_counts().to_string())
    return df


def compare_hub_vs_nonhub_risk(scored_df: pd.DataFrame) -> dict:
    """Descriptive check (not used to build the score): do hub suppliers score
    higher or lower composite risk than non-hub suppliers? Reported for
    consistency with -- and as an independent replication of -- the Stage 6D
    finding that hub status is inversely related to raw anomaly rate."""
    valid = scored_df.dropna(subset=["is_hub_supplier"])
    if valid.empty or valid["is_hub_supplier"].nunique() < 2:
        logger.warning("Insufficient hub-status data for hub-vs-non-hub risk-score comparison.")
        return {}

    hub = valid.loc[valid["is_hub_supplier"], "composite_risk_score"]
    nonhub = valid.loc[~valid["is_hub_supplier"], "composite_risk_score"]
    u_stat, p_value = stats.mannwhitneyu(hub, nonhub, alternative="two-sided") if len(hub) and len(nonhub) else (np.nan, np.nan)

    result = {
        "n_hub": len(hub), "n_nonhub": len(nonhub),
        "mean_risk_score_hub": round(hub.mean(), 2) if len(hub) else np.nan,
        "mean_risk_score_nonhub": round(nonhub.mean(), 2) if len(nonhub) else np.nan,
        "mann_whitney_u": u_stat, "p_value": p_value,
    }
    logger.info(
        "Composite risk score, hub vs non-hub: hub=%.2f (n=%d) vs non-hub=%.2f (n=%d) | Mann-Whitney p=%.4g",
        result["mean_risk_score_hub"], result["n_hub"], result["mean_risk_score_nonhub"], result["n_nonhub"], p_value,
    )
    return result


def run_supplier_risk_score():
    signals = load_supplier_signals()
    signals = attach_hub_status(signals)
    scored = compute_composite_score(signals)
    hub_comparison = compare_hub_vs_nonhub_risk(scored)

    out_cols = [
        "supplier", "n_transactions", "total_amount", "mean_amount",
        "anomaly_rate", "mean_anomaly_score", "rule_flag_rate",
        "is_hub_supplier", "composite_risk_score", "risk_tier",
    ]
    scored_out = scored[out_cols].sort_values("composite_risk_score", ascending=False)
    scored_out.to_csv(config.SUPPLIER_RISK_SCORE_PATH, index=False)
    logger.info("Saved composite supplier risk scores -> %s", config.SUPPLIER_RISK_SCORE_PATH)

    logger.info("Top 10 highest-risk suppliers:\n%s", scored_out.head(10).to_string(index=False))

    return scored_out, hub_comparison


if __name__ == "__main__":
    run_supplier_risk_score()
