"""
Phase 5 — Validation against literature-derived audit red-flag criteria.

Confidential NAO / NHS Counter Fraud Authority (NHSCFA) case-level audit data
is not publicly accessible, so this project uses a defensible proxy: published
red-flag criteria from Transparency International UK's 2024 review of
pandemic-era procurement, and NHSCFA's published fraud-risk indicators, are
encoded as explicit rules. The overlap between rule-flagged and
Isolation-Forest-flagged records is then reported as a triangulation /
construct-validity check on the unsupervised model (per proposal Stage 5).

Rules implemented:
  R1  Direct-award / single-source contract notice issued during COVID.
  R2  Trust-spend transaction where amount > 3x the supplier's own historical
      median amount ("price spike vs historical median").
  R3  New supplier (per is_new_supplier flag) whose first transaction during
      COVID exceeds the 90th percentile of same-period spend
      ("new supplier + large COVID contract").
  R4  Suspiciously round invoice amounts (exact multiples of 10,000, excluding
      very small amounts where rounding is common for legitimate reasons).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
    panel["is_new_supplier"] = panel["is_new_supplier"].astype(bool)
    return panel


def flag_direct_award_covid(panel: pd.DataFrame) -> pd.Series:
    is_contract = panel["record_type"] == "contract_notice"
    is_covid = panel["period"] == "covid"
    is_direct = panel["category"].astype(str).str.lower().isin(["limited", "direct", "negotiated without prior publication"])
    return is_contract & is_covid & is_direct


def flag_price_spike(panel: pd.DataFrame) -> pd.Series:
    trust = panel[panel["record_type"] == "trust_spend"]
    median_by_supplier = trust.groupby(["entity_norm", "supplier_norm"])["amount"].transform("median")
    spike = pd.Series(False, index=panel.index)
    trust_spike = trust["amount"] > (3 * median_by_supplier.clip(lower=1))
    spike.loc[trust.index] = trust_spike
    return spike


def flag_new_supplier_large_covid(panel: pd.DataFrame) -> pd.Series:
    trust = panel[panel["record_type"] == "trust_spend"]
    covid_trust = trust[trust["period"] == "covid"]
    if covid_trust.empty:
        return pd.Series(False, index=panel.index)
    p90 = covid_trust["amount"].quantile(0.90)
    flag = pd.Series(False, index=panel.index)
    mask = (panel["period"] == "covid") & (panel["record_type"] == "trust_spend") & panel["is_new_supplier"] & (panel["amount"] > p90)
    flag.loc[mask.index] = mask
    return flag


def flag_round_amount(panel: pd.DataFrame) -> pd.Series:
    trust = panel[panel["record_type"] == "trust_spend"]
    flag = pd.Series(False, index=panel.index)
    is_round = (trust["amount"] >= 10_000) & (trust["amount"] % 10_000 == 0)
    flag.loc[trust.index] = is_round
    return flag


def run_validation():
    panel = load_panel()
    panel["flag_direct_award_covid"] = flag_direct_award_covid(panel)
    panel["flag_price_spike"] = flag_price_spike(panel)
    panel["flag_new_supplier_large_covid"] = flag_new_supplier_large_covid(panel)
    panel["flag_round_amount"] = flag_round_amount(panel)

    rule_cols = ["flag_direct_award_covid", "flag_price_spike", "flag_new_supplier_large_covid", "flag_round_amount"]
    panel["rule_flag_count"] = panel[rule_cols].sum(axis=1)
    panel["rule_flagged"] = panel["rule_flag_count"] > 0

    for col in rule_cols:
        logger.info("%s: %d flagged (%.2f%%)", col, panel[col].sum(), panel[col].mean() * 100)
    logger.info("Any rule flagged: %d (%.2f%%)", panel["rule_flagged"].sum(), panel["rule_flagged"].mean() * 100)

    # triangulation against ML (Isolation Forest) anomalies, where available
    try:
        ml_scores = pd.read_csv(config.ANOMALY_SCORES_PATH, low_memory=False)[["record_id", "is_anomaly", "anomaly_score"]]
        merged = panel.merge(ml_scores, on="record_id", how="left")
        merged["is_anomaly"] = merged["is_anomaly"].fillna(False)

        both = merged[merged["is_anomaly"] & merged["rule_flagged"]]
        ml_only = merged[merged["is_anomaly"] & ~merged["rule_flagged"]]
        rule_only = merged[~merged["is_anomaly"] & merged["rule_flagged"]]

        n_ml = merged["is_anomaly"].sum()
        n_rule = merged["rule_flagged"].sum()
        precision_vs_rules = len(both) / n_ml * 100 if n_ml else np.nan

        logger.info(
            "Triangulation: ML-flagged=%d, rule-flagged=%d, overlap=%d, "
            "ML-only=%d, rule-only=%d, overlap-rate (%% of ML flags with >=1 rule)=%.2f%%",
            n_ml, n_rule, len(both), len(ml_only), len(rule_only), precision_vs_rules,
        )
        merged.to_csv(config.VALIDATION_PATH, index=False)
        logger.info("Saved validation/triangulation table -> %s", config.VALIDATION_PATH)
        return merged
    except FileNotFoundError:
        logger.warning("Anomaly scores not found — run isolation_forest_shap.py first for full triangulation. Saving rule-only output.")
        panel.to_csv(config.VALIDATION_PATH, index=False)
        return panel


if __name__ == "__main__":
    run_validation()
