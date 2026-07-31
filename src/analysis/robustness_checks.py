"""
Phase 7C — Robustness checks.

Every headline result up to Stage 6 rests on three modelling choices that are
each, to some degree, arbitrary: the 98th-percentile anomaly-score cut-off,
the exact calendar boundaries used to split pre/COVID/post periods, and the
feature set fed into Isolation Forest. Reporting sensitivity of the headline
findings to reasonable variations in each of these is standard practice for
unsupervised anomaly-detection studies without ground-truth labels (see e.g.
Aggarwal, C.C. (2017). "Outlier Analysis", 2nd ed., Springer, ch. 12, on the
importance of threshold/parameter sensitivity analysis for unsupervised
outlier ensembles; Emmott et al. (2015), "A Meta-Analysis of the Anomaly
Detection Problem", arXiv:1503.01158, on benchmarking detector robustness).

Three checks, each re-using artefacts already produced earlier in the
pipeline rather than duplicating expensive model training where avoidable:

  (a) Threshold sensitivity - re-flag anomalies at 95th/98th/99th percentile
      cut-offs on the *same* already-computed anomaly_score column (Phase 4),
      then recompute the period-level anomaly rate and the ML/rule
      triangulation hypergeometric test (Phase 6C) at each threshold. Checks
      whether the period ordering and ML/rule overlap significance survive a
      stricter or looser cut-off.
  (b) Period-boundary sensitivity - recompute period labels with the COVID
      start/end boundaries shifted +/- 1 calendar month (applied
      symmetrically) and recompute the anomaly rate by period under each
      shifted definition, using the same per-record anomaly scores. Checks
      whether the period-level ordering is an artefact of the exact
      boundary dates chosen.
  (c) Feature-set ablation - retrains Isolation Forest with `is_new_supplier`
      removed from the feature set (the single most policy-salient/least
      "generic" feature) and compares the resulting anomaly flags against the
      baseline (full feature set) via Jaccard index and Spearman rank
      correlation of the two continuous anomaly-score vectors. Checks whether
      headline anomalies are highly sensitive to a single engineered feature.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from src import config
from src.modeling.isolation_forest_shap import FEATURE_COLUMNS, engineer_features, load_trust_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _jaccard(a: pd.Series, b: pd.Series) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else np.nan


# ---------------------------------------------------------------------------
# (a) Threshold sensitivity
# ---------------------------------------------------------------------------
def threshold_sensitivity() -> pd.DataFrame:
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, low_memory=False, usecols=["record_id", "period", "anomaly_score"])
    validation = pd.read_csv(config.VALIDATION_PATH, low_memory=False, usecols=["record_id", "rule_flagged"])
    merged = scores.merge(validation, on="record_id", how="left")
    merged["rule_flagged"] = merged["rule_flagged"].fillna(False)

    flag_cols = {}
    rows = []
    for pct in config.ROBUSTNESS_THRESHOLD_PERCENTILES:
        threshold = np.percentile(merged["anomaly_score"], pct)
        col = f"is_anomaly_p{pct}"
        merged[col] = merged["anomaly_score"] >= threshold
        flag_cols[pct] = col

        rate_by_period = merged.groupby("period")[col].mean().mul(100).round(3)
        overlap = int((merged[col] & merged["rule_flagged"]).sum())
        n_ml = int(merged[col].sum())
        n_rule = int(merged["rule_flagged"].sum())
        N = len(merged)
        log_p = stats.hypergeom.logsf(max(overlap - 1, -1), N, n_rule, n_ml)

        rows.append({
            "threshold_percentile": pct, "score_cutoff": round(threshold, 4),
            "n_flagged": n_ml, "flagged_rate_pct": round(n_ml / N * 100, 3),
            "rate_pre_covid_pct": rate_by_period.get("pre_covid", np.nan),
            "rate_covid_pct": rate_by_period.get("covid", np.nan),
            "rate_post_covid_pct": rate_by_period.get("post_covid", np.nan),
            "rule_overlap_n": overlap,
            "rule_overlap_rate_pct": round(overlap / n_ml * 100, 3) if n_ml else np.nan,
            "hypergeometric_log_p": log_p,
        })

    result = pd.DataFrame(rows)
    orderings = result[["rate_pre_covid_pct", "rate_covid_pct", "rate_post_covid_pct"]].rank(axis=1)
    order_stable = bool((orderings == orderings.iloc[0]).all(axis=None))
    logger.info(
        "Threshold sensitivity (95/98/99th pct): period-rate ordering %s across thresholds; "
        "ML/rule overlap remains hypergeometrically significant (log p) at every threshold: %s",
        "STABLE" if order_stable else "NOT stable", (result["hypergeometric_log_p"] < -10).all(),
    )
    logger.info("Threshold sensitivity table:\n%s", result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# (b) Period-boundary sensitivity
# ---------------------------------------------------------------------------
def _assign_period(dates: pd.Series, covid_start: pd.Timestamp, covid_end: pd.Timestamp) -> pd.Series:
    period = pd.Series("pre_covid", index=dates.index)
    period[(dates >= covid_start) & (dates <= covid_end)] = "covid"
    period[dates > covid_end] = "post_covid"
    period[dates < pd.Timestamp(config.PRE_COVID_START)] = "pre_covid"
    return period


def period_shift_sensitivity() -> pd.DataFrame:
    panel = pd.read_csv(config.MASTER_PANEL_PATH, low_memory=False, usecols=["record_id", "date"])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce", format="mixed")
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, low_memory=False, usecols=["record_id", "is_anomaly"])
    merged = panel.merge(scores, on="record_id", how="inner").dropna(subset=["date"])

    shift = pd.DateOffset(months=config.ROBUSTNESS_PERIOD_SHIFT_MONTHS)
    baseline_start, baseline_end = pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END)
    variants = {
        "baseline": (baseline_start, baseline_end),
        f"shifted_earlier_{config.ROBUSTNESS_PERIOD_SHIFT_MONTHS}m": (baseline_start - shift, baseline_end - shift),
        f"shifted_later_{config.ROBUSTNESS_PERIOD_SHIFT_MONTHS}m": (baseline_start + shift, baseline_end + shift),
    }

    rows = []
    for variant, (c_start, c_end) in variants.items():
        merged["period_variant"] = _assign_period(merged["date"], c_start, c_end)
        rate_by_period = merged.groupby("period_variant")["is_anomaly"].mean().mul(100).round(3)
        rows.append({
            "boundary_variant": variant,
            "covid_start": c_start.date().isoformat(), "covid_end": c_end.date().isoformat(),
            "rate_pre_covid_pct": rate_by_period.get("pre_covid", np.nan),
            "rate_covid_pct": rate_by_period.get("covid", np.nan),
            "rate_post_covid_pct": rate_by_period.get("post_covid", np.nan),
        })

    result = pd.DataFrame(rows)
    orderings = result[["rate_pre_covid_pct", "rate_covid_pct", "rate_post_covid_pct"]].rank(axis=1)
    order_stable = (orderings == orderings.iloc[0]).all(axis=None)
    logger.info(
        "Period-boundary sensitivity (+/- %d month shift): period-rate ordering %s across boundary variants",
        config.ROBUSTNESS_PERIOD_SHIFT_MONTHS, "STABLE" if order_stable else "NOT stable",
    )
    logger.info("Period-shift sensitivity table:\n%s", result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# (c) Feature-set ablation
# ---------------------------------------------------------------------------
def feature_ablation_sensitivity() -> pd.DataFrame:
    panel = load_trust_panel()
    df = engineer_features(panel)

    baseline_scores = pd.read_csv(config.ANOMALY_SCORES_PATH, low_memory=False, usecols=["record_id", "anomaly_score", "is_anomaly"])
    baseline_scores = baseline_scores.rename(columns={"anomaly_score": "baseline_score", "is_anomaly": "baseline_is_anomaly"})

    ablated_features = [f for f in FEATURE_COLUMNS if f != "is_new_supplier"]
    train_mask = df["period"] == "pre_covid"

    model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    X_ablated = df[ablated_features]
    model.fit(X_ablated[train_mask])
    df["ablated_score"] = -model.decision_function(X_ablated)
    ablated_threshold = np.percentile(df["ablated_score"], config.ANOMALY_SCORE_PERCENTILE)
    df["ablated_is_anomaly"] = df["ablated_score"] >= ablated_threshold

    merged = df[["record_id", "ablated_score", "ablated_is_anomaly"]].merge(baseline_scores, on="record_id", how="inner")

    jaccard = _jaccard(merged["baseline_is_anomaly"], merged["ablated_is_anomaly"])
    spearman_rho, spearman_p = stats.spearmanr(merged["baseline_score"], merged["ablated_score"])

    result = pd.DataFrame([{
        "ablation": "remove_is_new_supplier",
        "baseline_features": len(FEATURE_COLUMNS), "ablated_features": len(ablated_features),
        "n_baseline_flagged": int(merged["baseline_is_anomaly"].sum()),
        "n_ablated_flagged": int(merged["ablated_is_anomaly"].sum()),
        "jaccard_overlap": round(jaccard, 4),
        "spearman_rho_scores": round(spearman_rho, 4),
        "spearman_p": spearman_p,
    }])
    logger.info(
        "Feature ablation (drop is_new_supplier): Jaccard overlap of flagged records=%.3f, "
        "Spearman rho of continuous anomaly scores=%.3f (p=%.4g)",
        jaccard, spearman_rho, spearman_p,
    )
    return result


def run_robustness_checks():
    thresh = threshold_sensitivity()
    period_shift = period_shift_sensitivity()
    ablation = feature_ablation_sensitivity()

    thresh.to_csv(config.ROBUSTNESS_THRESHOLD_PATH, index=False)
    period_shift.to_csv(config.ROBUSTNESS_PERIOD_SHIFT_PATH, index=False)
    ablation.to_csv(config.ROBUSTNESS_FEATURE_ABLATION_PATH, index=False)
    logger.info(
        "Saved robustness check outputs -> %s, %s, %s",
        config.ROBUSTNESS_THRESHOLD_PATH, config.ROBUSTNESS_PERIOD_SHIFT_PATH, config.ROBUSTNESS_FEATURE_ABLATION_PATH,
    )
    return thresh, period_shift, ablation


if __name__ == "__main__":
    run_robustness_checks()
