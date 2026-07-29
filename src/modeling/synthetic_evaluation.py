"""
Phase 3 extension — Synthetic anomaly injection evaluation.

Confidential NAO / NHSCFA case-level audit labels are not publicly available
(see `src/validation/audit_validation.py`), so this module provides a
complementary, fully quantitative benchmark: known synthetic anomalies with
literature-motivated fraud/error signatures are injected into a held-out
evaluation sample, all four detectors from `method_comparison.py` are
(re)trained on the same pre-COVID data and scored on the injected sample, and
genuine precision / recall / F1 / average-precision (PR-AUC) are computed
against the known injection labels.

This is standard practice for evaluating unsupervised anomaly detectors when
no real ground truth exists (e.g. Emmott et al., 2013's synthetic-anomaly
benchmarking methodology; see also the "simulation study" approach in
https://scholarworks.utrgv.edu/mss_fac/560/).

Three literature-motivated synthetic anomaly types are injected:
  1. invoice_inflation      — an existing supplier's invoice amount is
                               multiplied 5-15x (mimics an inflated/duplicate
                               invoice).
  2. ghost_vendor_burst     — a "new" supplier (never seen before) receives an
                               unusually large first payment with no prior
                               transaction history (classic ghost-vendor /
                               fictitious-supplier red flag).
  3. round_number_kickback  — the amount is forced to a suspiciously round
                               figure (kickback/skimming red flag per
                               NHSCFA guidance).

IMPORTANT CAVEAT (reported in the dissertation, not hidden): precision here
is a lower bound, because some records in the "normal" background may
themselves be genuine anomalies the detectors correctly (but not credited
for) flagging. Recall against the KNOWN injected anomalies is the more
reliable of the two headline metrics for this reason.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.modeling.isolation_forest_shap import FEATURE_COLUMNS, engineer_features, load_trust_panel
from src.modeling.method_comparison import (
    METHOD_NAMES,
    fit_autoencoder,
    fit_isolation_forest,
    fit_lof,
    fit_one_class_svm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

INJECTION_TYPES = ["invoice_inflation", "ghost_vendor_burst", "round_number_kickback"]


def _inject_invoice_inflation(row: pd.Series, rng: np.random.RandomState) -> pd.Series:
    factor = rng.uniform(*config.SYNTHETIC_AMOUNT_INFLATION_RANGE)
    row = row.copy()
    row["amount"] = row["amount"] * factor
    row["injection_type"] = "invoice_inflation"
    return row


def _inject_ghost_vendor_burst(row: pd.Series, rng: np.random.RandomState, category_p90: float) -> pd.Series:
    row = row.copy()
    row["is_new_supplier"] = 1
    row["supplier_txn_seq"] = 1
    row["days_since_last_txn"] = 0.0  # no prior history for this "supplier"
    row["amount"] = max(row["amount"], category_p90) * rng.uniform(1.5, 3.0)
    row["injection_type"] = "ghost_vendor_burst"
    return row


def _inject_round_number_kickback(row: pd.Series, rng: np.random.RandomState) -> pd.Series:
    row = row.copy()
    magnitude_options = [10_000, 25_000, 50_000, 100_000, 250_000]
    weights = np.array([1 / m for m in magnitude_options])
    weights = weights / weights.sum()
    base_unit = rng.choice(magnitude_options, p=weights)
    n_units = rng.randint(2, 20)
    row["amount"] = float(base_unit * n_units)
    row["injection_type"] = "round_number_kickback"
    return row


def build_injected_evaluation_set(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel = load_trust_panel()
    df = engineer_features(panel)

    # Evaluate on covid + post-covid records only (pre-covid is the training
    # window for every detector, so injecting there would contaminate training).
    eval_pool = df[df["period"] != "pre_covid"].copy()
    n_eval = min(config.SYNTHETIC_EVAL_SAMPLE_SIZE, len(eval_pool))
    eval_sample = eval_pool.sample(n=n_eval, random_state=config.RANDOM_STATE).reset_index(drop=True)
    eval_sample["synthetic_anomaly"] = False
    eval_sample["injection_type"] = "none"

    rng = np.random.RandomState(config.RANDOM_STATE)
    n_inject = int(round(n_eval * config.SYNTHETIC_INJECTION_RATE))
    inject_idx = rng.choice(eval_sample.index, size=n_inject, replace=False)

    # category-level amount stats (for ghost-vendor sizing + z-score recompute)
    cat_p90 = df.groupby(["source", "category"])["amount"].quantile(0.90)
    cat_log_mean = df.groupby(["source", "category"])["log_amount"].mean()
    cat_log_std = df.groupby(["source", "category"])["log_amount"].std(ddof=0).fillna(1e-9) + 1e-9

    injectors = [
        _inject_invoice_inflation,
        _inject_ghost_vendor_burst,
        _inject_round_number_kickback,
    ]
    type_assignment = rng.choice(len(injectors), size=n_inject)

    for pos, idx in enumerate(inject_idx):
        row = eval_sample.loc[idx]
        key = (row["source"], row["category"])
        if type_assignment[pos] == 0:
            new_row = _inject_invoice_inflation(row, rng)
        elif type_assignment[pos] == 1:
            p90 = cat_p90.get(key, row["amount"])
            new_row = _inject_ghost_vendor_burst(row, rng, p90)
        else:
            new_row = _inject_round_number_kickback(row, rng)

        new_row["log_amount"] = np.log1p(new_row["amount"])
        mean_, std_ = cat_log_mean.get(key, 0.0), cat_log_std.get(key, 1.0)
        new_row["amount_zscore_category"] = (new_row["log_amount"] - mean_) / std_
        new_row["synthetic_anomaly"] = True
        eval_sample.loc[idx] = new_row

    logger.info(
        "Built evaluation sample: %d records, %d synthetic anomalies injected (%.2f%%) -- breakdown:\n%s",
        n_eval, n_inject, n_inject / n_eval * 100,
        eval_sample.loc[inject_idx, "injection_type"].value_counts().to_string(),
    )
    return eval_sample


def run_synthetic_evaluation(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    if panel is None:
        panel = load_trust_panel()
    df = engineer_features(panel)
    X_train = df.loc[df["period"] == "pre_covid", FEATURE_COLUMNS].astype(float)

    eval_sample = build_injected_evaluation_set(panel)
    X_eval = eval_sample[FEATURE_COLUMNS].astype(float)
    y_true = eval_sample["synthetic_anomaly"].astype(int).values

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    scores = {
        "isolation_forest": fit_isolation_forest(X_train, X_eval),
        "local_outlier_factor": fit_lof(X_train, X_eval),
        "one_class_svm": fit_one_class_svm(
            X_train_scaled[
                np.random.RandomState(config.RANDOM_STATE).choice(
                    X_train_scaled.shape[0], size=min(config.OCSVM_TRAIN_SAMPLE_SIZE, X_train_scaled.shape[0]), replace=False
                )
            ],
            X_eval_scaled,
        ),
        "autoencoder": fit_autoencoder(X_train_scaled, X_eval_scaled),
    }

    rows = []
    per_type_rows = []
    for method in METHOD_NAMES:
        s = scores[method]
        pct = config.AUTOENCODER_ANOMALY_PERCENTILE if method == "autoencoder" else config.ANOMALY_SCORE_PERCENTILE
        threshold = np.percentile(s, pct)
        y_pred = (s >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        ap = average_precision_score(y_true, s)

        rows.append({
            "method": method, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "average_precision_pr_auc": round(ap, 4),
            "n_flagged": int(y_pred.sum()), "n_true_anomalies": int(y_true.sum()),
        })

        eval_sample[f"{method}_pred"] = y_pred
        for inj_type in INJECTION_TYPES:
            mask = eval_sample["injection_type"] == inj_type
            if mask.sum() == 0:
                continue
            recall_type = y_pred[mask.values].mean()
            per_type_rows.append({"method": method, "injection_type": inj_type, "recall": round(recall_type, 4), "n": int(mask.sum())})

    results_df = pd.DataFrame(rows)
    per_type_df = pd.DataFrame(per_type_rows)

    logger.info("Synthetic-injection evaluation (headline metrics):\n%s", results_df.to_string(index=False))
    logger.info("Recall by injection type per method:\n%s", per_type_df.pivot(index="injection_type", columns="method", values="recall").to_string())

    results_df.to_csv(config.SYNTHETIC_RESULTS_PATH, index=False)
    per_type_df.to_csv(str(config.SYNTHETIC_RESULTS_PATH).replace(".csv", "_by_type.csv"), index=False)
    logger.info("Saved synthetic-injection evaluation results -> %s", config.SYNTHETIC_RESULTS_PATH)

    return results_df, per_type_df


if __name__ == "__main__":
    run_synthetic_evaluation()
