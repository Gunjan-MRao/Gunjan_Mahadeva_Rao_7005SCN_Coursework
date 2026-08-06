"""Generate dissertation figures from processed analytical artefacts.

Figures visualise STL decomposition, supplier concentration, anomaly detection,
SHAP attribution, network structure, composite risk scores, category shocks,
and robustness checks after pipeline outputs have been generated.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src import config

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

COVID_START = pd.Timestamp(config.COVID_START)
COVID_END = pd.Timestamp(config.COVID_END)


def fig_stl_decomposition():
    """Plot observed expenditure and STL trend, seasonal, and residual components."""
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "stl_decomposition.csv", parse_dates=["date"])
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, col, title in zip(
        axes, ["observed", "trend", "seasonal", "resid"],
        ["Observed monthly spend (£)", "Trend component", "Seasonal component", "Residual"],
    ):
        ax.plot(df["date"], df[col], color="#1f5fae", linewidth=1.3)
        ax.axvspan(COVID_START, COVID_END, color="grey", alpha=0.15)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("STL Decomposition of Monthly NHS Trust Procurement Spend\n(shaded band = COVID-19 period, 23 Mar 2020 – 23 Feb 2022)")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = config.FIGURES_DIR / "stl_decomposition.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_hhi_trend():
    """Plot source-specific monthly HHI to retain trust-level concentration variation."""
    # National aggregation can attenuate concentration variation across heterogeneous trusts.
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "hhi_monthly_by_source.csv")
    df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Bradford_Teaching_Hospitals": "#c1272d", "United_Lincolnshire_Hospitals": "#1f5fae", "NHS_England": "#6a4c93"}
    for source, grp in df.groupby("source"):
        ax.plot(grp["date"], grp["hhi"], label=source.replace("_", " "), linewidth=1.3, color=colors.get(source))
    ax.axvspan(COVID_START, COVID_END, color="grey", alpha=0.15, label="COVID-19 period")
    ax.axhline(1500, color="orange", linestyle="--", linewidth=1, label="Moderate concentration (HHI=1500)")
    ax.axhline(2500, color="red", linestyle="--", linewidth=1, label="High concentration (HHI=2500)")
    ax.set_title("Monthly Supplier Concentration (HHI) by Data Source")
    ax.set_ylabel("HHI")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "hhi_trend.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_anomaly_timeline():
    """Plot the monthly Isolation Forest anomaly rate across the study horizon."""
    df = pd.read_csv(config.ANOMALY_SCORES_PATH, parse_dates=["date"])
    monthly = df.set_index("date").resample("MS").agg(
        n_total=("is_anomaly", "size"), n_anomaly=("is_anomaly", "sum")
    )
    monthly["rate_pct"] = monthly["n_anomaly"] / monthly["n_total"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly["rate_pct"], color="#1f5fae", linewidth=1.4)
    ax.axvspan(COVID_START, COVID_END, color="grey", alpha=0.15, label="COVID-19 period")
    ax.set_title("Monthly Isolation Forest Anomaly Rate\n(trained on pre-COVID data, 98th percentile threshold)")
    ax.set_ylabel("Anomaly rate (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "anomaly_timeline.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_shap_summary():
    """Plot the dominant SHAP feature among the highest-scored anomalies."""
    df = pd.read_csv(config.SHAP_VALUES_PATH)
    counts = df["top_shap_feature"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.sort_values().plot(kind="barh", ax=ax, color="#2a9d8f")
    ax.set_title("Dominant SHAP Feature Among Top-200 Isolation Forest Anomalies")
    ax.set_xlabel("Number of anomalies where this feature was the top driver")
    ax.set_ylabel("")
    fig.tight_layout()
    out = config.FIGURES_DIR / "shap_top_feature_summary.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_new_supplier_rate():
    """Plot monthly new-supplier entry as a share of trust-spend transactions."""
    df = pd.read_csv(config.DATA_PROCESSED_DIR / "new_supplier_rate_by_month.csv")
    df["date"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["new_supplier_rate_pct"], color="#6a4c93", linewidth=1.4)
    ax.axvspan(COVID_START, COVID_END, color="grey", alpha=0.15, label="COVID-19 period")
    ax.set_title("Monthly New-Supplier Rate (% of trust-spend transactions)")
    ax.set_ylabel("New-supplier rate (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = config.FIGURES_DIR / "new_supplier_rate.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Phase 6 figures
# ---------------------------------------------------------------------------

def fig_method_agreement_heatmap():
    """Plot pairwise Jaccard agreement among the anomaly-detection methods."""
    agree = pd.read_csv(str(config.METHOD_COMPARISON_SUMMARY_PATH).replace(".csv", "_agreement.csv"))
    methods = sorted(set(agree["method_a"]) | set(agree["method_b"]))
    label_map = {"isolation_forest": "Isolation\nForest", "local_outlier_factor": "Local Outlier\nFactor",
                 "one_class_svm": "One-Class\nSVM", "autoencoder": "MLP\nAutoencoder"}
    n = len(methods)
    mat = np.eye(n)
    idx = {m: i for i, m in enumerate(methods)}
    for _, row in agree.iterrows():
        i, j = idx[row["method_a"]], idx[row["method_b"]]
        mat[i, j] = row["jaccard_index"]
        mat[j, i] = row["jaccard_index"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels([label_map.get(m, m) for m in methods], fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([label_map.get(m, m) for m in methods], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.5 else "black", fontsize=10)
    ax.set_title("Pairwise Detector Agreement (Jaccard Index)\nAnomaly flags on pre-COVID-trained models")
    fig.colorbar(im, ax=ax, label="Jaccard index", shrink=0.8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "method_agreement_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_synthetic_precision_recall():
    """Plot detector precision, recall, and F1 against synthetic ground truth."""
    df = pd.read_csv(config.SYNTHETIC_RESULTS_PATH)
    label_map = {"isolation_forest": "Isolation\nForest", "local_outlier_factor": "Local Outlier\nFactor",
                 "one_class_svm": "One-Class\nSVM", "autoencoder": "MLP\nAutoencoder",
                 "new_supplier_amount_baseline": "New-Supplier +\nTop-2% Amount\n(baseline rule)"}
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#1f5fae", "#c1272d", "#2a9d8f"]
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[metric], width, label=metric.upper(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(m, m) for m in df["method"]], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Synthetic Anomaly Injection Evaluation\n(400 known injected anomalies in a 20,000-record held-out sample)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = config.FIGURES_DIR / "synthetic_precision_recall.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_network_hub_comparison():
    """Compare hub status, transaction volume, and supplier anomaly rates."""
    nodes = pd.read_csv(config.NETWORK_NODE_METRICS_PATH)
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"], low_memory=False)
    supplier_rate = scores.groupby("supplier")["is_anomaly"].mean()

    suppliers = nodes[nodes["node_type"] == "supplier"].copy()
    suppliers["anomaly_rate_pct"] = suppliers["node"].map(supplier_rate) * 100
    suppliers = suppliers.dropna(subset=["anomaly_rate_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    means = suppliers.groupby("is_hub_supplier")["anomaly_rate_pct"].mean()
    labels = ["Non-hub\nsuppliers", "Hub\nsuppliers"]
    vals = [means.get(False, 0), means.get(True, 0)]
    axes[0].bar(labels, vals, color=["#94a3b8", "#c1272d"])
    axes[0].set_ylabel("Mean Isolation Forest anomaly rate (%)")
    axes[0].set_title("Hub vs Non-Hub Supplier\nAnomaly Rate")
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)

    sample = suppliers.sample(n=min(1500, len(suppliers)), random_state=config.RANDOM_STATE)
    colors = sample["is_hub_supplier"].map({True: "#c1272d", False: "#94a3b8"})
    axes[1].scatter(sample["weighted_degree_n_transactions"], sample["anomaly_rate_pct"],
                     c=colors, alpha=0.5, s=14)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Transaction count (log scale)")
    axes[1].set_ylabel("Anomaly rate (%)")
    axes[1].set_title("Supplier Transaction Volume vs\nAnomaly Rate (red = hub supplier)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / "network_hub_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_community_anomaly_rate():
    """Plot high-anomaly supplier co-occurrence communities of adequate size."""
    communities = pd.read_csv(config.NETWORK_COMMUNITY_PATH)
    scores = pd.read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"], low_memory=False)
    supplier_rate = scores.groupby("supplier")["is_anomaly"].mean()

    communities["anomaly_rate_pct"] = communities["node"].map(supplier_rate) * 100
    communities = communities.dropna(subset=["anomaly_rate_pct"])
    summary = communities.groupby("community_id").agg(
        size=("node", "size"), mean_rate=("anomaly_rate_pct", "mean")
    ).reset_index()
    summary = summary[summary["size"] >= 3].sort_values("mean_rate", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh([f"Community {int(c)}\n(n={int(s)})" for c, s in zip(summary["community_id"], summary["size"])],
                   summary["mean_rate"], color="#1f5fae")
    overall_rate = scores["is_anomaly"].mean() * 100
    ax.axvline(overall_rate, color="red", linestyle="--", linewidth=1, label=f"Overall rate ({overall_rate:.2f}%)")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Isolation Forest anomaly rate (%)")
    ax.set_title("Top 10 Supplier Co-occurrence Communities\nby Mean Anomaly Rate (communities of size >= 3)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = config.FIGURES_DIR / "community_anomaly_rate.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Phase 7 figures
# ---------------------------------------------------------------------------

def fig_supplier_risk_score():
    """Plot risk-tier prevalence and the highest-ranked composite-risk suppliers."""
    df = pd.read_csv(config.SUPPLIER_RISK_SCORE_PATH)
    tier_order = ["Low", "Medium", "High", "Critical"]
    tier_colors = {"Low": "#94a3b8", "Medium": "#f4a259", "High": "#e07a5f", "Critical": "#c1272d"}
    counts = df["risk_tier"].value_counts().reindex(tier_order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].bar(tier_order, counts.values, color=[tier_colors[t] for t in tier_order])
    axes[0].set_ylabel("Number of suppliers")
    axes[0].set_title(f"Composite Supplier Risk Score\nTier Distribution (n={len(df):,} scored suppliers)")
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts.values) * 0.01, f"{int(v):,}", ha="center", fontsize=9)

    top15 = df.sort_values("composite_risk_score", ascending=False).head(15).iloc[::-1]
    bar_colors = top15["is_hub_supplier"].map({True: "#c1272d", False: "#1f5fae"})
    labels = [s[:28] + ("..." if len(s) > 28 else "") for s in top15["supplier"]]
    axes[1].barh(labels, top15["composite_risk_score"], color=bar_colors)
    axes[1].set_xlabel("Composite risk score (0-100)")
    axes[1].set_title("Top 15 Highest-Risk Suppliers\n(red = network hub supplier, blue = non-hub)")
    axes[1].grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = config.FIGURES_DIR / "supplier_risk_score.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_category_covid_shock():
    """Plot category-level pre-COVID-to-COVID expenditure growth rankings."""
    df = pd.read_csv(config.CATEGORY_SHOCK_RANKING_PATH)
    df = df.dropna(subset=["pct_change_pre_to_covid"])
    top10 = df.sort_values("pct_change_pre_to_covid", ascending=False).head(10).iloc[::-1]
    labels = [c[:30] + ("..." if len(c) > 30 else "") for c in top10["category"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, top10["pct_change_pre_to_covid"], color="#2a9d8f")
    ax.set_xlabel("% change in total spend, pre-COVID -> COVID")
    ax.set_title("Top 10 Spend Categories Driving the COVID-19 Shock\n(categories with >= 20 transactions)")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(top10["pct_change_pre_to_covid"]):
        ax.text(v, i, f" {v:,.0f}%", va="center", fontsize=8)
    fig.tight_layout()
    out = config.FIGURES_DIR / "category_covid_shock.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_robustness_checks():
    """Plot threshold and period-boundary sensitivity of anomaly-rate estimates."""
    thresh = pd.read_csv(config.ROBUSTNESS_THRESHOLD_PATH)
    shift = pd.read_csv(config.ROBUSTNESS_PERIOD_SHIFT_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    periods = ["rate_pre_covid_pct", "rate_covid_pct", "rate_post_covid_pct"]
    period_labels = ["Pre-COVID", "COVID", "Post-COVID"]
    x = np.arange(len(period_labels))
    width = 0.25
    colors = ["#94a3b8", "#1f5fae", "#c1272d"]
    for i, (_, row) in enumerate(thresh.iterrows()):
        axes[0].bar(x + (i - 1) * width, [row[p] for p in periods], width,
                    label=f"{int(row['threshold_percentile'])}th pct", color=colors[i])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(period_labels)
    axes[0].set_ylabel("Anomaly rate (%)")
    axes[0].set_title("Threshold Sensitivity:\nAnomaly Rate by Period")
    axes[0].legend(fontsize=8, title="Score cutoff")
    axes[0].grid(alpha=0.3, axis="y")

    for i, (_, row) in enumerate(shift.iterrows()):
        axes[1].plot(period_labels, [row[p] for p in periods], marker="o",
                     label=row["boundary_variant"].replace("_", " "), linewidth=1.6)
    axes[1].set_ylabel("Anomaly rate (%)")
    axes[1].set_title("Period-Boundary Sensitivity:\nAnomaly Rate under Shifted COVID Boundaries")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / "robustness_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def run_make_figures():
    """Generate the complete reproducible dissertation figure set."""
    paths = [
        fig_stl_decomposition(),
        fig_hhi_trend(),
        fig_anomaly_timeline(),
        fig_shap_summary(),
        fig_new_supplier_rate(),
        fig_method_agreement_heatmap(),
        fig_synthetic_precision_recall(),
        fig_network_hub_comparison(),
        fig_community_anomaly_rate(),
        fig_supplier_risk_score(),
        fig_category_covid_shock(),
        fig_robustness_checks(),
    ]
    for p in paths:
        print("Saved:", p)
    return paths


# Compatibility alias for direct script execution.
main = run_make_figures


if __name__ == "__main__":
    main()
