"""
Generates the report figures referenced in the README/methodology writeup.
Run after `python -m src.run_pipeline` has produced the processed CSVs.
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
    # Per-source HHI is the informative view here: the national NHS_England
    # aggregate spans hundreds of distinct trusts/suppliers and is mechanically
    # always low-HHI (unconcentrated), whereas individual smaller trusts
    # (Bradford, Lincolnshire) show real month-to-month concentration shifts.
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
# Phase 5 (dissertation-advancement) figures
# ---------------------------------------------------------------------------

def fig_method_agreement_heatmap():
    """Jaccard-index agreement heatmap between the four anomaly detectors."""
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
    """Grouped bar chart of precision/recall/F1 per detector against known
    synthetic-injection ground truth."""
    df = pd.read_csv(config.SYNTHETIC_RESULTS_PATH)
    label_map = {"isolation_forest": "Isolation\nForest", "local_outlier_factor": "Local Outlier\nFactor",
                 "one_class_svm": "One-Class\nSVM", "autoencoder": "MLP\nAutoencoder"}
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
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
    """Bar chart comparing mean anomaly rate for hub vs non-hub suppliers,
    and a scatter of transaction volume vs anomaly rate colour-coded by hub status."""
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
    """Bar chart of the top supplier co-occurrence communities by mean
    Isolation Forest anomaly rate."""
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


def main():
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
    ]
    for p in paths:
        print("Saved:", p)


if __name__ == "__main__":
    main()
