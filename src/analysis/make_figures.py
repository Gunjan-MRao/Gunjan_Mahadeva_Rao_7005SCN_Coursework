"""
Generates the report figures referenced in the README/methodology writeup.
Run after `python -m src.run_pipeline` has produced the processed CSVs.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def main():
    paths = [
        fig_stl_decomposition(),
        fig_hhi_trend(),
        fig_anomaly_timeline(),
        fig_shap_summary(),
        fig_new_supplier_rate(),
    ]
    for p in paths:
        print("Saved:", p)


if __name__ == "__main__":
    main()
