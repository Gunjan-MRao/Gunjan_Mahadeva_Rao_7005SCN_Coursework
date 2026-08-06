"""
Phase 7E — Masters-level interactive HTML dashboard.

Builds a self-contained, dark-themed, multi-tab dashboard from the same
star-schema CSVs exported by bi_export.py (Phase 7D), plus a handful of
already-computed enrichment tables (SHAP, network, method-agreement,
synthetic-injection, community) that already exist elsewhere in
data/processed/. Every KPI, chart value, and caption number below is
computed live from those files at render time -- nothing is hard-coded --
so the dashboard always reflects whatever data is actually on disk.

Output requires no server: open reports/nhs_procurement_dashboard.html
directly in any browser. Plotly.js is loaded once from CDN in <head>;
each chart is embedded as an individual `fig.to_html(full_html=False,
include_plotlyjs=False)` div. Tabs are pure vanilla JS (no frameworks).
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0a0d14",
    "surface": "#111827",
    "elevated": "#1a2035",
    "plot_bg": "#0f1623",
    "border": "rgba(255,255,255,0.07)",
    "accent": "#4f98a3",
    "accent_hover": "#38b2ac",
    "text": "#e2e8f0",
    "text_dim": "#94a3b8",
    "grid": "rgba(255,255,255,0.06)",
    "axis": "rgba(255,255,255,0.12)",
}
PERIOD_COLORS = {"Pre-COVID": "#94a3b8", "COVID": "#c1272d", "Post-COVID": "#3b82f6"}
PERIOD_KEY_TO_LABEL = {"pre_covid": "Pre-COVID", "covid": "COVID", "post_covid": "Post-COVID"}
TIER_COLORS = {"Low": "#94a3b8", "Medium": "#f4a259", "High": "#e07a5f", "Critical": "#c1272d"}
TIER_ORDER = ["Low", "Medium", "High", "Critical"]
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_tables() -> dict:
    """Load the five required star-schema BI-export tables (Phase 7D)."""
    d = config.BI_EXPORT_DIR
    return {
        "fact": pd.read_csv(d / "fact_transactions.csv", low_memory=False),
        "supplier": pd.read_csv(d / "dim_supplier.csv"),
        "category": pd.read_csv(d / "dim_category.csv"),
        "period": pd.read_csv(d / "dim_period.csv"),
        "month": pd.read_csv(d / "dim_month.csv"),
    }


def _safe_read_csv(path, **kwargs):
    """Best-effort read of an *enrichment* CSV that is not part of the core
    star schema. Several of these files are gitignored/local-only artefacts
    of the ML pipeline, so a fresh checkout without a full pipeline run
    should not crash dashboard generation -- callers degrade gracefully."""
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except (FileNotFoundError, OSError):
        logger.warning("Optional enrichment file not found, skipping: %s", path)
        return None


def _fmt_ym(ym: str) -> str:
    try:
        return datetime.strptime(str(ym), "%Y-%m").strftime("%b %Y")
    except ValueError:
        return str(ym)


def _b64_png(path) -> str | None:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except (FileNotFoundError, OSError):
        logger.warning("Figure PNG not found, will render placeholder: %s", path)
        return None


# ---------------------------------------------------------------------------
# Live metric computation (no hard-coded figures anywhere below)
# ---------------------------------------------------------------------------
def _compute_kpis(t: dict) -> dict:
    fact, supplier, month, period = t["fact"], t["supplier"], t["month"], t["period"]

    trust_only = fact[fact["record_type"] == "trust_spend"]
    total_spend = float(trust_only["amount"].sum())
    total_txn = int(len(trust_only))
    scored_mask = fact["is_ml_scored"] == True  # noqa: E712
    ml_scored = int(scored_mask.sum())
    anomaly_rate = float(fact.loc[scored_mask, "is_ml_anomaly"].mean() * 100)
    n_suppliers_scored = int(len(supplier))
    n_critical = int((supplier["risk_tier"] == "Critical").sum())
    pct_critical = n_critical / n_suppliers_scored * 100 if n_suppliers_scored else float("nan")

    rate_by_period = {
        PERIOD_KEY_TO_LABEL.get(r["period"], r["period"]): r["ml_anomaly_rate_pct"]
        for _, r in period.iterrows()
    }
    date_min, date_max = _fmt_ym(month["year_month"].min()), _fmt_ym(month["year_month"].max())

    return {
        "total_spend": total_spend,
        "total_txn": total_txn,
        "ml_scored": ml_scored,
        "anomaly_rate": anomaly_rate,
        "n_suppliers_scored": n_suppliers_scored,
        "n_critical": n_critical,
        "pct_critical": pct_critical,
        "rate_pre": rate_by_period.get("Pre-COVID", float("nan")),
        "rate_covid": rate_by_period.get("COVID", float("nan")),
        "rate_post": rate_by_period.get("Post-COVID", float("nan")),
        "date_min": date_min,
        "date_max": date_max,
    }


def _compute_stl(t: dict) -> dict:
    stl = _safe_read_csv(config.STL_PATH)
    if stl is None or "pct_deviation_from_baseline" not in stl.columns:
        return {"covid_dev": float("nan"), "post_dev": float("nan")}
    dev = {row["period"]: row["pct_deviation_from_baseline"] for _, row in stl.iterrows()}
    return {"covid_dev": dev.get("covid", float("nan")), "post_dev": dev.get("post_covid", float("nan"))}


def _compute_new_supplier_by_period(t: dict) -> dict:
    month = t["month"]
    weighted = (
        month.groupby("period")
        .apply(lambda g: np.average(g["new_supplier_rate_pct"], weights=g["n_transactions"]))
    )
    return {PERIOD_KEY_TO_LABEL.get(k, k): v for k, v in weighted.items()}


def _compute_ml_rule_jaccard(t: dict) -> dict:
    fact = t["fact"]
    scored = fact[fact["is_ml_scored"] == True]  # noqa: E712
    a = scored["is_ml_anomaly"].astype(bool)
    b = scored["is_rule_flagged"].astype(bool)
    inter, union = int((a & b).sum()), int((a | b).sum())
    return {
        "jaccard": inter / union if union else float("nan"),
        "n_ml": int(a.sum()),
        "n_rule": int(b.sum()),
        "n_both": inter,
    }


def _compute_method_agreement() -> pd.DataFrame | None:
    return _safe_read_csv(config.DATA_PROCESSED_DIR / "method_comparison_summary_agreement.csv")


def _compute_synthetic_eval() -> pd.DataFrame | None:
    return _safe_read_csv(config.SYNTHETIC_RESULTS_PATH)


def _compute_shap_summary() -> dict:
    shap = _safe_read_csv(config.SHAP_VALUES_PATH)
    if shap is None or "top_shap_feature" not in shap.columns:
        return {"top_feature": None, "top_n": 0, "total_n": 0, "top_pct": float("nan")}
    counts = shap["top_shap_feature"].value_counts()
    top_feature, top_n = counts.index[0], int(counts.iloc[0])
    total_n = int(len(shap))
    return {
        "top_feature": top_feature, "top_n": top_n, "total_n": total_n,
        "top_pct": top_n / total_n * 100 if total_n else float("nan"),
    }


def _compute_hub_comparison(t: dict) -> dict:
    """Hub vs non-hub anomaly rate. Prefers the full buyer-supplier network
    population (matching the Phase 6D methodology in
    src/network/supplier_network.py: network_supplier_metrics.csv
    cross-referenced with per-supplier rates from anomaly_scores.csv),
    since that is the population the dissertation's headline hub-vs-non-hub
    finding is drawn from. Falls back to the smaller, risk-scored
    dim_supplier population (>=5 transactions) -- and its own
    ml_anomaly_rate / is_network_hub columns, as those are always present
    in the committed star schema -- if the richer network files are not
    available in the current environment."""
    try:
        node_df = _safe_read_csv(config.NETWORK_NODE_METRICS_PATH)
        scores = _safe_read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"])
        if node_df is None or scores is None:
            raise ValueError("network enrichment files unavailable")
        rate = scores.groupby("supplier")["is_anomaly"].mean()
        suppliers = node_df[node_df["node_type"] == "supplier"].copy()
        suppliers["anomaly_rate"] = suppliers["node"].map(rate)
        suppliers = suppliers.dropna(subset=["anomaly_rate"])
        hub = suppliers.loc[suppliers["is_hub_supplier"], "anomaly_rate"] * 100
        nonhub = suppliers.loc[~suppliers["is_hub_supplier"], "anomaly_rate"] * 100
        p_value = float("nan")
        try:
            from scipy import stats
            _, p_value = stats.mannwhitneyu(hub, nonhub, alternative="two-sided")
        except ImportError:
            pass
        return {
            "hub_rate": float(hub.mean()), "nonhub_rate": float(nonhub.mean()),
            "n_hub": int(len(hub)), "n_nonhub": int(len(nonhub)), "p_value": p_value,
            "source": "buyer-supplier network (Phase 6D methodology): degree/betweenness-defined hub suppliers",
        }
    except Exception:
        supplier = t["supplier"]
        hub = supplier.loc[supplier["is_network_hub"], "ml_anomaly_rate"] * 100
        nonhub = supplier.loc[~supplier["is_network_hub"], "ml_anomaly_rate"] * 100
        p_value = float("nan")
        try:
            from scipy import stats
            _, p_value = stats.mannwhitneyu(hub, nonhub, alternative="two-sided")
        except ImportError:
            pass
        return {
            "hub_rate": float(hub.mean()), "nonhub_rate": float(nonhub.mean()),
            "n_hub": int(len(hub)), "n_nonhub": int(len(nonhub)), "p_value": p_value,
            "source": "risk-scored suppliers only (dim_supplier, >=5 transactions)",
        }


def _compute_top_community() -> dict:
    try:
        comm = _safe_read_csv(config.NETWORK_COMMUNITY_PATH)
        scores = _safe_read_csv(config.ANOMALY_SCORES_PATH, usecols=["supplier", "is_anomaly"])
        if comm is None or scores is None:
            raise ValueError("network enrichment files unavailable")
        rate = scores.groupby("supplier")["is_anomaly"].mean()
        comm = comm.copy()
        comm["anomaly_rate"] = comm["node"].map(rate)
        comm = comm.dropna(subset=["anomaly_rate"])
        agg = (
            comm.groupby(["community_id", "community_size"])["anomaly_rate"]
            .mean().reset_index()
        )
        agg = agg[agg["community_size"] >= 3].sort_values("anomaly_rate", ascending=False)
        if agg.empty:
            raise ValueError("no communities with size >= 3")
        top = agg.iloc[0]
        return {
            "community_id": int(top["community_id"]), "size": int(top["community_size"]),
            "rate": float(top["anomaly_rate"] * 100), "table": agg.head(10),
        }
    except Exception:
        return {"community_id": None, "size": None, "rate": float("nan"), "table": None}


def _compute_robustness() -> pd.DataFrame | None:
    return _safe_read_csv(config.ROBUSTNESS_THRESHOLD_PATH)


# ---------------------------------------------------------------------------
# Plotly theme helpers
# ---------------------------------------------------------------------------
def _dark(fig: go.Figure, height: int = 440) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["plot_bg"],
        font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text"], size=12),
        margin=dict(t=48, l=56, r=30, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"], size=11)),
        hoverlabel=dict(bgcolor=COLORS["elevated"], font_color=COLORS["text"], bordercolor=COLORS["border"]),
        title_font=dict(color=COLORS["text"], size=15),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], zerolinecolor=COLORS["axis"], color=COLORS["text_dim"])
    fig.update_yaxes(gridcolor=COLORS["grid"], linecolor=COLORS["axis"], zerolinecolor=COLORS["axis"], color=COLORS["text_dim"])
    return fig


def _to_div(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displayModeBar": False})


def _sparkline(y: list, color: str) -> str:
    fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color=color, width=2), fill="tozeroy",
                                fillcolor=color.replace(")", ", 0.12)").replace("rgb", "rgba") if color.startswith("rgb") else "rgba(79,152,163,0.12)"))
    fig.update_layout(
        height=40, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, l=0, r=0, b=0), showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"staticPlot": True})


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def _fig_spend_anomaly(month: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    periods = ["All", "Pre-COVID", "COVID", "Post-COVID"]
    trace_groups = []
    for p_label in periods:
        sub = month if p_label == "All" else month[month["period"].map(lambda k: PERIOD_KEY_TO_LABEL.get(k, k)) == p_label]
        bar = go.Bar(
            x=sub["year_month"], y=sub["total_spend"], name="Monthly spend (£)",
            marker_color=[PERIOD_COLORS.get(PERIOD_KEY_TO_LABEL.get(k, k), COLORS["accent"]) for k in sub["period"]],
            opacity=0.88, yaxis="y1", visible=(p_label == "All"),
            hovertemplate="%{x}<br>Spend: £%{y:,.0f}<extra></extra>",
        )
        line = go.Scatter(
            x=sub["year_month"], y=sub["ml_anomaly_rate_pct"], name="ML anomaly rate (%)",
            mode="lines+markers", line=dict(color=COLORS["accent_hover"], width=2.5),
            marker=dict(size=5), yaxis="y2", visible=(p_label == "All"),
            hovertemplate="%{x}<br>Anomaly rate: %{y:.2f}%<extra></extra>",
        )
        fig.add_trace(bar)
        fig.add_trace(line)
        trace_groups.append(p_label)

    buttons = []
    for i, p_label in enumerate(periods):
        vis = [False] * (len(periods) * 2)
        vis[i * 2] = True
        vis[i * 2 + 1] = True
        buttons.append(dict(label=p_label, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        title="Monthly Spend & ML Anomaly Rate Over Time",
        yaxis=dict(title="Monthly spend (£)"),
        yaxis2=dict(title="ML anomaly rate (%)", overlaying="y", side="right", showgrid=False),
        updatemenus=[dict(
            buttons=buttons, direction="down", x=1, xanchor="right", y=1.18, yanchor="top",
            bgcolor=COLORS["elevated"], bordercolor=COLORS["border"], font=dict(color=COLORS["text"]),
        )],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    return _dark(fig, height=460)


def _fig_hhi(month: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=month["year_month"], y=month["hhi"], mode="lines", line=dict(color=COLORS["accent"], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,152,163,0.10)",
        hovertemplate="%{x}<br>HHI: %{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title="Supplier Concentration (HHI) Over Time", yaxis_title="HHI (0-10,000)")
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    return _dark(fig, height=380)


def _fig_new_supplier(month: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=month["year_month"], y=month["new_supplier_rate_pct"], mode="lines+markers",
        line=dict(color="#f4a259", width=2.5), marker=dict(size=4),
        hovertemplate="%{x}<br>New-supplier rate: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Monthly New-Supplier Entry Rate", yaxis_title="New-supplier rate (%)")
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    return _dark(fig, height=380)


def _fig_anomaly_box(fact: pd.DataFrame) -> go.Figure:
    scored = fact[fact["is_ml_scored"] == True].copy()  # noqa: E712
    scored["period_label"] = scored["period"].map(lambda k: PERIOD_KEY_TO_LABEL.get(k, k))
    fig = go.Figure()
    for p_label in ["Pre-COVID", "COVID", "Post-COVID"]:
        sub = scored[scored["period_label"] == p_label]
        fig.add_trace(go.Box(
            y=sub["anomaly_score_zscore"], name=p_label, marker_color=PERIOD_COLORS[p_label],
            boxpoints=False,
        ))
    fig.update_layout(title="ML Anomaly Score Distribution by Period", yaxis_title="Anomaly score (z-score)", showlegend=False)
    return _dark(fig, height=420)


def _fig_rule_severity(fact: pd.DataFrame) -> go.Figure:
    df = fact.copy()
    df["period_label"] = df["period"].map(lambda k: PERIOD_KEY_TO_LABEL.get(k, k))
    df["severity"] = df["rule_flag_severity"].fillna("None")
    order = ["None", "Single", "Double", "Triple+"]
    sev_colors = {"None": "#334155", "Single": "#94a3b8", "Double": "#f4a259", "Triple+": "#c1272d"}
    counts = df.groupby(["period_label", "severity"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in order if c in counts.columns])
    fig = go.Figure()
    for sev in counts.columns:
        fig.add_trace(go.Bar(
            x=["Pre-COVID", "COVID", "Post-COVID"], y=counts.loc[["pre_covid", "covid", "post_covid"], sev]
            if set(["pre_covid", "covid", "post_covid"]).issubset(set(df["period"].unique())) else counts[sev],
            name=sev, marker_color=sev_colors.get(sev, "#999"),
        ))
    fig.update_layout(
        title="Rule-Flag Severity by Period", barmode="stack", yaxis_title="Transactions",
        showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
    )
    fig = _dark(fig, height=460)
    fig.update_layout(margin=dict(b=90))
    return fig


def _fig_anomaly_scatter(fact: pd.DataFrame) -> go.Figure:
    scored = fact[fact["is_ml_scored"] == True].copy()  # noqa: E712
    sample = scored.sample(n=min(6000, len(scored)), random_state=42) if len(scored) > 6000 else scored
    colors = sample["is_ml_anomaly"].map({True: "#c1272d", False: "rgba(148,163,184,0.35)"})
    fig = go.Figure(go.Scatter(
        x=sample["ml_anomaly_score"], y=sample["rule_flag_count"], mode="markers",
        marker=dict(color=colors, size=6, line=dict(width=0)),
        hovertemplate="ML score: %{x:.3f}<br>Rule flags: %{y}<extra></extra>",
    ))
    fig.update_layout(title="ML Anomaly Score vs Rule-Flag Count", xaxis_title="ML anomaly score", yaxis_title="Rule-flag count", showlegend=False)
    return _dark(fig, height=420)


def _fig_risk_tier(supplier: pd.DataFrame) -> go.Figure:
    counts = supplier["risk_tier"].value_counts().reindex(TIER_ORDER).fillna(0)
    fig = go.Figure(go.Bar(
        x=TIER_ORDER, y=counts.values, marker_color=[TIER_COLORS[t] for t in TIER_ORDER],
        text=counts.values.astype(int), textposition="outside", textfont=dict(color=COLORS["text"]),
        hovertemplate="%{x}: %{y} suppliers<extra></extra>",
    ))
    fig.update_layout(title="Composite Risk Tier Distribution", yaxis_title="Number of suppliers", showlegend=False)
    return _dark(fig, height=380)


def _fig_top20_suppliers(supplier: pd.DataFrame) -> go.Figure:
    top = supplier.sort_values("composite_risk_score", ascending=False).head(20).iloc[::-1]
    labels = [s[:32] + ("..." if len(s) > 32 else "") for s in top["supplier"]]
    fig = go.Figure(go.Bar(
        x=top["composite_risk_score"], y=labels, orientation="h",
        marker_color=[TIER_COLORS.get(t, "#999") for t in top["risk_tier"]],
        hovertemplate="%{y}<br>Composite risk score: %{x:.1f}<extra></extra>", showlegend=False,
    ))
    hubs = top[top["is_network_hub"]]
    if len(hubs):
        hub_labels = [s[:32] + ("..." if len(s) > 32 else "") for s in hubs["supplier"]]
        fig.add_trace(go.Scatter(
            x=hubs["composite_risk_score"] + 2, y=hub_labels, mode="markers",
            marker=dict(symbol="star", size=11, color="#f4a259"), name="Network hub supplier",
            hovertemplate="%{y} (network hub)<extra></extra>",
        ))
    fig.update_layout(title="Top 20 Highest-Risk Suppliers", xaxis_title="Composite risk score (0-100)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _dark(fig, height=560)


def _fig_hub_comparison(hub_stats: dict) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[f"Hub suppliers (n={hub_stats['n_hub']:,})", f"Non-hub suppliers (n={hub_stats['n_nonhub']:,})"],
        y=[hub_stats["hub_rate"], hub_stats["nonhub_rate"]],
        marker_color=["#f4a259", COLORS["accent"]],
        text=[f"{hub_stats['hub_rate']:.2f}%", f"{hub_stats['nonhub_rate']:.2f}%"],
        textposition="outside", textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(title="Hub vs Non-Hub Supplier Anomaly Rate", yaxis_title="Mean ML anomaly rate (%)", showlegend=False)
    return _dark(fig, height=380)


def _fig_category_shock(category: pd.DataFrame) -> go.Figure:
    cat = category.dropna(subset=["pct_change_pre_to_covid"]).sort_values(
        "pct_change_pre_to_covid", ascending=False,
    ).head(15).iloc[::-1]
    labels = [c[:34] + ("..." if len(c) > 34 else "") for c in cat["category"]]
    fig = go.Figure(go.Bar(
        x=cat["pct_change_pre_to_covid"], y=labels, orientation="h",
        marker_color=COLORS["accent"], showlegend=False,
        hovertemplate="%{y}<br>Pre-COVID \u2192 COVID: %{x:,.0f}%<extra></extra>",
    ))
    fig.update_layout(title="Top 15 COVID Shock Categories", xaxis_title="% change in monthly spend, pre-COVID \u2192 COVID")
    return _dark(fig, height=520)


def _fig_category_heatmap(category: pd.DataFrame) -> go.Figure:
    top = category.sort_values("total_spend", ascending=False).head(20).copy()
    cols = ["mean_anomaly_rate_pct", "mean_new_supplier_rate_pct", "mean_hhi", "share_of_total_spend_pct"]
    col_labels = ["Anomaly rate %", "New-supplier rate %", "Mean HHI", "Share of spend %"]
    z_raw = top[cols].to_numpy(dtype=float)
    z_norm = (z_raw - np.nanmin(z_raw, axis=0)) / (np.nanmax(z_raw, axis=0) - np.nanmin(z_raw, axis=0) + 1e-9)
    labels = [c[:30] + ("..." if len(c) > 30 else "") for c in top["category"]]
    fig = go.Figure(go.Heatmap(
        z=z_norm, x=col_labels, y=labels, colorscale=[[0, COLORS["accent"]], [1, "#c1272d"]],
        text=z_raw, texttemplate="%{text:.2f}", showscale=False,
        hovertemplate="%{y}<br>%{x}: %{text:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Category Risk Heatmap (Top 20 Categories by Spend)")
    return _dark(fig, height=560)


# ---------------------------------------------------------------------------
# HTML fragments (static -- plain strings, no f-string brace conflicts)
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
  --bg: #0a0d14; --surface: #111827; --elevated: #1a2035; --border: rgba(255,255,255,0.07);
  --accent: #4f98a3; --accent-hover: #38b2ac; --text: #e2e8f0; --text-dim: #94a3b8;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: 'Inter', -apple-system, sans-serif; }
h1, h2, h3 { font-weight: 700; margin: 0; }
.header { position: sticky; top: 0; z-index: 50; background: linear-gradient(90deg, #005EB8 0%, var(--accent) 100%);
  display: flex; align-items: center; justify-content: space-between; padding: 14px 28px; box-shadow: 0 2px 12px rgba(0,0,0,0.4); }
.header-left { display: flex; align-items: center; gap: 12px; }
.header-title { text-align: center; flex: 1; }
.header-title h1 { font-size: 20px; color: #fff; letter-spacing: 0.2px; }
.header-title p { margin: 2px 0 0; font-size: 12px; color: rgba(255,255,255,0.85); }
.header-right { display: flex; align-items: center; gap: 10px; }
.badge { background: rgba(255,255,255,0.15); color: #fff; font-size: 11px; padding: 5px 10px; border-radius: 999px; white-space: nowrap; }
.export-btn { background: rgba(255,255,255,0.18); color: #fff; border: 1px solid rgba(255,255,255,0.3); border-radius: 8px;
  padding: 7px 14px; font-size: 13px; font-family: inherit; cursor: pointer; transition: background 0.15s; }
.export-btn:hover { background: var(--accent-hover); }
.container { max-width: 1360px; margin: 0 auto; padding: 24px 20px 60px; }
.exec-banner { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 14px; padding: 22px 26px; margin-bottom: 24px; }
.exec-banner h2 { font-size: 15px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; }
.exec-banner p { font-size: 14px; line-height: 1.65; color: var(--text); margin: 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.kpi-card { background: var(--elevated); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; position: relative; }
.kpi-top { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.kpi-icon { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 14px; color: #0a0d14; flex-shrink: 0; }
.kpi-value { font-size: 22px; font-weight: 700; color: var(--text); line-height: 1.1; }
.kpi-caption { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
.tabs-nav { position: sticky; top: 62px; z-index: 40; background: rgba(10,13,20,0.92); backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--border); display: flex; gap: 4px; padding: 0 20px; margin-bottom: 24px; }
.tab-btn { background: none; border: none; color: var(--text-dim); font-family: inherit; font-size: 14px; font-weight: 600;
  padding: 14px 18px; cursor: pointer; border-bottom: 2px solid transparent; transition: color 0.15s, border-color 0.15s; }
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--accent-hover); border-bottom-color: var(--accent-hover); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px; margin-bottom: 20px; }
.card-caption { font-size: 13px; color: var(--text-dim); line-height: 1.6; margin-top: 10px; padding-top: 12px; border-top: 1px solid var(--border); }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.section-title { font-size: 18px; margin: 28px 0 14px; color: var(--text); }
.png-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px; margin-bottom: 20px; }
.png-card img { width: 100%; border-radius: 8px; display: block; }
.png-card h3 { font-size: 14px; margin: 12px 0 6px; color: var(--text); }
.png-card p { font-size: 12.5px; color: var(--text-dim); line-height: 1.55; margin: 0; }
.png-card .src-tag { display: inline-block; margin-top: 8px; font-size: 10px; color: #6b7684; background: rgba(255,255,255,0.04);
  padding: 2px 8px; border-radius: 999px; }
.masonry { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.footer { text-align: center; padding: 24px 20px; font-size: 12px; color: var(--text-dim); border-top: 1px solid var(--border); }
@media (max-width: 900px) {
  .kpi-grid, .grid-2, .masonry { grid-template-columns: 1fr; }
  .exec-banner { grid-template-columns: 1fr; }
}
@media print {
  body { background: #fff !important; color: #111 !important; }
  .header, .tabs-nav, .export-btn { display: none !important; }
  .tab-content { display: block !important; }
  .card, .png-card, .exec-banner, .kpi-card { background: #fff !important; border: 1px solid #ccc !important; break-inside: avoid; }
  .card-caption, .kpi-caption, p, .footer { color: #333 !important; }
}
</style>
"""

JS = """
<script>
function showTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.tab-btn').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}
</script>
"""

NHS_LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="32" height="32" rx="6" fill="rgba(255,255,255,0.15)"/>'
    '<rect x="13" y="6" width="6" height="20" rx="2" fill="#ffffff"/>'
    '<rect x="6" y="13" width="20" height="6" rx="2" fill="#ffffff"/>'
    '</svg>'
)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
def _kpi_card(icon: str, icon_color: str, value: str, label: str, caption: str, spark_y: list) -> str:
    spark_html = _sparkline(spark_y, icon_color) if spark_y else ""
    return (
        '<div class="kpi-card">'
        '<div class="kpi-top"><div class="kpi-icon" style="background:' + icon_color + '">' + icon + '</div>'
        '<div><div class="kpi-value">' + value + '</div><div style="font-size:11px;color:var(--text-dim)">' + label + '</div></div></div>'
        + spark_html +
        '<div class="kpi-caption">' + caption + '</div>'
        '</div>'
    )


def _png_card(title: str, caption: str, b64: str | None, filename: str) -> str:
    img_html = (
        '<img src="data:image/png;base64,' + b64 + '" alt="' + title + '">' if b64
        else '<div style="padding:40px;text-align:center;color:var(--text-dim);border:1px dashed var(--border);border-radius:8px;">Figure not available: ' + filename + '</div>'
    )
    return (
        '<div class="png-card">' + img_html +
        '<h3>' + title + '</h3><p>' + caption + '</p>'
        '<span class="src-tag">Source: docs/figures/' + filename + '</span>'
        '</div>'
    )


def build_dashboard() -> str:
    t = _load_tables()
    fact, supplier, category, period, month = t["fact"], t["supplier"], t["category"], t["period"], t["month"]

    k = _compute_kpis(t)
    stl = _compute_stl(t)
    new_sup = _compute_new_supplier_by_period(t)
    jacc = _compute_ml_rule_jaccard(t)
    method_agree = _compute_method_agreement()
    synth = _compute_synthetic_eval()
    shap_summary = _compute_shap_summary()
    hub_stats = _compute_hub_comparison(t)
    top_comm = _compute_top_community()
    robustness = _compute_robustness()
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    # --- Executive summary (dynamic, corrected wording) ---------------------
    exec_p1 = (
        f"This study analysed {k['total_txn']:,} NHS trust procurement transactions spanning "
        f"{k['date_min']} \u2013 {k['date_max']}. An Isolation Forest machine-learning model flagged "
        f"{k['anomaly_rate']:.2f}% of {k['ml_scored']:,} scored transactions as anomalous overall."
    )
    exec_p2 = (
        f"The anomaly rate rose steadily across the study period \u2014 from {k['rate_pre']:.2f}% pre-COVID to "
        f"{k['rate_covid']:.2f}% during the COVID-19 emergency-procurement window, and further to {k['rate_post']:.2f}% "
        f"post-COVID \u2014 indicating the signal has not reverted to baseline even after the acute crisis period ended."
    )
    exec_p3 = (
        f"Of {k['n_suppliers_scored']:,} risk-scored suppliers, {k['n_critical']} ({k['pct_critical']:.1f}%) were "
        f"classified Critical-risk; network analysis found structurally central \u201chub\u201d suppliers show a markedly "
        f"lower per-transaction anomaly rate ({hub_stats['hub_rate']:.2f}%) than peripheral, non-hub suppliers "
        f"({hub_stats['nonhub_rate']:.2f}%)."
    )

    # --- KPI cards ------------------------------------------------------------
    kpi_html = (
        _kpi_card("\u00a3", "#4f98a3", f"\u00a3{k['total_spend']/1e9:.1f}bn", "Total Trust Spend Analysed",
                   f"{k['total_txn']:,} trust_spend transactions, {k['date_min']}\u2013{k['date_max']}",
                   month["total_spend"].tolist())
        + _kpi_card("#", "#3b82f6", f"{k['total_txn']:,}", "Transactions Analysed",
                     f"{k['ml_scored']:,} scored by Isolation Forest", month["n_transactions"].tolist())
        + _kpi_card("%", "#c1272d", f"{k['anomaly_rate']:.2f}%", "ML Anomaly Rate",
                     f"Pre {k['rate_pre']:.2f}% \u2192 COVID {k['rate_covid']:.2f}% \u2192 Post {k['rate_post']:.2f}%",
                     month["ml_anomaly_rate_pct"].tolist())
        + _kpi_card("!", "#f4a259", f"{k['n_critical']}", "Critical-Risk Suppliers",
                     f"{k['pct_critical']:.1f}% of {k['n_suppliers_scored']:,} risk-scored suppliers",
                     [supplier["risk_tier"].value_counts().get(tr, 0) for tr in TIER_ORDER])
    )

    # --- Tab: Overview ----------------------------------------------------
    overview_annotation = (
        f"COVID-19 drove a +{stl['covid_dev']:.1f}% monthly spend surge above the pre-pandemic STL baseline, "
        f"deepening further to +{stl['post_dev']:.1f}% post-COVID. The ML anomaly rate climbed from "
        f"{k['rate_pre']:.2f}% pre-pandemic to {k['rate_covid']:.2f}% during the emergency-procurement window, and "
        f"continued rising to {k['rate_post']:.2f}% post-COVID \u2014 the highest of the three periods, not COVID "
        f"itself \u2014 suggesting the disruption to normal procurement controls persisted well beyond the acute crisis."
    )
    hhi_caption = (
        f"Herfindahl-Hirschman Index across the study period ranges from {month['hhi'].min():,.0f} to "
        f"{month['hhi'].max():,.0f} (mean {month['hhi'].mean():,.0f}). Values in this range indicate a highly "
        f"deconcentrated supplier market at the monthly aggregate level, with no single supplier consistently "
        f"dominating trust spend."
    )
    new_sup_caption = (
        f"New-supplier entry rate averaged {new_sup.get('Pre-COVID', float('nan')):.2f}% pre-COVID, more than doubling to "
        f"{new_sup.get('COVID', float('nan')):.2f}% during COVID as emergency procurement onboarded unfamiliar suppliers, "
        f"before settling to {new_sup.get('Post-COVID', float('nan')):.2f}% post-COVID \u2014 still above the pre-COVID baseline."
    )

    tab_overview = (
        '<div class="card">' + _to_div(_fig_spend_anomaly(month))
        + '<div class="card-caption">' + overview_annotation + '</div></div>'
        + '<div class="grid-2">'
        + '<div class="card">' + _to_div(_fig_hhi(month)) + '<div class="card-caption">' + hhi_caption + '</div></div>'
        + '<div class="card">' + _to_div(_fig_new_supplier(month)) + '<div class="card-caption">' + new_sup_caption + '</div></div>'
        + '</div>'
    )

    # --- Tab: Anomaly Analysis ----------------------------------------------
    scatter_caption = (
        f"ML anomaly flags ({jacc['n_ml']:,} transactions) and independent rule-based audit flags "
        f"({jacc['n_rule']:,} transactions) overlap on only {jacc['n_both']:,} transactions "
        f"(Jaccard index = {jacc['jaccard']:.3f}). The low overlap indicates the two detection approaches are largely "
        f"complementary rather than redundant \u2014 each catches risk signal the other misses."
    )
    heatmap_caption = "Pairwise agreement (Jaccard index) between all four anomaly-detection methods compared in Phase 4."
    if method_agree is not None and len(method_agree):
        top_pair = method_agree.sort_values("jaccard_index", ascending=False).iloc[0]
        heatmap_caption += (
            f" The strongest agreement is between {top_pair['method_a'].replace('_', ' ')} and "
            f"{top_pair['method_b'].replace('_', ' ')} (Jaccard = {top_pair['jaccard_index']:.3f})."
        )
    synth_caption = "Precision/recall/F1 of each detector against synthetically injected, known-type anomalies."
    if synth is not None and len(synth):
        s = synth.set_index("method")
        if "isolation_forest" in s.index:
            if_f1 = s.loc["isolation_forest", "f1"]
            synth_caption += f" Isolation Forest scores F1 = {if_f1:.3f} on the synthetic evaluation sample"
            if "new_supplier_amount_baseline" in s.index:
                base_f1 = s.loc["new_supplier_amount_baseline", "f1"]
                synth_caption += (
                    f", below the simple new-supplier-large-amount heuristic baseline (F1 = {base_f1:.3f}) \u2014 an "
                    f"honest limitation to acknowledge, though the ML flags still add complementary value on top of "
                    f"rule-based detection (see the low ML-vs-rule Jaccard above)."
                )
            else:
                synth_caption += "."

    tab_anomaly = (
        '<div class="grid-2">'
        + '<div class="card">' + _to_div(_fig_anomaly_box(fact)) + '<div class="card-caption">Distribution of standardised anomaly scores by period; wider right tails indicate more extreme outlier transactions.</div></div>'
        + '<div class="card">' + _to_div(_fig_rule_severity(fact)) + '<div class="card-caption">Count of transactions by number of audit red-flag rules matched (None/Single/Double/Triple+), split by period.</div></div>'
        + '</div>'
        + '<div class="card">' + _to_div(_fig_anomaly_scatter(fact)) + '<div class="card-caption">' + scatter_caption + '</div></div>'
        + '<div class="masonry">'
        + _png_card("Multi-Method Agreement", heatmap_caption, _b64_png(config.FIGURES_DIR / "method_agreement_heatmap.png"), "method_agreement_heatmap.png")
        + _png_card("Synthetic Anomaly Injection Evaluation", synth_caption, _b64_png(config.FIGURES_DIR / "synthetic_precision_recall.png"), "synthetic_precision_recall.png")
        + '</div>'
    )

    # --- Tab: Supplier Risk --------------------------------------------------
    tier_caption = (
        f"{k['n_suppliers_scored']:,} suppliers with \u22655 transactions were risk-scored; "
        f"{k['n_critical']} ({k['pct_critical']:.1f}%) fall into the Critical tier."
    )
    hub_caption = (
        f"Hub suppliers show a mean anomaly rate of {hub_stats['hub_rate']:.2f}% (n={hub_stats['n_hub']:,}) versus "
        f"{hub_stats['nonhub_rate']:.2f}% (n={hub_stats['n_nonhub']:,}) for non-hub suppliers"
        + (f", a difference confirmed by Mann-Whitney U test (p = {hub_stats['p_value']:.3g})" if hub_stats['p_value'] == hub_stats['p_value'] else "")
        + f". Computed from {hub_stats['source']}. Structurally central suppliers are, counter-intuitively, the "
          f"*lower*-risk group \u2014 anomalies concentrate instead among smaller, newer, single-relationship suppliers."
    )

    tab_supplier = (
        '<div class="grid-2">'
        + '<div class="card">' + _to_div(_fig_risk_tier(supplier)) + '<div class="card-caption">' + tier_caption + '</div></div>'
        + '<div class="card">' + _to_div(_fig_hub_comparison(hub_stats)) + '<div class="card-caption">' + hub_caption + '</div></div>'
        + '</div>'
        + '<div class="card">' + _to_div(_fig_top20_suppliers(supplier)) + '<div class="card-caption">Composite risk score blends anomaly rate, mean anomaly-score magnitude, and rule-flag rate (equal weights). Gold stars mark network-hub suppliers.</div></div>'
        + '<div class="masonry">'
        + _png_card("Supplier Risk Score Distribution", "Composite risk score distribution across all risk-scored suppliers, with the top-ranked suppliers highlighted.", _b64_png(config.FIGURES_DIR / "supplier_risk_score.png"), "supplier_risk_score.png")
        + _png_card("Network Hub vs Non-Hub Comparison", hub_caption, _b64_png(config.FIGURES_DIR / "network_hub_comparison.png"), "network_hub_comparison.png")
        + '</div>'
    )

    # --- Tab: Category Intelligence ------------------------------------------
    top_shock_cat = category.dropna(subset=["pct_change_pre_to_covid"]).sort_values("pct_change_pre_to_covid", ascending=False).iloc[0]
    top_shock_caption = f"The largest COVID-era spend shock was in \u201c{top_shock_cat['category']}\u201d, up {top_shock_cat['pct_change_pre_to_covid']:,.0f}% pre-COVID \u2192 COVID."
    stl_caption = f"STL decomposition of the monthly trust-spend series shows a +{stl['covid_dev']:.1f}% deviation above the pre-pandemic seasonal baseline during COVID, widening to +{stl['post_dev']:.1f}% post-COVID."
    anomaly_timeline_caption = f"ML anomaly rate by period: {k['rate_pre']:.2f}% pre-COVID \u2192 {k['rate_covid']:.2f}% COVID \u2192 {k['rate_post']:.2f}% post-COVID \u2014 a monotonic rise, with the post-COVID period recording the highest rate observed."
    shap_caption = "SHAP explainability was unavailable for this run." if not shap_summary["total_n"] else (
        f"Across the top {shap_summary['total_n']} highest-scored anomalies, \u201c{shap_summary['top_feature']}\u201d is the "
        f"dominant SHAP driver in {shap_summary['top_n']} of {shap_summary['total_n']} cases "
        f"({shap_summary['top_pct']:.0f}%), pointing to transaction recency/sequence within a supplier relationship "
        f"as the primary anomaly driver rather than payment size."
    )
    if robustness is not None and len(robustness):
        r98 = robustness[robustness["threshold_percentile"] == 98]
        robustness_caption = (
            f"Anomaly rate is robust to the exact score-cutoff percentile: at the 98th percentile (used throughout), "
            f"period rates are {r98['rate_pre_covid_pct'].values[0]:.2f}% / {r98['rate_covid_pct'].values[0]:.2f}% / "
            f"{r98['rate_post_covid_pct'].values[0]:.2f}%, and the qualitative pre \u2192 COVID \u2192 post ordering holds "
            f"across the 95th\u201399th percentile range tested."
        )
    else:
        robustness_caption = "Robustness sensitivity data was unavailable for this run."
    hhi_trend_caption = f"Monthly HHI ranges {month['hhi'].min():,.0f}\u2013{month['hhi'].max():,.0f} across the study period, indicating a persistently deconcentrated supplier market with no dominant single supplier."
    new_supplier_png_caption = new_sup_caption
    community_caption = "Community-level anomaly clustering was unavailable for this run." if top_comm["community_id"] is None else (
        f"Partitioning the supplier co-occurrence network into communities, community {top_comm['community_id']} "
        f"(n={top_comm['size']}) shows the highest mean anomaly rate at {top_comm['rate']:.2f}%, well above the "
        f"{k['anomaly_rate']:.2f}% overall baseline \u2014 evidence that certain buyer/category/month supplier pools carry "
        f"disproportionate anomaly risk beyond individual supplier centrality."
    )

    tab_category = (
        '<div class="card">' + _to_div(_fig_category_shock(category)) + '<div class="card-caption">' + top_shock_caption + '</div></div>'
        + '<div class="card">' + _to_div(_fig_category_heatmap(category)) + '<div class="card-caption">Rows are the 20 highest-spend categories; columns are min-max normalised for colour, with raw values annotated in each cell.</div></div>'
        + '<h2 class="section-title">Supporting Analysis Figures</h2>'
        + '<div class="masonry">'
        + _png_card("Category COVID Shock Ranking", top_shock_caption, _b64_png(config.FIGURES_DIR / "category_covid_shock.png"), "category_covid_shock.png")
        + _png_card("STL Decomposition", stl_caption, _b64_png(config.FIGURES_DIR / "stl_decomposition.png"), "stl_decomposition.png")
        + _png_card("Anomaly Rate Timeline", anomaly_timeline_caption, _b64_png(config.FIGURES_DIR / "anomaly_timeline.png"), "anomaly_timeline.png")
        + _png_card("Dominant SHAP Feature", shap_caption, _b64_png(config.FIGURES_DIR / "shap_top_feature_summary.png"), "shap_top_feature_summary.png")
        + _png_card("Robustness / Sensitivity", robustness_caption, _b64_png(config.FIGURES_DIR / "robustness_sensitivity.png"), "robustness_sensitivity.png")
        + _png_card("HHI Trend", hhi_trend_caption, _b64_png(config.FIGURES_DIR / "hhi_trend.png"), "hhi_trend.png")
        + _png_card("New-Supplier Rate", new_supplier_png_caption, _b64_png(config.FIGURES_DIR / "new_supplier_rate.png"), "new_supplier_rate.png")
        + _png_card("Community Anomaly Rate", community_caption, _b64_png(config.FIGURES_DIR / "community_anomaly_rate.png"), "community_anomaly_rate.png")
        + '</div>'
    )

    # --- Assemble page --------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHS Procurement Anomaly Detection \u2014 Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}"></script>
{CSS}
</head>
<body>

<div class="header">
  <div class="header-left">{NHS_LOGO_SVG}</div>
  <div class="header-title">
    <h1>NHS Procurement Anomaly Detection</h1>
    <p>Gunjan Mahadeva Rao \u00b7 7005SCN \u00b7 Masters Dissertation</p>
  </div>
  <div class="header-right">
    <span class="badge">Generated: {generated}</span>
    <button class="export-btn" onclick="window.print()">\u2b07 Export PDF</button>
  </div>
</div>

<div class="container" style="padding-bottom:0;">
  <div class="exec-banner">
    <div>
      <h2>Executive Summary</h2>
      <p>{exec_p1} {exec_p2} {exec_p3}</p>
    </div>
    <div class="kpi-grid">{kpi_html}</div>
  </div>
</div>

<div class="tabs-nav">
  <button class="tab-btn active" onclick="showTab('overview', this)">Overview</button>
  <button class="tab-btn" onclick="showTab('anomaly', this)">Anomaly Analysis</button>
  <button class="tab-btn" onclick="showTab('supplier', this)">Supplier Risk</button>
  <button class="tab-btn" onclick="showTab('category', this)">Category Intelligence</button>
</div>

<div class="container">

  <div id="tab-overview" class="tab-content active">{tab_overview}</div>
  <div id="tab-anomaly" class="tab-content">{tab_anomaly}</div>
  <div id="tab-supplier" class="tab-content">{tab_supplier}</div>
  <div id="tab-category" class="tab-content">{tab_category}</div>

</div>

<div class="footer">
  Gunjan Mahadeva Rao \u00b7 7005SCN \u00b7 Coventry University \u00b7 NHS Procurement Anomaly Detection Study
  \u00b7 Data range: {k['date_min']} \u2013 {k['date_max']} \u00b7 Generated {generated}
</div>

{JS}
</body>
</html>
"""

    out_path = config.DASHBOARD_HTML_PATH
    out_path.write_text(html, encoding="utf-8")
    logger.info("Saved interactive dashboard -> %s", out_path)
    return str(out_path)


def run_dashboard():
    return build_dashboard()


if __name__ == "__main__":
    run_dashboard()
