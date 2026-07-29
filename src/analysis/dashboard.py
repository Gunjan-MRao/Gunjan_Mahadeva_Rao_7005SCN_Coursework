"""
Phase 6E — Interactive Plotly HTML dashboard.

As explained in bi_export.py's module docstring, a genuine .pbix/.twbx file
cannot be reliably produced without the desktop BI application itself. This
module builds a self-contained, single-file HTML dashboard from the same
star-schema CSVs exported by bi_export.py (Phase 6D), laid out to mimic a
typical Power BI / Tableau report page: a row of headline KPI cards, trend
lines across the study period, and category/supplier/risk breakdown panels.
The output requires no server and no installed BI software -- open the .html
file directly in any browser. All charts are interactive (hover tooltips,
zoom, pan) via Plotly.js, which is embedded directly into the file so it
also works completely offline.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PERIOD_COLORS = {"Pre-COVID": "#94a3b8", "COVID": "#c1272d", "Post-COVID": "#1f5fae"}
TIER_COLORS = {"Low": "#94a3b8", "Medium": "#f4a259", "High": "#e07a5f", "Critical": "#c1272d"}
BRAND = "#1f5fae"


def _load_tables() -> dict:
    d = config.BI_EXPORT_DIR
    return {
        "fact": pd.read_csv(d / "fact_transactions.csv", low_memory=False),
        "supplier": pd.read_csv(d / "dim_supplier.csv"),
        "category": pd.read_csv(d / "dim_category.csv"),
        "period": pd.read_csv(d / "dim_period.csv"),
        "month": pd.read_csv(d / "dim_month.csv"),
    }


def _build_figure(t: dict) -> go.Figure:
    fact, supplier, category, month = t["fact"], t["supplier"], t["category"], t["month"]

    # "Total Spend Analysed" is restricted to trust_spend transactions (the
    # population anomaly detection actually runs on). UK_Contracts_Finder
    # records are contract *award notice* ceiling values, not incurred
    # spend, so summing them alongside trust_spend would badly overstate
    # total spend and mix incompatible units.
    trust_only = fact[fact["record_type"] == "trust_spend"]
    total_spend = trust_only["amount"].sum()
    total_txn = len(trust_only)
    ml_scored = fact["is_ml_scored"].sum()
    anomaly_rate = fact.loc[fact["is_ml_scored"] == True, "is_ml_anomaly"].mean() * 100  # noqa: E712
    n_suppliers_scored = len(supplier)
    n_critical = (supplier["risk_tier"] == "Critical").sum()

    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "xy", "colspan": 2, "secondary_y": True}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.12, 0.12, 0.38, 0.38],
        vertical_spacing=0.08,
        subplot_titles=(
            None, None, None, None,
            "Monthly Spend and ML Anomaly Rate Over Time",
            "Composite Supplier Risk Tier Distribution",
            "Top 10 Categories by Pre-COVID \u2192 COVID Spend Growth",
        ),
    )

    # --- Row 1-2: KPI cards -------------------------------------------------
    fig.add_trace(go.Indicator(
        mode="number", value=total_spend,
        number={"prefix": "£", "valueformat": ",.0f", "font": {"size": 34, "color": BRAND}},
        title={"text": "Total Trust Spend Analysed<br><sup>(trust_spend records only)</sup>"},
    ), row=1, col=1)
    fig.add_trace(go.Indicator(
        mode="number", value=total_txn,
        number={"valueformat": ",.0f", "font": {"size": 34, "color": BRAND}},
        title={"text": "Trust Spend Transactions<br><sup>(excludes contract award notices)</sup>"},
    ), row=1, col=2)
    fig.add_trace(go.Indicator(
        mode="number", value=anomaly_rate,
        number={"suffix": "%", "valueformat": ".2f", "font": {"size": 34, "color": "#c1272d"}},
        title={"text": f"ML Anomaly Rate<br><sup>({ml_scored:,.0f} trust_spend records scored)</sup>"},
    ), row=2, col=1)
    fig.add_trace(go.Indicator(
        mode="number", value=n_critical,
        number={"valueformat": ",.0f", "font": {"size": 34, "color": "#c1272d"}},
        title={"text": f"Critical-Risk Suppliers<br><sup>(of {n_suppliers_scored:,} scored)</sup>"},
    ), row=2, col=2)

    # --- Row 3: monthly spend + anomaly-rate trend (dual axis) -------------
    fig.add_trace(go.Bar(
        x=month["year_month"], y=month["total_spend"], name="Monthly spend (£)",
        marker_color=[PERIOD_COLORS.get(p, "#999") for p in month["period"]],
        opacity=0.85,
        hovertemplate="%{x}<br>Spend: £%{y:,.0f}<extra></extra>",
    ), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=month["year_month"], y=month["ml_anomaly_rate_pct"], name="ML anomaly rate (%)",
        mode="lines+markers", line=dict(color="#111827", width=2),
        hovertemplate="%{x}<br>Anomaly rate: %{y:.2f}%<extra></extra>",
    ), row=3, col=1, secondary_y=True)

    # --- Row 4 left: risk tier distribution ---------------------------------
    tier_order = ["Low", "Medium", "High", "Critical"]
    tier_counts = supplier["risk_tier"].value_counts().reindex(tier_order).fillna(0)
    fig.add_trace(go.Bar(
        x=tier_order, y=tier_counts.values, name="Suppliers",
        marker_color=[TIER_COLORS[t_] for t_ in tier_order],
        text=tier_counts.values.astype(int), textposition="outside",
        hovertemplate="%{x} risk: %{y} suppliers<extra></extra>", showlegend=False,
    ), row=4, col=1)

    # --- Row 4 right: top 10 categories by COVID shock growth ---------------
    cat = category.dropna(subset=["pct_change_pre_to_covid"]).sort_values(
        "pct_change_pre_to_covid", ascending=False,
    ).head(10).iloc[::-1]
    labels = [c[:28] + ("..." if len(c) > 28 else "") for c in cat["category"]]
    fig.add_trace(go.Bar(
        x=cat["pct_change_pre_to_covid"], y=labels, orientation="h", name="% growth",
        marker_color="#2a9d8f", showlegend=False,
        hovertemplate="%{y}<br>Pre-COVID to COVID: %{x:,.0f}%<extra></extra>",
    ), row=4, col=2)

    # --- Layout --------------------------------------------------------------
    fig.update_layout(
        height=1400, width=1300,
        title={
            "text": "NHS Procurement Anomaly Detection — Interactive Dashboard"
                    "<br><sup>Coursework 7005SCN | Data through "
                    f"{month['year_month'].max()} | Generated from the BI export (Phase 6D) star schema</sup>",
            "x": 0.5, "xanchor": "center",
        },
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=10)),
        margin=dict(t=140, l=60, r=40, b=60),
        bargap=0.15,
    )
    fig.update_xaxes(tickangle=45, row=3, col=1, tickfont=dict(size=8))
    fig.update_yaxes(title_text="Monthly spend (£)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ML anomaly rate (%)", row=3, col=1, secondary_y=True, showgrid=False)

    fig.update_xaxes(title_text="Composite risk tier", row=4, col=1)
    fig.update_yaxes(title_text="Number of suppliers", row=4, col=1)
    fig.update_xaxes(title_text="% change in category spend", row=4, col=2)

    return fig


def build_dashboard() -> str:
    t = _load_tables()
    fig = _build_figure(t)
    out_path = config.DASHBOARD_HTML_PATH
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    logger.info("Saved interactive dashboard -> %s", out_path)
    return str(out_path)


def run_dashboard():
    return build_dashboard()


if __name__ == "__main__":
    run_dashboard()
