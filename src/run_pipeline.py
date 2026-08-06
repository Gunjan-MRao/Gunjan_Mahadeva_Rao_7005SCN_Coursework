"""End-to-end orchestration for the NHS procurement anomaly-detection study.

Executes data acquisition, panel construction, descriptive analysis, anomaly
detection, validation, sensitivity analyses, and report-output generation in
the required dependency order. Phase 1 is conditional on the availability of
the consolidated raw inputs.
"""
from __future__ import annotations

import logging
import time

from src import config
from src.data_engineering import build_raw_from_drive
from src.data_engineering.clean_merge import clean_and_merge
from src.analysis.eda import run_eda
from src.analysis.hhi import compute_hhi
from src.analysis.stl_shock import run_stl_shock_analysis
from src.modeling.isolation_forest_shap import run_modeling_pipeline
from src.validation.audit_validation import run_validation
from src.modeling.method_comparison import run_method_comparison
from src.modeling.synthetic_evaluation import run_synthetic_evaluation
from src.analysis.statistical_tests import run_statistical_tests
from src.network.supplier_network import run_network_analysis
from src.analysis.supplier_risk_score import run_supplier_risk_score
from src.analysis.category_deep_dive import run_category_deep_dive
from src.analysis.robustness_checks import run_robustness_checks
from src.analysis.bi_export import run_bi_export
from src.analysis.dashboard import run_dashboard
from src.analysis.make_figures import run_make_figures

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Execute all pipeline phases and report study-level summary statistics."""
    t0 = time.time()

    # Acquire sources only when consolidated inputs are absent, preserving
    # manually provisioned raw data and reproducible input provenance.
    missing = [k for k, p in config.RAW_FILES.items() if not p.exists()]
    if missing:
        logger.info(
            "Consolidated raw file(s) missing (%s): running Phase 1 to fetch and "
            "consolidate the original per-month FOI/Contracts Finder exports from the "
            "project's public Google Drive archive. This downloads ~500MB and takes a "
            "few minutes; it is a one-off. To supply the files manually instead, see "
            "the README's 'Reproducing raw data from Google Drive' section.",
            ", ".join(missing),
        )
        build_raw_from_drive.build_all()

    logger.info("=" * 70)
    logger.info("PHASE 2: Data Engineering")
    logger.info("=" * 70)
    panel = clean_and_merge()

    logger.info("=" * 70)
    logger.info("PHASE 3: EDA, HHI, STL Shock Analysis")
    logger.info("=" * 70)
    dq, dist, new_supplier = run_eda()
    hhi_monthly, hhi_period_source, top_suppliers = compute_hhi(panel)
    stl_summary, stl_decomp = run_stl_shock_analysis(panel)

    logger.info("=" * 70)
    logger.info("PHASE 4: Isolation Forest + SHAP")
    logger.info("=" * 70)
    scored_df, shap_df, top_anomalies = run_modeling_pipeline()

    logger.info("=" * 70)
    logger.info("PHASE 5: Audit Red-Flag Validation")
    logger.info("=" * 70)
    validation_df = run_validation()

    logger.info("=" * 70)
    logger.info("PHASE 6A: Multi-Method Anomaly Detection Comparison")
    logger.info("=" * 70)
    # ``panel`` includes expenditure and contract-notice records. The comparison
    # and injection analyses independently load the trust-spend population used
    # by Isolation Forest, excluding notices without supplier-derived covariates.
    method_comparison_df, method_comparison_summary = run_method_comparison()

    logger.info("=" * 70)
    logger.info("PHASE 6B: Synthetic Anomaly Injection Evaluation")
    logger.info("=" * 70)
    synthetic_results_df, synthetic_by_type_df = run_synthetic_evaluation()

    logger.info("=" * 70)
    logger.info("PHASE 6C: Statistical Significance Testing")
    logger.info("=" * 70)
    bootstrap_df, mwu_df, hyper_result, circularity_df = run_statistical_tests()

    logger.info("=" * 70)
    logger.info("PHASE 6D: Supplier-Buyer Network / Collusion-Indicator Analysis")
    logger.info("=" * 70)
    network_node_df, hub_comparison, community_df, top_communities = run_network_analysis()

    logger.info("=" * 70)
    logger.info("PHASE 7A: Composite Supplier Risk Score")
    logger.info("=" * 70)
    risk_score_df, risk_hub_comparison = run_supplier_risk_score()

    logger.info("=" * 70)
    logger.info("PHASE 7B: Category-Level Deep Dive")
    logger.info("=" * 70)
    category_table_df, category_shock_ranking_df = run_category_deep_dive()

    logger.info("=" * 70)
    logger.info("PHASE 7C: Robustness Checks")
    logger.info("=" * 70)
    robustness_threshold_df, robustness_period_shift_df, robustness_ablation_df = run_robustness_checks()

    logger.info("=" * 70)
    logger.info("PHASE 7D: BI-Ready Data Export (star schema)")
    logger.info("=" * 70)
    bi_tables = run_bi_export()

    logger.info("=" * 70)
    logger.info("PHASE 7E: Interactive Plotly Dashboard")
    logger.info("=" * 70)
    dashboard_path = run_dashboard()

    logger.info("=" * 70)
    logger.info("PHASE 7F: Generate Report Figures")
    logger.info("=" * 70)
    run_make_figures()

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("=" * 70)

    n_anomaly = scored_df["is_anomaly"].sum()
    n_ml = int((validation_df["is_anomaly"]).sum()) if "is_anomaly" in validation_df else n_anomaly
    n_overlap = int((validation_df["is_anomaly"] & validation_df["rule_flagged"]).sum()) if "rule_flagged" in validation_df else None

    print("\n" + "=" * 70)
    print("HEADLINE SUMMARY")
    print("=" * 70)
    print(f"Total merged panel rows:        {len(panel):,}")
    print(f"Records scored by IsolationForest: {len(scored_df):,}")
    print(f"Anomalies flagged (98th pct):    {n_anomaly:,} ({n_anomaly/len(scored_df)*100:.2f}%)")
    print("\nAnomaly rate by period:")
    print(scored_df.groupby("period")["is_anomaly"].mean().mul(100).round(2).to_string())
    print("\nCOVID shock (STL decomposition), avg %% deviation from pre-COVID baseline:")
    print(stl_summary[["period", "pct_deviation_from_baseline"]].to_string(index=False))
    print("\nNew-supplier rate by period (%):")
    print(new_supplier.groupby("period")["new_supplier_rate_pct"].mean().round(2).to_string())
    if n_overlap is not None and n_ml:
        print(f"\nML/rule-based triangulation overlap: {n_overlap:,}/{n_ml:,} ({n_overlap/n_ml*100:.1f}% of ML flags corroborated by >=1 independent rule)")
        print(f"Hypergeometric significance test on this overlap: log(p)={hyper_result['log_p_value']:.1f} (p underflows float64 to 0; astronomically significant)")
    print("\nTriangulation circularity check (R3 shares is_new_supplier with the ML feature set):")
    print(circularity_df[["scenario", "rule_cols_used", "n_ml_flagged", "observed_overlap", "overlap_rate_pct", "log_p_value"]].to_string(index=False))
    print("\nMulti-method consensus (>=2/4 detectors agree):", f"{method_comparison_df['consensus_flag_majority'].sum():,}/{len(method_comparison_df):,} records")
    print("\nSynthetic-injection evaluation (precision/recall/F1 vs known injected anomalies):")
    print(synthetic_results_df.to_string(index=False))
    print(f"\nHub vs non-hub supplier anomaly rate: {hub_comparison['mean_anomaly_rate_hub_pct']:.2f}% vs {hub_comparison['mean_anomaly_rate_nonhub_pct']:.2f}% (Mann-Whitney p={hub_comparison['p_value']:.3g})")
    print("\nComposite supplier risk score, risk tier counts:")
    print(risk_score_df["risk_tier"].value_counts().to_string())
    print("\nTop 5 highest composite-risk suppliers:")
    print(risk_score_df[["supplier", "n_transactions", "composite_risk_score", "risk_tier"]].head(5).to_string(index=False))
    print("\nTop 3 categories by pre-COVID -> COVID spend growth:")
    print(category_shock_ranking_df[["category", "spend_pre_covid", "spend_covid", "pct_change_pre_to_covid"]].head(3).to_string(index=False))
    print("\nRobustness: threshold sensitivity (95/98/99th pct):")
    print(robustness_threshold_df[["threshold_percentile", "flagged_rate_pct", "rate_covid_pct", "rule_overlap_rate_pct"]].to_string(index=False))
    print("\nRobustness: feature ablation (drop is_new_supplier), Jaccard overlap vs baseline flags:", f"{robustness_ablation_df['jaccard_overlap'].iloc[0]:.3f}")
    print(f"\nBI-ready star-schema export ({len(bi_tables)} tables): {config.BI_EXPORT_DIR}")
    print(f"Interactive dashboard: {dashboard_path}")
    print(f"\nAll outputs saved under: {config.DATA_PROCESSED_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
