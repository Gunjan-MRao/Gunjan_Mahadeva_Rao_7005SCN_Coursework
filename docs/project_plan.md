# Project Plan and Gantt Chart

**Module:** 7005SCN Individual Research Project
**Student:** Gunjan Mahadeva Rao (15848655)
**Supervisor:** Dr. Nahid Salimi
**Repository:** [Gunjan_Mahadeva_Rao_7005SCN_Coursework](https://github.com/Gunjan-MRao/Gunjan_Mahadeva_Rao_7005SCN_Coursework)

This document responds directly to supervisor feedback from Dr. Salimi: *"The project
plan is realistic and well organised. More detailed project management information
would improve it. It could include a Gantt chart or more detailed milestones."* It
expands the original project plan with a week-by-week Gantt chart mapped to the
module's 9-week academic structure, a milestones table with target and actual dates,
a deliverables checklist mapped to real repository artefacts, a summary of work
delivered beyond the original proposal, and a brief risk log.

---

## 1. Gantt Chart

```mermaid
gantt
    title 7005SCN Project Timeline — Academic Weeks 1 to 9
    dateFormat YYYY-MM-DD
    axisFormat %d %b

    section Wk1 Module Induction and Expectations
    Module induction and expectations review      :done, wk1_induction, 2026-05-04, 2d
    Repository scaffolding and folder structure    :done, wk1_scaffold, 2026-05-06, 2d
    Conda environment and requirements.txt setup   :done, wk1_env, 2026-05-08, 3d

    section Wk2 Defining Research Focus and Ethics
    Research question and objectives defined                          :done, wk2_research, 2026-05-11, 2d
    Ethics approval form completed                                     :done, wk2_ethics, 2026-05-13, 2d
    Google Drive raw-data pipeline built (build_raw_from_drive.py)     :done, wk2_drive, 2026-05-15, 3d

    section Wk3 Project Planning, Risk and Ethics
    Project plan and risk log drafted                                          :done, wk3_plan, 2026-05-18, 2d
    Data loaders built (loaders.py)                                            :done, wk3_loaders, 2026-05-20, 2d
    Clean-merge pipeline built (clean_merge.py) - 326991 rows, 4 NHS sources    :done, wk3_merge, 2026-05-22, 3d

    section Wk4 Literature Review and Critical Thinking
    Literature review - anomaly detection and procurement fraud    :done, wk4_litreview, 2026-05-25, 6d
    Literature review - NHS spending and public-sector audit       :done, wk4_litreview2, 2026-05-31, 4d
    Central configuration module built (config.py)                 :done, wk4_config, 2026-06-04, 2d
    src package structure finalised                                :done, wk4_srcstruct, 2026-06-06, 2d

    section Wk5 Research Methods and Design
    Research methods selected - Isolation Forest and SHAP          :done, wk5_methods, 2026-06-08, 3d
    STL decomposition and HHI concentration methodology designed   :done, wk5_stlhhi, 2026-06-11, 3d
    Exploratory data analysis built (eda.py)                        :done, wk5_eda, 2026-06-14, 4d
    HHI concentration module built (hhi.py)                         :done, wk5_hhi, 2026-06-18, 4d

    section Wk6 Data Collection and Implementation
    Data collection finalised and STL shock analysis (stl_shock.py)             :done, wk6_data_stl, 2026-06-22, 4d
    Isolation Forest and SHAP scoring implemented (isolation_forest_shap.py)    :done, wk6_ifshap, 2026-06-26, 4d
    Audit red-flag validation implemented (audit_validation.py)                :done, wk6_audit, 2026-06-30, 3d
    Phase 6A multi-method benchmarking - LOF, One-Class SVM, Autoencoder        :done, wk6_phase6a, 2026-07-03, 3d

    section Wk7 Data Analysis and Interpretation
    Phase 6B synthetic anomaly injection evaluation - precision, recall, F1, PR-AUC       :done, wk7_phase6b, 2026-07-06, 4d
    Phase 6C statistical significance - hypergeometric test, bootstrap CIs, Mann-Whitney U :done, wk7_phase6c, 2026-07-10, 5d
    Phase 6D supplier-buyer bipartite network analysis - 32 communities, modularity 0.65   :done, wk7_phase6d, 2026-07-15, 5d

    section Wk8 Artefact Development and Evaluation
    Phase 7A composite supplier risk score - 3682 suppliers, 4 tiers               :done, wk8_phase7a, 2026-07-20, 2d
    Phase 7B category deep-dive (534 categories) and Phase 7C robustness checks    :done, wk8_phase7bc, 2026-07-22, 3d
    Phase 7D star-schema BI export and Phase 7E interactive Plotly dashboard        :done, wk8_phase7de, 2026-07-25, 4d
    Phase 7F 12 report figures committed and 36 pytest tests passing                :done, wk8_phase7f, 2026-07-29, 5d

    section Wk9 Writing the Dissertation
    Dissertation sections drafted (dissertation_sections.md)              :done, wk9_dissertation, 2026-08-03, 5d
    Progress update sent to supervisor (progress_update_dr_salimi.md)     :done, wk9_progress, 2026-08-08, 2d
    CI and GitHub Actions workflow finalised                              :done, wk9_ci, 2026-08-10, 2d
    Final review and submission                                          :done, wk9_final, 2026-08-12, 6d
```

---

## 2. Milestones Table

| Milestone | Academic Week | Target Date | Actual Date | Status |
|---|---|---|---|---|
| Module induction complete; repository scaffolded and environment ready | Wk1 | 2026-05-10 | 2026-05-10 | ✅ Complete |
| Research question defined, ethics form approved, Google Drive raw-data pipeline live | Wk2 | 2026-05-17 | 2026-05-17 | ✅ Complete |
| Project plan and risk log finalised; 326,991-row merge across 4 NHS sources complete | Wk3 | 2026-05-24 | 2026-05-24 | ✅ Complete |
| Literature review complete; config.py and src/ package structure finalised | Wk4 | 2026-06-07 | 2026-06-07 | ✅ Complete |
| Research methods selected (Isolation Forest + SHAP, STL, HHI); EDA modules complete | Wk5 | 2026-06-21 | 2026-06-21 | ✅ Complete |
| Data collection finalised; Isolation Forest/SHAP scoring and Phase 6A multi-method benchmarking complete | Wk6 | 2026-07-05 | 2026-07-05 | ✅ Complete |
| Phase 6B synthetic evaluation, Phase 6C statistical testing, Phase 6D network analysis complete | Wk7 | 2026-07-19 | 2026-07-19 | ✅ Complete |
| Phase 7A-7F artefact suite complete (risk scoring, category deep-dive, robustness, BI export, dashboard, figures); 36 pytest tests passing | Wk8 | 2026-08-02 | 2026-08-02 | ✅ Complete |
| Dissertation written, progress update sent to supervisor, CI/CD finalised, final submission | Wk9 | 2026-08-17 | 2026-08-17 | ✅ Complete |

---

## 3. Deliverables Checklist

| Deliverable | Proposed | Actual Artefact | Status |
|---|---|---|---|
| Novel Integrated NHS Dataset (2019-2024) | 4-source merge | `data/raw/*_clean.csv` — 326,991 rows, 4 sources | ✅ |
| Auditable Anomaly Detection System | Isolation Forest + SHAP | `src/modeling/isolation_forest_shap.py` + 4-method benchmarking + SHAP explainability | ✅ |
| Evidence-Based Policy Recommendations | NAO/NHS audit aligned | Risk tier table (185 Critical suppliers), `docs/dissertation_sections.md` | ✅ |
| BI Dashboard | Power BI / Tableau export | `reports/nhs_procurement_dashboard.html` + `data/processed/bi_export/` star schema (5 tables) | ✅ |
| 12 Report Figures | Matplotlib charts | `docs/figures/*.png` (12 files committed) | ✅ |
| Test Suite | Basic tests | `tests/` — 36 pytest tests, all passing, GitHub Actions CI | ✅ |

---

## 4. Beyond the Proposal

The following work was delivered above and beyond the three deliverables in the
original proposal:

- 4-method unsupervised detector benchmarking (Local Outlier Factor, One-Class SVM, MLP Autoencoder, Isolation Forest)
- Synthetic anomaly injection evaluation across 3 fraud archetypes (invoice inflation, ghost-vendor burst, round-number kickback), scored on precision, recall, F1 and PR-AUC
- Supplier-buyer bipartite network analysis — 5,602 nodes, 6,122 edges, 32 communities (Clauset-Newman-Moore greedy modularity, Q = 0.65)
- Formal statistical validation — hypergeometric exact test (log p ≈ -2,472.7), bootstrap confidence intervals (n = 2,000 resamples), Mann-Whitney U test with rank-biserial effect size
- Composite supplier risk score on a 0-100 scale across 4 tiers (Low/Medium/High/Critical), covering 3,682 scored suppliers
- Category-by-period deep-dive across 534 categories and 3 periods, with a COVID-shock growth ranking
- Three-dimension robustness checks — threshold sensitivity (95th/98th/99th percentile), period-boundary shift (±1 month), feature ablation (Jaccard = 0.707)
- Star-schema BI CSV export — 1 fact table (326,991 rows × 29 columns) plus 4 dimension tables
- Self-contained interactive Plotly HTML dashboard requiring no server or BI licence
- Automated Google Drive raw-data pipeline, removing the need for manual downloads
- GitHub Actions CI with 36 automated pytest tests running on every push

---

## 5. Risk Log

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data unavailability (NHS portal down) | Medium | High | Raw sources cached on Google Drive; automated pipeline re-downloads and rebuilds without manual intervention |
| Environment reproducibility across machines | Low | Medium | Conda environment with `requirements.txt` pinning tested package versions |
| Pipeline runtime exceeding available session time | Low | Low | Intermediate outputs cached; skip-if-exists logic avoids redundant recomputation |
| Scope creep beyond original proposal | Occurred | Positive | All additions (network analysis, statistical validation, robustness checks, BI export) strengthen the dissertation and were incorporated deliberately rather than displacing planned work |
