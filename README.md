
# Algorithmic Fairness in Credit Scoring: Base-Rate Mechanics vs. Proxy Discrimination

## 1. Overview

This empirical study audits and mitigates algorithmic bias in an automated credit scoring system. It specifically distinguishes between **proxy discrimination** (reconstructing protected traits from features) and **base-rate mechanics** (disparities arising mechanically from unequal historical group distributions).

## 2. Core Pipeline

* **Phase 1 (DGP):** Simulates $N=50,000$ credit applications. Financial features are statistically identical across groups, but historical labels (`loan_approved`) include an additive penalty against Group B.
* **Phase 2 (Baseline):** Trains an XGBoost model under "fairness-by-unawareness" (withholding gender), establishing a $\sim14.1\%$ equalized odds violation.
* **Phase 3 (Mitigation):** Benchmarks an in-processing reduction (Agarwal et al., 2018) against a post-processing Threshold Optimizer (Hardt et al., 2016).
* **Phase 4 (Mechanism Audit):** Computes and disaggregates Shapley values (`shap.TreeExplainer`) by group to isolate the statistical failure mode.

## 3. Empirical Results & The Pareto Frontier
Phase 2 and 3 evaluate the predictive-fairness trade-offs. The unconstrained unawareness baseline exhibits a severe equalized-odds violation (~14.1%), demonstrating that withholding protected attributes fails to secure fairness under label contamination. The table below contrasts the baseline with the optimized post-processing model:

| Metric                             | Baseline (Unawareness) | Mitigated (Threshold Optimizer) |
| **Accuracy**                       | 0.8137                 | 0.8321                          |        
| **Equalized Odds Violation**       | 0.1408                 | 0.0077                          |
   (Max TPR/FPR Difference) 
| **Disparate Impact Ratio**         | 0.9899                 | 0.7372                          |
(Group B Approval / Group A Approval)

* **The Fairness Trade-off:** The Threshold Optimizer eliminates the equalized-odds violation (bringing it below 0.01) but degrades the disparate impact ratio to 0.7372. This instantiates the canonical impossibility theorem of fairness (Kleinberg et al., 2016; Chouldechova, 2017): equalized odds and demographic parity are mutually exclusive under unequal group base rates. Selecting the optimal deployment threshold is a normative decision that empirical data alone cannot resolve. (In-processing reduction metrics are archived in `reports/mitigated_metrics.json`).

## 4. Key Diagnostic Finding
Disaggregating mean absolute SHAP values reveals negligible cross-group structural drift. The maximum observed drift occurs on `baseline_credit_score` ($+0.029$, representing $\sim1.3\%$ of the feature's global importance); drift across all other covariates is minor ($<0.002$). 

* **Conclusion:** The baseline's 14.08% equalized odds violation is **not** driven by implicit proxy encoding. It is a mechanical byproduct of a single global decision threshold interacting with divergent group base rates in the historical data.
## 4. Key Diagnostic Finding

Disaggregating mean absolute SHAP values reveals negligible cross-group structural drift. The maximum observed drift occurs on `baseline_credit_score` ($+0.029$, representing $\sim1.3\%$ of the feature's global importance); drift across all other covariates is minor ($<0.002$).

* **Conclusion:** The baseline's 14.08% equalized odds violation is **not** driven by implicit proxy encoding. It is a mechanical byproduct of a single global decision threshold interacting with divergent group base rates in the historical data.

## 5. Setup & Execution

### 5.1 Environment Setup

```bash
pip install polars numpy scikit-learn xgboost fairlearn shap matplotlib

```

### 5.2 Pipeline Execution

Run the entire empirical sequence from the repository root:

```bash
python run_pipeline.py

```

### 5.3 Repository Topology

```
.
├── run_pipeline.py              Pipeline orchestrator
├── README.md                    Flagship project documentation
├── src/
│   ├── generate_data.py         Phase 1: Synthetic data engine
│   ├── train_and_audit.py       Phase 2: Baseline & group audit
│   ├── mitigate_bias.py         Phase 3: Constrained optimization
│   └── explainability.py        Phase 4: SHAP diagnostic audit
├── data/raw/                    Polars Parquet artifacts
└── reports/
    ├── *.json                   Structured metric manifests
    └── figures/                 SHAP global & beeswarm plots

```
