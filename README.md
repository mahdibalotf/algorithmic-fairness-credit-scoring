# Algorithmic Fairness in Credit Scoring: Disentangling Proxy Discrimination from Base-Rate Mechanics

## 1. Executive Summary & Framework
The objective of this empirical study is to audit an automated credit decisioning system for group-level disparity and isolate the causal mechanism driving fairness violations. Applied fairness research frequently conflates two distinct failure modes: proxy discrimination (where a model reconstructs omitted protected attributes from correlated covariates) and base-rate mechanics (where a uniformly applied decision rule yields unequal error rates due to differing historical outcome distributions between groups).

This repository implements a fully reproducible end-to-end pipeline to disentangle these mechanisms:
* **Synthetic DGP (Phase 1):** Simulates 50,000 credit applications where financial features are generated independently of the protected attribute (gender), but historical target labels (`loan_approved`) are contaminated with an additive penalty against Group B.
* **Baseline Evaluation (Phase 2):** Trains an XGBoost classifier under a "fairness-by-unawareness" protocol to quantify how a blind model propagates historical label bias.
* **Algorithmic Mitigation (Phase 3):** Benchmarks an in-processing method (Exponentiated Gradient reduction; Agarwal et al., 2018) against a post-processing method (Threshold Optimizer; Hardt et al., 2016) to map the performance-fairness Pareto frontier.
* **Proxy Auditing (Phase 4):** Disaggregates game-theoretic feature attributions (exact `shap.TreeExplainer` Shapley values) across protected groups to test for structural drift in feature reliance.

## 2. Data Generating Process & Bias Injection
To ensure ground-truth verifiability, Phase 1 simulates $N=50,000$ records across five financial covariates: annual income, age, debt-to-income (DTI) ratio, employment duration, and baseline credit score. By construction, feature distributions across protected groups are statistically identical.

Systemic bias is injected solely into the historical target label (`loan_approved`). A flat, additive penalty ($\delta = 0.65$) is subtracted from the latent creditworthiness score of Group B applicants. The final binary target is obtained by thresholding the penalized latent score at a quantile that yields a fixed 50% aggregate approval rate. Consequently, the groups possess identical observed credentials but divergent base outcome rates, isolating the model's reaction to base-rate disparities from structural proxy encoding.

## 3. Empirical Results & The Pareto Frontier
Phase 2 and 3 evaluate the predictive-fairness trade-offs. The unconstrained unawareness baseline exhibits a severe equalized-odds violation (~14.1%), demonstrating that withholding protected attributes fails to secure fairness under label contamination. The table below contrasts the baseline with the optimized post-processing model:

| Metric | Baseline (Unawareness) | Mitigated (Threshold Optimizer) |
| :--- | :---: | :---: |
| **Accuracy** | 0.8137 | 0.8321 |
| **Equalized Odds Violation** (Max TPR/FPR Difference) | 0.1408 | 0.0077 |
| **Disparate Impact Ratio** (Group B / Group A Approval) | 0.9899 | 0.7372 |

The Threshold Optimizer minimizes the equalized-odds violation to $<0.01$ while marginally improving accuracy, as group-specific thresholds recover predictive utility that a rigid global threshold forgoes.

However, this optimization incurs a severe accounting cost under a different fairness paradigm: the disparate impact ratio degrades to 0.7372, crossing the US-EEOC four-fifths threshold. This behavior directly instantiates the canonical impossibility theorem of algorithmic fairness (Kleinberg et al., 2016; Chouldechova, 2017): under unequal group base rates, equalized odds and demographic parity are mutually exclusive. Selecting the optimal deployment threshold is a normative decision that empirical data alone cannot resolve. (In-processing reduction metrics are archived in `reports/mitigated_metrics.json`).

## 4. Mechanism Identification: Proxy vs. Base-Rate
Phase 4 determines whether the 14.08% equalized-odds violation stems from implicit proxy encoding or a mechanical base-rate effect. These mechanisms yield distinct, verifiable attribution profiles: proxy discrimination requires the model to assign group-dependent weights to covariates (causing significant cross-group structural drift in Shapley values), whereas a pure base-rate effect leaves feature weights symmetric across groups.

Disaggregating mean absolute SHAP values (computed in raw log-odds space) reveals negligible cross-group structural drift. The maximum observed drift occurs on `baseline_credit_score` ($+0.029$, representing $\sim1.3\%$ of the feature's global importance magnitude); drift across all other covariates is numerically minor ($<0.002$).

This confirms that the baseline model uses all financial covariates even-handedly. The equalized-odds violation is not driven by hidden proxy reconstruction, but is a mechanical byproduct of a single global decision threshold interacting with divergent group base rates. While SHAP disaggregation confirms symmetric feature utilization, full verification of the feature space's proxy capacity requires an auxiliary classifier to test if the covariates can predict the protected attribute—identified here as the subsequent research step.

## 5. Setup & Execution

### 5.1 Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install polars numpy scikit-learn xgboost fairlearn shap matplotlib
```

### 5.2 Pipeline Execution
Run the entire empirical sequence from the repository root:
```bash
python run_pipeline.py
```

Or execute phases independently in dependency order:
```bash
python src/generate_data.py      # Phase 1: DGP & Bias Injection
python src/train_and_audit.py    # Phase 2: Baseline & Group Audit
python src/mitigate_bias.py      # Phase 3: Algorithmic Mitigation
python src/explainability.py     # Phase 4: SHAP Diagnostic Audit
```

### 5.3 Repository Topology
```text
.
├── run_pipeline.py              Pipeline orchestrator
├── README.md                    Flagship project documentation
├── src/
│   ├── generate_data.py         Phase 1 synthetic data engine
│   ├── train_and_audit.py       Phase 2 training & auditing matrix
│   ├── mitigate_bias.py         Phase 3 constrained optimization
│   └── explainability.py        Phase 4 disaggregated SHAP audit
├── data/raw/                    Polars Parquet artifacts
└── reports/
    ├── *.json                   Structured metric manifests
    └── figures/                 SHAP global & beeswarm distribution plots
```

### 5.4 References
* Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., and Wallach, H. (2018). A Reductions Approach to Fair Classification. *International Conference on Machine Learning (ICML)*.
* Hardt, M., Price, E., and Srebro, N. (2016). Equality of Opportunity in Supervised Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
* Kleinberg, J., Mullainathan, S., and Raghavan, M. (2016). Inherent Trade-Offs in the Fair Determination of Risk Scores. *arXiv preprint arXiv:1609.05807*.
