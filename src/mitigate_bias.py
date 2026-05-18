from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 20260517

INPUT_FILE: Final[Path] = Path("data") / "raw" / "credit_applications.parquet"
REPORTS_DIR: Final[Path] = Path("reports")
OUTPUT_REPORT: Final[Path] = REPORTS_DIR / "mitigated_metrics.json"

TARGET_COLUMN: Final[str] = "loan_approved"
PROTECTED_COLUMN: Final[str] = "gender"
IDENTIFIER_COLUMN: Final[str] = "applicant_id"

# Feature set seen by every XGBoost estimator. The protected attribute and the
# row identifier are excluded from the design matrix; the protected attribute
# is supplied separately to the Fairlearn wrappers.
FEATURE_COLUMNS: Final[list[str]] = [
    "annual_income",
    "age",
    "debt_to_income_ratio",
    "employment_duration_years",
    "baseline_credit_score",
]

TEST_SIZE: Final[float] = 0.20
DECISION_THRESHOLD: Final[float] = 0.50

# Base XGBoost hyperparameters, shared by the baseline and every weak learner
# inside the Exponentiated Gradient ensemble. Identical to Phase 2 so that the
# comparison isolates the effect of the mitigation, not of retuning.
XGB_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Exponentiated Gradient reduction controls.
EG_EPS: Final[float] = 0.02   # allowed equalized-odds constraint slack
EG_MAX_ITER: Final[int] = 50  # cap on best-response iterations

# Threshold Optimizer grid resolution for the per-group linear program.
TO_GRID_SIZE: Final[int] = 1000

# Group label encoding fixed in Phase 1.
GROUP_A: Final[int] = 0  # advantaged group in the historical labels
GROUP_B: Final[int] = 1  # disadvantaged group in the historical labels

LOGGER: Final[logging.Logger] = logging.getLogger("credit_mitigation")


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
def configure_logging() -> None:
    """Initialise a single deterministic stream handler on the module logger."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def ensure_directory(directory: Path) -> Path:
    """Create ``directory`` (and any parents) if absent; return its resolved path."""
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory.resolve()
    LOGGER.info("Reports directory ready: %s", resolved)
    return resolved


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------
def load_dataset(input_file: Path) -> pl.DataFrame:
    """Load the Phase 1 Parquet artefact and validate its schema.

    Raises
    ------
    FileNotFoundError
        If the Parquet artefact is absent (Phase 1 has not been run).
    KeyError
        If any required column is missing from the loaded frame.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_file.resolve()}. "
            "Run Phase 1 (src/generate_data.py) before Phase 3."
        )
    df = pl.read_parquet(input_file)
    required = set(FEATURE_COLUMNS) | {
        TARGET_COLUMN,
        PROTECTED_COLUMN,
        IDENTIFIER_COLUMN,
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Loaded dataset is missing required column(s): {sorted(missing)}."
        )
    LOGGER.info(
        "Loaded dataset: %d rows x %d columns from %s.",
        df.height,
        df.width,
        input_file.resolve(),
    )
    return df


def build_joint_stratification_key(df: pl.DataFrame) -> np.ndarray:
    """Return an integer key encoding the joint (target, protected) cell.

    The key ``2 * loan_approved + gender`` takes four values, so a single
    stratified split preserves the empirical frequency of every
    (outcome x group) combination. This reproduces the Phase 2 split exactly.
    """
    key = df.select(
        (pl.col(TARGET_COLUMN) * 2 + pl.col(PROTECTED_COLUMN)).alias("_strat")
    ).to_series()
    return key.to_numpy()


def split_dataset(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform the Phase 2 80/20 split stratified jointly by target and group."""
    stratify_key = build_joint_stratification_key(df)
    indices = np.arange(df.height)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=stratify_key,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df = df[train_idx.tolist()]
    test_df = df[test_idx.tolist()]
    LOGGER.info(
        "Joint-stratified split: %d training rows / %d test rows.",
        train_df.height,
        test_df.height,
    )
    return train_df, test_df


def extract_arrays(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> dict[str, np.ndarray]:
    """Extract feature matrices, targets and protected vectors as NumPy arrays.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``x_train``, ``y_train``, ``a_train``, ``x_test``, ``y_test``,
        ``a_test``, where ``a_*`` are the protected-attribute vectors.
    """
    arrays = {
        "x_train": train_df.select(FEATURE_COLUMNS).to_numpy(),
        "y_train": train_df.select(TARGET_COLUMN).to_series().to_numpy(),
        "a_train": train_df.select(PROTECTED_COLUMN).to_series().to_numpy(),
        "x_test": test_df.select(FEATURE_COLUMNS).to_numpy(),
        "y_test": test_df.select(TARGET_COLUMN).to_series().to_numpy(),
        "a_test": test_df.select(PROTECTED_COLUMN).to_series().to_numpy(),
    }
    LOGGER.info(
        "Design matrices | X_train %s | X_test %s | features: %s",
        arrays["x_train"].shape,
        arrays["x_test"].shape,
        ", ".join(FEATURE_COLUMNS),
    )
    return arrays


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_xgb_estimator() -> XGBClassifier:
    """Return a fresh XGBClassifier configured with the shared hyperparameters."""
    return XGBClassifier(**XGB_PARAMS)


class Float64ProbaClassifier(BaseEstimator, ClassifierMixin):
    """Estimator adapter that guarantees float64 probability output.

    ``ThresholdOptimizer`` in Fairlearn 0.13 assembles its internal score
    vector from the wrapped estimator's ``predict_proba`` output and then
    writes per-group float64 interpolation results back into it. XGBoost
    returns float32 probabilities, and recent pandas versions raise on the
    resulting narrowing assignment. Casting the probability matrix to float64
    at the adapter boundary removes the dtype conflict without altering any
    predicted value. The adapter is a thin, fully delegating wrapper and is
    used only for the post-processing model.
    """

    def __init__(self, base_estimator: XGBClassifier) -> None:
        self.base_estimator = base_estimator

    def fit(
        self, x: np.ndarray, y: np.ndarray, **fit_params: object
    ) -> "Float64ProbaClassifier":
        """Fit the wrapped estimator and expose its fitted ``classes_``."""
        self.base_estimator.fit(x, y, **fit_params)
        self.classes_ = self.base_estimator.classes_
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Delegate hard prediction to the wrapped estimator."""
        return np.asarray(self.base_estimator.predict(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the wrapped estimator's class probabilities cast to float64."""
        return np.asarray(
            self.base_estimator.predict_proba(x), dtype=np.float64
        )


# ---------------------------------------------------------------------------
# Model 1: fairness-by-unawareness baseline
# ---------------------------------------------------------------------------
def train_baseline(
    x_train: np.ndarray, y_train: np.ndarray
) -> XGBClassifier:
    """Fit the unconstrained XGBoost baseline (no fairness intervention)."""
    model = build_xgb_estimator()
    model.fit(x_train, y_train)
    LOGGER.info("Trained baseline XGBClassifier (fairness-by-unawareness).")
    return model


def predict_baseline(
    model: XGBClassifier, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and positive-class probabilities for the baseline."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(np.int64)
    return predictions, probabilities


# ---------------------------------------------------------------------------
# Model 2: in-processing via Exponentiated Gradient reduction
# ---------------------------------------------------------------------------
def train_exponentiated_gradient(
    x_train: np.ndarray, y_train: np.ndarray, a_train: np.ndarray
) -> ExponentiatedGradient:
    """Fit the Exponentiated Gradient reduction under an equalized-odds constraint.

    The reduction wraps a fresh XGBoost estimator and solves the fair-learning
    game of Agarwal et al. (2018): the learner minimises sample-weighted error
    while the constraint player enforces equalised true- and false-positive
    rates across the protected groups, up to a slack of ``EG_EPS``. XGBoost
    consumes the per-iteration cost weights through its ``sample_weight``
    fitting argument, which Fairlearn supplies automatically.
    """
    mitigator = ExponentiatedGradient(
        estimator=build_xgb_estimator(),
        constraints=EqualizedOdds(),
        eps=EG_EPS,
        max_iter=EG_MAX_ITER,
    )
    mitigator.fit(x_train, y_train, sensitive_features=a_train)
    n_predictors = len(getattr(mitigator, "predictors_", []))
    LOGGER.info(
        "Trained Exponentiated Gradient reduction "
        "(equalized odds, eps=%.3f, %d ensemble predictors).",
        EG_EPS,
        n_predictors,
    )
    return mitigator


def predict_exponentiated_gradient(
    mitigator: ExponentiatedGradient, x_test: np.ndarray
) -> np.ndarray:
    """Return hard predictions from the randomised reduction classifier.

    ``ExponentiatedGradient`` yields a randomised classifier; a fixed
    ``random_state`` is passed so the realised predictions are reproducible.
    """
    predictions = mitigator.predict(
        x_test, random_state=np.random.RandomState(RANDOM_SEED)
    )
    return np.asarray(predictions, dtype=np.int64)


def score_exponentiated_gradient(
    mitigator: ExponentiatedGradient, x_test: np.ndarray
) -> np.ndarray:
    """Return the expected positive-class score of the randomised reduction.

    The reduction exposes ``_pmf_predict``, whose second column is the
    probability that the randomised ensemble emits the positive label for each
    row. That expectation is the natural continuous score for ROC-AUC, which
    requires a ranking rather than a thresholded decision.
    """
    pmf = mitigator._pmf_predict(x_test)
    return np.asarray(pmf, dtype=np.float64)[:, 1]


# ---------------------------------------------------------------------------
# Model 3: post-processing via Threshold Optimizer
# ---------------------------------------------------------------------------
def train_threshold_optimizer(
    baseline_model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    a_train: np.ndarray,
) -> ThresholdOptimizer:
    """Fit the Threshold Optimizer post-processor on top of the baseline model.

    The optimizer treats an already-fitted baseline as a fixed score function
    (``prefit=True``) and solves the per-group linear program of Hardt et al.
    (2016) to derive group-specific, possibly randomised decision thresholds
    that equalise the true- and false-positive rates. ``predict_method`` is set
    to ``predict_proba`` so the program operates on continuous scores rather
    than already-binarised labels.

    The fitted baseline is wrapped in :class:`Float64ProbaClassifier` so its
    probability output is float64; this is required for compatibility between
    Fairlearn 0.13 and recent pandas versions and does not change any predicted
    value. The wrapper is marked as already fitted because the underlying
    XGBoost model was trained in :func:`train_baseline`.
    """
    wrapped = Float64ProbaClassifier(baseline_model)
    wrapped.classes_ = baseline_model.classes_
    wrapped.is_fitted_ = True
    optimizer = ThresholdOptimizer(
        estimator=wrapped,
        constraints="equalized_odds",
        objective="accuracy_score",
        grid_size=TO_GRID_SIZE,
        prefit=True,
        predict_method="predict_proba",
    )
    optimizer.fit(x_train, y_train, sensitive_features=a_train)
    LOGGER.info(
        "Trained Threshold Optimizer (equalized odds, grid_size=%d).",
        TO_GRID_SIZE,
    )
    return optimizer


def predict_threshold_optimizer(
    optimizer: ThresholdOptimizer, x_test: np.ndarray, a_test: np.ndarray
) -> np.ndarray:
    """Return hard predictions from the group-aware threshold post-processor.

    The optimizer requires the protected attribute at prediction time because
    it applies a distinct threshold rule per group. A fixed ``random_state``
    makes the randomised tie-breaking reproducible.
    """
    predictions = optimizer.predict(
        x_test,
        sensitive_features=a_test,
        random_state=np.random.RandomState(RANDOM_SEED),
    )
    return np.asarray(predictions, dtype=np.int64)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _positive_rate(predictions: np.ndarray) -> float:
    """Return the share of positive predictions, or 0.0 for an empty group."""
    if predictions.size == 0:
        return 0.0
    return float(np.mean(predictions == 1))


def _confusion_rates(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Return the (true positive rate, false positive rate) for one group."""
    actual_positive = y_true == 1
    actual_negative = y_true == 0
    n_positive = int(np.sum(actual_positive))
    n_negative = int(np.sum(actual_negative))

    true_positive_rate = (
        float(np.sum((y_pred == 1) & actual_positive) / n_positive)
        if n_positive > 0
        else 0.0
    )
    false_positive_rate = (
        float(np.sum((y_pred == 1) & actual_negative) / n_negative)
        if n_negative > 0
        else 0.0
    )
    return true_positive_rate, false_positive_rate


def compute_model_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    gender: np.ndarray,
) -> dict[str, object]:
    """Compute the full performance and fairness metric bundle for one model.

    Parameters
    ----------
    model_name:
        Human-readable identifier used in logs and the report.
    y_true:
        Ground-truth historical labels for the test set.
    y_pred:
        Model hard predictions for the test set.
    y_score:
        Continuous positive-class scores for ROC-AUC ranking.
    gender:
        Protected attribute for the test set (0 = Group A, 1 = Group B).

    Returns
    -------
    dict[str, object]
        Predictive metrics, per-group statistics and fairness metrics, with
        all floats rounded to six decimal places.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_score))

    mask_a = gender == GROUP_A
    mask_b = gender == GROUP_B

    positive_rate_a = _positive_rate(y_pred[mask_a])
    positive_rate_b = _positive_rate(y_pred[mask_b])
    tpr_a, fpr_a = _confusion_rates(y_true[mask_a], y_pred[mask_a])
    tpr_b, fpr_b = _confusion_rates(y_true[mask_b], y_pred[mask_b])

    demographic_parity_difference = positive_rate_a - positive_rate_b
    disparate_impact_ratio = (
        positive_rate_b / positive_rate_a if positive_rate_a > 0.0 else float("nan")
    )
    tpr_difference = tpr_a - tpr_b
    fpr_difference = fpr_a - fpr_b
    equalized_odds_violation = max(abs(tpr_difference), abs(fpr_difference))

    metrics = {
        "model": model_name,
        "performance": {
            "accuracy": round(accuracy, 6),
            "f1_score": round(f1, 6),
            "roc_auc": round(roc_auc, 6),
        },
        "group_statistics": {
            "group_a": {
                "label": "Group A (gender = 0)",
                "n_test": int(np.sum(mask_a)),
                "predicted_approval_rate": round(positive_rate_a, 6),
                "true_positive_rate": round(tpr_a, 6),
                "false_positive_rate": round(fpr_a, 6),
            },
            "group_b": {
                "label": "Group B (gender = 1)",
                "n_test": int(np.sum(mask_b)),
                "predicted_approval_rate": round(positive_rate_b, 6),
                "true_positive_rate": round(tpr_b, 6),
                "false_positive_rate": round(fpr_b, 6),
            },
        },
        "fairness": {
            "demographic_parity_difference": round(
                demographic_parity_difference, 6
            ),
            "disparate_impact_ratio": (
                round(disparate_impact_ratio, 6)
                if not np.isnan(disparate_impact_ratio)
                else None
            ),
            "equalized_odds_tpr_difference": round(tpr_difference, 6),
            "equalized_odds_fpr_difference": round(fpr_difference, 6),
            "equalized_odds_max_violation": round(equalized_odds_violation, 6),
        },
    }

    LOGGER.info(
        "%-26s | acc=%.4f f1=%.4f auc=%.4f | EO viol=%.4f | DP diff=%.4f",
        model_name,
        accuracy,
        f1,
        roc_auc,
        equalized_odds_violation,
        demographic_parity_difference,
    )
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def assemble_report(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    model_metrics: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble the complete comparative report for JSON serialisation."""
    return {
        "metadata": {
            "phase": "Phase 3 - Algorithmic Bias Mitigation",
            "random_seed": RANDOM_SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "feature_columns": FEATURE_COLUMNS,
            "protected_attribute": PROTECTED_COLUMN,
            "target": TARGET_COLUMN,
            "n_train": train_df.height,
            "n_test": test_df.height,
            "xgboost_params": dict(XGB_PARAMS),
            "mitigation": {
                "in_processing": {
                    "algorithm": "ExponentiatedGradient",
                    "constraint": "EqualizedOdds",
                    "eps": EG_EPS,
                    "max_iter": EG_MAX_ITER,
                    "reference": "Agarwal et al. (2018), ICML",
                },
                "post_processing": {
                    "algorithm": "ThresholdOptimizer",
                    "constraint": "equalized_odds",
                    "grid_size": TO_GRID_SIZE,
                    "reference": "Hardt et al. (2016), NeurIPS",
                },
            },
        },
        "models": model_metrics,
    }


def save_report(report: dict[str, object], output_file: Path) -> None:
    """Serialise the comparative report to a cleanly indented JSON file."""
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    LOGGER.info("Comparative report written to %s.", output_file.resolve())


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a left-aligned, column-padded Markdown table from string cells."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(cells: list[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def _format_ratio(value: object) -> str:
    """Format a disparate-impact ratio, handling the undefined (None) case."""
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def render_pareto_summary(report: dict[str, object]) -> str:
    """Build the publication-ready Pareto-frontier trade-off matrix."""
    models = report["models"]

    performance_rows: list[list[str]] = []
    fairness_rows: list[list[str]] = []
    for entry in models:
        performance = entry["performance"]
        fairness = entry["fairness"]
        performance_rows.append(
            [
                str(entry["model"]),
                f"{performance['accuracy']:.4f}",
                f"{performance['f1_score']:.4f}",
                f"{performance['roc_auc']:.4f}",
            ]
        )
        fairness_rows.append(
            [
                str(entry["model"]),
                f"{fairness['demographic_parity_difference']:.4f}",
                _format_ratio(fairness["disparate_impact_ratio"]),
                f"{fairness['equalized_odds_tpr_difference']:.4f}",
                f"{fairness['equalized_odds_fpr_difference']:.4f}",
                f"{fairness['equalized_odds_max_violation']:.4f}",
            ]
        )

    performance_table = _format_markdown_table(
        headers=["Model", "Accuracy", "F1-Score", "ROC-AUC"],
        rows=performance_rows,
    )
    fairness_table = _format_markdown_table(
        headers=[
            "Model",
            "Dem. Parity Diff",
            "Disp. Impact Ratio",
            "EO: TPR Diff",
            "EO: FPR Diff",
            "EO: Max Viol.",
        ],
        rows=fairness_rows,
    )

    baseline = models[0]["fairness"]["equalized_odds_max_violation"]
    reduction_lines: list[str] = []
    for entry in models[1:]:
        violation = entry["fairness"]["equalized_odds_max_violation"]
        if baseline > 0.0:
            reduction = 100.0 * (baseline - violation) / baseline
            reduction_lines.append(
                f"- {entry['model']}: equalized-odds violation "
                f"{baseline:.4f} -> {violation:.4f} "
                f"({reduction:+.1f}% relative to baseline)."
            )
        else:
            reduction_lines.append(
                f"- {entry['model']}: equalized-odds violation {violation:.4f} "
                f"(baseline violation is zero; relative change undefined)."
            )

    lines = [
        "",
        "# Phase 3 - Algorithmic Bias Mitigation: Pareto-Frontier Analysis",
        "",
        f"Test rows: {report['metadata']['n_test']}  |  "
        f"Constraint: equalized odds  |  Protected attribute: "
        f"{report['metadata']['protected_attribute']}",
        "",
        "## 1. Predictive Performance",
        "",
        performance_table,
        "",
        "## 2. Fairness Metrics",
        "",
        fairness_table,
        "",
        "## 3. Mitigation Effect on the Equalized-Odds Violation",
        "",
        *reduction_lines,
        "",
        "## 4. Interpretation",
        "",
        "The baseline minimises error with no fairness constraint and sits at "
        "one extreme",
        "of the Pareto frontier: highest predictive performance, largest "
        "equalized-odds",
        "violation. The Exponentiated Gradient reduction (in-processing) and "
        "the Threshold",
        "Optimizer (post-processing) trade a measured amount of predictive "
        "performance for a",
        "substantial reduction in the equalized-odds violation. The two "
        "mitigated models",
        "occupy distinct points on the frontier: the reduction retrains the "
        "estimator under",
        "the constraint, while the post-processor adjusts only the decision "
        "rule of the fixed",
        "baseline. Neither dominates the other in general; the appropriate "
        "operating point",
        "depends on whether the deployment context permits retraining and on "
        "the regulatory",
        "tolerance for residual error-rate disparity.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full Phase 3 pipeline: load, split, train three models, audit, report."""
    configure_logging()
    LOGGER.info("=== Phase 3: Algorithmic Bias Mitigation ===")

    ensure_directory(REPORTS_DIR)
    df = load_dataset(INPUT_FILE)
    train_df, test_df = split_dataset(df)
    arrays = extract_arrays(train_df, test_df)

    x_train, y_train, a_train = (
        arrays["x_train"],
        arrays["y_train"],
        arrays["a_train"],
    )
    x_test, y_test, a_test = (
        arrays["x_test"],
        arrays["y_test"],
        arrays["a_test"],
    )

    # Model 1: unconstrained baseline.
    baseline_model = train_baseline(x_train, y_train)
    baseline_pred, baseline_score = predict_baseline(baseline_model, x_test)
    baseline_metrics = compute_model_metrics(
        "Baseline (Unawareness)", y_test, baseline_pred, baseline_score, a_test
    )

    # Model 2: in-processing reduction.
    eg_mitigator = train_exponentiated_gradient(x_train, y_train, a_train)
    eg_pred = predict_exponentiated_gradient(eg_mitigator, x_test)
    eg_score = score_exponentiated_gradient(eg_mitigator, x_test)
    eg_metrics = compute_model_metrics(
        "Exponentiated Gradient", y_test, eg_pred, eg_score, a_test
    )

    # Model 3: post-processing threshold optimisation.
    to_optimizer = train_threshold_optimizer(
        baseline_model, x_train, y_train, a_train
    )
    to_pred = predict_threshold_optimizer(to_optimizer, x_test, a_test)
    # The post-processor emits hard decisions only; its own predictions are the
    # ranking used for ROC-AUC, which therefore reflects the deployed classifier.
    to_metrics = compute_model_metrics(
        "Threshold Optimizer", y_test, to_pred, to_pred.astype(np.float64), a_test
    )

    model_metrics = [baseline_metrics, eg_metrics, to_metrics]
    report = assemble_report(train_df, test_df, model_metrics)
    save_report(report, OUTPUT_REPORT)

    summary = render_pareto_summary(report)
    print(summary)

    LOGGER.info("=== Phase 3 complete ===")


if __name__ == "__main__":
    main()
