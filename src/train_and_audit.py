from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 20260517

INPUT_FILE: Final[Path] = Path("data") / "raw" / "credit_applications.parquet"
REPORTS_DIR: Final[Path] = Path("reports")
OUTPUT_REPORT: Final[Path] = REPORTS_DIR / "baseline_audit_metrics.json"

TARGET_COLUMN: Final[str] = "loan_approved"
PROTECTED_COLUMN: Final[str] = "gender"
IDENTIFIER_COLUMN: Final[str] = "applicant_id"

# The feature set seen by the model. The protected attribute and the row
# identifier are deliberately excluded: this is the fairness-by-unawareness
# baseline.
FEATURE_COLUMNS: Final[list[str]] = [
    "annual_income",
    "age",
    "debt_to_income_ratio",
    "employment_duration_years",
    "baseline_credit_score",
]

TEST_SIZE: Final[float] = 0.20
DECISION_THRESHOLD: Final[float] = 0.50

# XGBoost hyperparameters. Modest, fixed values: the objective here is a
# reproducible, well-regularised baseline, not a tuned production model.
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

# Group label encoding fixed in Phase 1.
GROUP_A: Final[int] = 0  # advantaged group in the historical labels
GROUP_B: Final[int] = 1  # disadvantaged group in the historical labels

LOGGER: Final[logging.Logger] = logging.getLogger("credit_audit")


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

    Parameters
    ----------
    input_file:
        Path to the Parquet file written by Phase 1.

    Returns
    -------
    pl.DataFrame
        The validated credit-application dataset.

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
            "Run Phase 1 (src/generate_data.py) before Phase 2."
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

    The key takes one of four values, ``2 * loan_approved + gender``, so that
    a single-vector stratified split preserves the empirical frequency of
    every (outcome x group) combination in both partitions.
    """
    key = df.select(
        (pl.col(TARGET_COLUMN) * 2 + pl.col(PROTECTED_COLUMN)).alias("_strat")
    ).to_series()
    return key.to_numpy()


def split_dataset(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform an 80/20 split stratified jointly by target and protected group.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        The training and test frames, each retaining all original columns.
    """
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
    _log_stratification_balance(train_df, test_df)
    return train_df, test_df


def _log_stratification_balance(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> None:
    """Log the (target, group) cell proportions to confirm split consistency."""
    for name, frame in (("train", train_df), ("test", test_df)):
        composition = (
            frame.group_by(TARGET_COLUMN, PROTECTED_COLUMN)
            .agg(pl.len().alias("n"))
            .with_columns((pl.col("n") / frame.height).alias("share"))
            .sort(TARGET_COLUMN, PROTECTED_COLUMN)
        )
        for row in composition.to_dicts():
            LOGGER.info(
                "Split balance | %-5s | loan_approved=%d gender=%d | "
                "n=%d share=%.4f",
                name,
                int(row[TARGET_COLUMN]),
                int(row[PROTECTED_COLUMN]),
                int(row["n"]),
                float(row["share"]),
            )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def extract_design_matrices(
    train_df: pl.DataFrame, test_df: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the feature matrices and target vectors as NumPy arrays.

    The feature matrices are restricted to ``FEATURE_COLUMNS``; the protected
    attribute and identifier are intentionally excluded so the classifier is
    blind to group membership.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(X_train, y_train, X_test, y_test)``.
    """
    x_train = train_df.select(FEATURE_COLUMNS).to_numpy()
    x_test = test_df.select(FEATURE_COLUMNS).to_numpy()
    y_train = train_df.select(TARGET_COLUMN).to_series().to_numpy()
    y_test = test_df.select(TARGET_COLUMN).to_series().to_numpy()
    LOGGER.info(
        "Design matrices | X_train %s | X_test %s | features: %s",
        x_train.shape,
        x_test.shape,
        ", ".join(FEATURE_COLUMNS),
    )
    return x_train, y_train, x_test, y_test


def train_baseline_model(
    x_train: np.ndarray, y_train: np.ndarray
) -> XGBClassifier:
    """Fit the XGBoost baseline credit-risk classifier on the training data."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)
    LOGGER.info(
        "Trained XGBClassifier (%d estimators, max_depth=%d, lr=%.3f).",
        XGB_PARAMS["n_estimators"],
        XGB_PARAMS["max_depth"],
        XGB_PARAMS["learning_rate"],
    )
    return model


def generate_predictions(
    model: XGBClassifier, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and positive-class probabilities for the test set."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(np.int64)
    LOGGER.info(
        "Generated predictions for %d test rows (threshold=%.2f).",
        x_test.shape[0],
        DECISION_THRESHOLD,
    )
    return predictions, probabilities


# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------
def compute_performance_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute standard classification metrics for the baseline model.

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, F1-score and ROC-AUC, each rounded to
        six decimal places.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    rounded = {name: round(value, 6) for name, value in metrics.items()}
    for name, value in rounded.items():
        LOGGER.info("Performance | %-10s = %.6f", name, value)
    return rounded


# ---------------------------------------------------------------------------
# Bias auditing
# ---------------------------------------------------------------------------
def _positive_rate(predictions: np.ndarray) -> float:
    """Return the share of positive predictions, or 0.0 for an empty group."""
    if predictions.size == 0:
        return 0.0
    return float(np.mean(predictions == 1))


def _confusion_rates(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float]:
    """Return the (true positive rate, false positive rate) for one group.

    The true positive rate is the share of genuinely approved applicants who
    are predicted approved; the false positive rate is the share of genuinely
    rejected applicants who are predicted approved. Either rate is reported as
    0.0 when its conditioning base (actual positives or actual negatives) is
    empty.
    """
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


def audit_group_fairness(
    y_true: np.ndarray, y_pred: np.ndarray, gender: np.ndarray
) -> dict[str, object]:
    """Run the formal group-fairness audit across the protected attribute.

    Metrics
    -------
    Demographic Parity Difference
        ``P(Yhat = 1 | gender = 0) - P(Yhat = 1 | gender = 1)``. Zero under
        statistical parity; a positive value indicates Group A is approved
        more often than Group B.
    Disparate Impact Ratio
        ``P(Yhat = 1 | gender = 1) / P(Yhat = 1 | gender = 0)``. Equals 1.0
        under parity; the US-EEOC "four-fifths rule" treats values below 0.80
        as evidence of adverse impact against Group B.
    Equalized Odds
        The between-group differences in the true positive rate and the false
        positive rate. Equalized odds holds when both differences are zero.

    Parameters
    ----------
    y_true:
        Ground-truth historical labels for the test set.
    y_pred:
        Model hard predictions for the test set.
    gender:
        Protected attribute for the test set (0 = Group A, 1 = Group B).

    Returns
    -------
    dict[str, object]
        A nested dictionary of per-group statistics and aggregate fairness
        metrics, with all floats rounded to six decimal places.
    """
    mask_a = gender == GROUP_A
    mask_b = gender == GROUP_B

    y_true_a, y_pred_a = y_true[mask_a], y_pred[mask_a]
    y_true_b, y_pred_b = y_true[mask_b], y_pred[mask_b]

    positive_rate_a = _positive_rate(y_pred_a)
    positive_rate_b = _positive_rate(y_pred_b)

    tpr_a, fpr_a = _confusion_rates(y_true_a, y_pred_a)
    tpr_b, fpr_b = _confusion_rates(y_true_b, y_pred_b)

    demographic_parity_difference = positive_rate_a - positive_rate_b
    disparate_impact_ratio = (
        positive_rate_b / positive_rate_a if positive_rate_a > 0.0 else float("nan")
    )
    tpr_difference = tpr_a - tpr_b
    fpr_difference = fpr_a - fpr_b
    equalized_odds_violation = max(abs(tpr_difference), abs(fpr_difference))

    audit = {
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
        "fairness_metrics": {
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
        "interpretation": {
            "four_fifths_rule_satisfied": (
                bool(disparate_impact_ratio >= 0.80)
                if not np.isnan(disparate_impact_ratio)
                else None
            ),
        },
    }

    LOGGER.info(
        "Fairness | demographic parity difference = %.6f",
        demographic_parity_difference,
    )
    LOGGER.info(
        "Fairness | disparate impact ratio = %s",
        "undefined"
        if np.isnan(disparate_impact_ratio)
        else f"{disparate_impact_ratio:.6f}",
    )
    LOGGER.info(
        "Fairness | equalized odds TPR diff = %.6f | FPR diff = %.6f",
        tpr_difference,
        fpr_difference,
    )
    return audit


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def assemble_report(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    performance: dict[str, float],
    fairness_audit: dict[str, object],
) -> dict[str, object]:
    """Assemble the complete structured audit report for JSON serialisation."""
    return {
        "metadata": {
            "phase": "Phase 2 - Baseline Model Training and Bias Auditing",
            "model": "XGBClassifier",
            "fairness_protocol": "fairness-by-unawareness",
            "random_seed": RANDOM_SEED,
            "decision_threshold": DECISION_THRESHOLD,
            "feature_columns": FEATURE_COLUMNS,
            "protected_attribute": PROTECTED_COLUMN,
            "target": TARGET_COLUMN,
            "n_train": train_df.height,
            "n_test": test_df.height,
            "xgboost_params": {
                key: value for key, value in XGB_PARAMS.items()
            },
        },
        "performance_metrics": performance,
        "bias_audit": fairness_audit,
    }


def save_report(report: dict[str, object], output_file: Path) -> None:
    """Serialise the audit report to a cleanly indented JSON file."""
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    LOGGER.info("Audit report written to %s.", output_file.resolve())


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


def render_console_summary(report: dict[str, object]) -> str:
    """Build the professional Markdown summary printed at the end of the run."""
    performance = report["performance_metrics"]
    audit = report["bias_audit"]
    group_a = audit["group_statistics"]["group_a"]
    group_b = audit["group_statistics"]["group_b"]
    fairness = audit["fairness_metrics"]

    performance_table = _format_markdown_table(
        headers=["Metric", "Value"],
        rows=[
            ["Accuracy", f"{performance['accuracy']:.4f}"],
            ["Precision", f"{performance['precision']:.4f}"],
            ["Recall", f"{performance['recall']:.4f}"],
            ["F1-Score", f"{performance['f1_score']:.4f}"],
            ["ROC-AUC", f"{performance['roc_auc']:.4f}"],
        ],
    )

    group_table = _format_markdown_table(
        headers=[
            "Group",
            "N (test)",
            "Pred. Approval Rate",
            "TPR",
            "FPR",
        ],
        rows=[
            [
                "Group A (gender = 0)",
                str(group_a["n_test"]),
                f"{group_a['predicted_approval_rate']:.4f}",
                f"{group_a['true_positive_rate']:.4f}",
                f"{group_a['false_positive_rate']:.4f}",
            ],
            [
                "Group B (gender = 1)",
                str(group_b["n_test"]),
                f"{group_b['predicted_approval_rate']:.4f}",
                f"{group_b['true_positive_rate']:.4f}",
                f"{group_b['false_positive_rate']:.4f}",
            ],
        ],
    )

    disparate_impact = fairness["disparate_impact_ratio"]
    disparate_impact_text = (
        f"{disparate_impact:.4f}" if disparate_impact is not None else "undefined"
    )
    four_fifths = report["bias_audit"]["interpretation"][
        "four_fifths_rule_satisfied"
    ]
    four_fifths_text = (
        "n/a"
        if four_fifths is None
        else ("satisfied" if four_fifths else "VIOLATED")
    )

    fairness_table = _format_markdown_table(
        headers=["Fairness Metric", "Value", "Parity Reference"],
        rows=[
            [
                "Demographic Parity Difference",
                f"{fairness['demographic_parity_difference']:.4f}",
                "0.0000",
            ],
            [
                "Disparate Impact Ratio",
                disparate_impact_text,
                "1.0000 (>= 0.80 rule)",
            ],
            [
                "Equalized Odds: TPR Difference",
                f"{fairness['equalized_odds_tpr_difference']:.4f}",
                "0.0000",
            ],
            [
                "Equalized Odds: FPR Difference",
                f"{fairness['equalized_odds_fpr_difference']:.4f}",
                "0.0000",
            ],
            [
                "Equalized Odds: Max Violation",
                f"{fairness['equalized_odds_max_violation']:.4f}",
                "0.0000",
            ],
        ],
    )

    lines = [
        "",
        "# Phase 2 - Baseline Model Training and Bias Audit Summary",
        "",
        f"Model: XGBClassifier  |  Protocol: fairness-by-unawareness  "
        f"|  Test rows: {report['metadata']['n_test']}",
        "",
        "## 1. Predictive Performance",
        "",
        performance_table,
        "",
        "## 2. Group-Conditional Behaviour",
        "",
        group_table,
        "",
        "## 3. Group-Fairness Audit",
        "",
        fairness_table,
        "",
        "## 4. Interpretation",
        "",
        f"Four-fifths (80%) adverse-impact rule: {four_fifths_text}.",
        "",
        "The classifier never observes the protected attribute, yet a "
        "demographic-parity",
        "difference and an equalized-odds violation persist. The disparity is "
        "reconstructed",
        "from features correlated with the historically biased labels: "
        "fairness-by-unawareness",
        "does not neutralise the structural bias embedded in Phase 1. This "
        "motivates the",
        "explicit bias-mitigation work in the next phase.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full Phase 2 pipeline: load, split, train, evaluate, audit, report."""
    configure_logging()
    LOGGER.info("=== Phase 2: Baseline Model Training and Bias Auditing ===")

    ensure_directory(REPORTS_DIR)
    df = load_dataset(INPUT_FILE)
    train_df, test_df = split_dataset(df)

    x_train, y_train, x_test, y_test = extract_design_matrices(train_df, test_df)
    model = train_baseline_model(x_train, y_train)
    predictions, probabilities = generate_predictions(model, x_test)

    performance = compute_performance_metrics(y_test, predictions, probabilities)

    gender_test = test_df.select(PROTECTED_COLUMN).to_series().to_numpy()
    fairness_audit = audit_group_fairness(y_test, predictions, gender_test)

    report = assemble_report(train_df, test_df, performance, fairness_audit)
    save_report(report, OUTPUT_REPORT)

    summary = render_console_summary(report)
    print(summary)

    LOGGER.info("=== Phase 2 complete ===")


if __name__ == "__main__":
    main()
