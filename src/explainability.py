from __future__ import annotations

import matplotlib

# Headless backend must be selected before pyplot is imported so that figure
# generation succeeds on a server with no display.
matplotlib.use("Agg")

import json
import logging
import sys
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 20260517

INPUT_FILE: Final[Path] = Path("data") / "raw" / "credit_applications.parquet"
REPORTS_DIR: Final[Path] = Path("reports")
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
OUTPUT_REPORT: Final[Path] = REPORTS_DIR / "shap_explainability_metrics.json"
GLOBAL_PLOT: Final[Path] = FIGURES_DIR / "shap_global_importance.png"
BEESWARM_PLOT: Final[Path] = FIGURES_DIR / "shap_beeswarm_distribution.png"

TARGET_COLUMN: Final[str] = "loan_approved"
PROTECTED_COLUMN: Final[str] = "gender"
IDENTIFIER_COLUMN: Final[str] = "applicant_id"

# Feature set seen by the baseline model. The protected attribute and the row
# identifier are excluded -- this is the fairness-by-unawareness baseline whose
# internal reasoning Phase 4 dissects.
FEATURE_COLUMNS: Final[list[str]] = [
    "annual_income",
    "age",
    "debt_to_income_ratio",
    "employment_duration_years",
    "baseline_credit_score",
]

TEST_SIZE: Final[float] = 0.20

# XGBoost hyperparameters, identical to Phase 2 so the model dissected here is
# exactly the audited baseline.
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

# Figure rendering controls.
FIGURE_DPI: Final[int] = 200

LOGGER: Final[logging.Logger] = logging.getLogger("credit_explainability")


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
    LOGGER.info("Directory ready: %s", resolved)
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
            "Run Phase 1 (src/generate_data.py) before Phase 4."
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
    stratified split reproduces the exact partition used in Phases 2 and 3.
    """
    key = df.select(
        (pl.col(TARGET_COLUMN) * 2 + pl.col(PROTECTED_COLUMN)).alias("_strat")
    ).to_series()
    return key.to_numpy()


def split_dataset(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Perform the shared 80/20 split stratified jointly by target and group."""
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


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_baseline_model(
    train_df: pl.DataFrame,
) -> XGBClassifier:
    """Re-train the Phase 2 baseline XGBoost classifier on the training split."""
    x_train = train_df.select(FEATURE_COLUMNS).to_numpy()
    y_train = train_df.select(TARGET_COLUMN).to_series().to_numpy()
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)
    LOGGER.info(
        "Re-trained baseline XGBClassifier on %d rows (%d features).",
        x_train.shape[0],
        x_train.shape[1],
    )
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_values(
    model: XGBClassifier, x_test: np.ndarray
) -> np.ndarray:
    """Compute the full SHAP value matrix for the test observations.

    A :class:`shap.TreeExplainer` is used: for tree ensembles it computes exact
    Shapley values in polynomial time. The explainer is constructed in the
    ``raw`` (log-odds / margin) output space, so each SHAP value is the
    additive marginal contribution of a feature to the predicted approval
    log-odds. Working in log-odds space keeps the contributions linearly
    additive, which is the property the disaggregated audit relies upon.

    Parameters
    ----------
    model:
        The trained baseline XGBoost classifier.
    x_test:
        The test-set feature matrix (rows x ``FEATURE_COLUMNS``).

    Returns
    -------
    np.ndarray
        A ``(n_test, n_features)`` float64 array of SHAP values.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_names=FEATURE_COLUMNS,
        model_output="raw",
    )
    explanation = explainer(x_test)
    shap_matrix = np.asarray(explanation.values, dtype=np.float64)

    # For a binary XGBoost classifier the TreeExplainer returns a 2-D matrix
    # of shape (n_test, n_features). The defensive branch below collapses the
    # rare 3-D layout (n_test, n_features, n_classes) to the positive class so
    # the function's contract holds across SHAP versions.
    if shap_matrix.ndim == 3:
        shap_matrix = shap_matrix[:, :, -1]

    LOGGER.info(
        "Computed SHAP value matrix: %d observations x %d features.",
        shap_matrix.shape[0],
        shap_matrix.shape[1],
    )
    return shap_matrix


# ---------------------------------------------------------------------------
# Proxy feature auditing
# ---------------------------------------------------------------------------
def compute_global_importance(shap_matrix: np.ndarray) -> dict[str, float]:
    """Return the pooled mean absolute SHAP value for each feature.

    The mean absolute SHAP value is the standard global importance measure: it
    quantifies how much, on average, a feature moves the predicted approval
    log-odds regardless of direction.
    """
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    importance = {
        feature: round(float(value), 6)
        for feature, value in zip(FEATURE_COLUMNS, mean_abs)
    }
    for feature, value in importance.items():
        LOGGER.info("Global importance | %-26s = %.6f", feature, value)
    return importance


def compute_group_importance(
    shap_matrix: np.ndarray, gender: np.ndarray, group_value: int
) -> dict[str, float]:
    """Return the mean absolute SHAP value per feature for a single group.

    Parameters
    ----------
    shap_matrix:
        The full ``(n_test, n_features)`` SHAP value matrix.
    gender:
        The protected-attribute vector aligned row-wise with ``shap_matrix``.
    group_value:
        The group label to select (``GROUP_A`` or ``GROUP_B``).

    Returns
    -------
    dict[str, float]
        Mean absolute SHAP value per feature, restricted to the group's rows.
    """
    mask = gender == group_value
    if not np.any(mask):
        return {feature: 0.0 for feature in FEATURE_COLUMNS}
    group_mean_abs = np.mean(np.abs(shap_matrix[mask]), axis=0)
    return {
        feature: round(float(value), 6)
        for feature, value in zip(FEATURE_COLUMNS, group_mean_abs)
    }


def compute_structural_drift(
    importance_a: dict[str, float], importance_b: dict[str, float]
) -> dict[str, float]:
    """Return the per-feature cross-group structural drift in SHAP importance.

    The drift for a feature is the signed difference between its mean absolute
    SHAP value in Group A and in Group B. A drift far from zero means the model
    leans on that feature to a different degree depending on group membership.
    Because the model never sees gender, such a difference is the empirical
    signature of *proxy* behaviour: a financial covariate is partially standing
    in for the omitted protected attribute.

    Returns
    -------
    dict[str, float]
        Per-feature drift ``importance_a - importance_b``, rounded to six
        decimal places.
    """
    drift = {
        feature: round(importance_a[feature] - importance_b[feature], 6)
        for feature in FEATURE_COLUMNS
    }
    for feature, value in drift.items():
        LOGGER.info("Structural drift | %-26s = %+.6f", feature, value)
    return drift


def identify_dominant_proxy(drift: dict[str, float]) -> str:
    """Return the feature with the largest absolute cross-group drift."""
    return max(drift.items(), key=lambda item: abs(item[1]))[0]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_global_importance_plot(
    shap_matrix: np.ndarray, x_test: np.ndarray, output_file: Path
) -> None:
    """Render and save the global SHAP summary bar plot.

    The bar plot ranks features by pooled mean absolute SHAP value, giving the
    headline (group-agnostic) importance ordering.
    """
    plt.figure(figsize=(8.0, 5.0))
    shap.summary_plot(
        shap_matrix,
        features=x_test,
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        show=False,
    )
    plt.title(
        "Global Feature Importance (Mean |SHAP|)\n"
        "Baseline XGBoost Credit Classifier",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    LOGGER.info("Saved global importance plot to %s.", output_file.resolve())


def save_beeswarm_plot(
    shap_matrix: np.ndarray, x_test: np.ndarray, output_file: Path
) -> None:
    """Render and save the SHAP beeswarm distribution plot.

    The beeswarm plot shows, for every test observation, the signed SHAP value
    of each feature coloured by the feature's own value. It exposes the
    *direction* of each feature's influence on the approval log-odds, not just
    its magnitude.
    """
    plt.figure(figsize=(8.0, 5.0))
    shap.summary_plot(
        shap_matrix,
        features=x_test,
        feature_names=FEATURE_COLUMNS,
        plot_type="dot",
        show=False,
    )
    plt.title(
        "SHAP Value Distribution (Directional Impact)\n"
        "Baseline XGBoost Credit Classifier",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close("all")
    LOGGER.info("Saved beeswarm distribution plot to %s.", output_file.resolve())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def assemble_report(
    test_df: pl.DataFrame,
    global_importance: dict[str, float],
    importance_a: dict[str, float],
    importance_b: dict[str, float],
    structural_drift: dict[str, float],
) -> dict[str, object]:
    """Assemble the complete explainability report for JSON serialisation."""
    n_group_a = int(
        test_df.select((pl.col(PROTECTED_COLUMN) == GROUP_A).sum()).item()
    )
    n_group_b = int(
        test_df.select((pl.col(PROTECTED_COLUMN) == GROUP_B).sum()).item()
    )
    dominant_proxy = identify_dominant_proxy(structural_drift)
    return {
        "metadata": {
            "phase": "Phase 4 - Model Explainability and Proxy Feature Auditing",
            "model": "XGBClassifier",
            "fairness_protocol": "fairness-by-unawareness",
            "explainer": "shap.TreeExplainer",
            "shap_output_space": "raw (approval log-odds)",
            "random_seed": RANDOM_SEED,
            "feature_columns": FEATURE_COLUMNS,
            "protected_attribute": PROTECTED_COLUMN,
            "target": TARGET_COLUMN,
            "n_test": test_df.height,
            "n_test_group_a": n_group_a,
            "n_test_group_b": n_group_b,
        },
        "global_mean_abs_shap": global_importance,
        "group_mean_abs_shap": {
            "group_a": importance_a,
            "group_b": importance_b,
        },
        "structural_drift": structural_drift,
        "dominant_proxy_feature": dominant_proxy,
    }


def save_report(report: dict[str, object], output_file: Path) -> None:
    """Serialise the explainability report to a cleanly indented JSON file."""
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")
    LOGGER.info("Explainability report written to %s.", output_file.resolve())


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
    """Build the publication-ready disaggregated importance summary table."""
    global_importance = report["global_mean_abs_shap"]
    importance_a = report["group_mean_abs_shap"]["group_a"]
    importance_b = report["group_mean_abs_shap"]["group_b"]
    structural_drift = report["structural_drift"]
    dominant_proxy = report["dominant_proxy_feature"]

    # Order rows by descending absolute structural drift so the strongest
    # proxy signal sits at the top of the table.
    ordered_features = sorted(
        FEATURE_COLUMNS, key=lambda f: abs(structural_drift[f]), reverse=True
    )
    rows = [
        [
            feature,
            f"{global_importance[feature]:.6f}",
            f"{importance_a[feature]:.6f}",
            f"{importance_b[feature]:.6f}",
            f"{structural_drift[feature]:+.6f}",
        ]
        for feature in ordered_features
    ]
    table = _format_markdown_table(
        headers=[
            "Feature",
            "Global Mean |SHAP|",
            "Group A Mean |SHAP|",
            "Group B Mean |SHAP|",
            "Structural Drift (A - B)",
        ],
        rows=rows,
    )

    lines = [
        "",
        "# Phase 4 - Model Explainability and Proxy Feature Audit",
        "",
        f"Model: XGBClassifier  |  Explainer: shap.TreeExplainer  "
        f"|  Output space: approval log-odds",
        f"Test rows: {report['metadata']['n_test']}  "
        f"(Group A: {report['metadata']['n_test_group_a']}, "
        f"Group B: {report['metadata']['n_test_group_b']})",
        "",
        "## Disaggregated Feature Importance Matrix",
        "",
        table,
        "",
        "## Interpretation",
        "",
        f"Feature with the largest cross-group structural drift: "
        f"{dominant_proxy}.",
        "",
        "The baseline model never observes the protected attribute. Where a "
        "feature's mean",
        "absolute SHAP value differs between the groups, the model is applying "
        "that financial",
        "covariate with group-dependent weight -- the empirical signature of "
        "proxy behaviour,",
        "in which an observed variable partially reconstructs the omitted "
        "gender attribute.",
        "A drift near zero indicates the feature is used even-handedly across "
        "groups. These",
        "magnitudes are diagnostic, not causal: they quantify reliance, and "
        "should be read",
        "alongside the directional beeswarm plot and the Phase 2 fairness "
        "metrics.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full Phase 4 pipeline: load, split, train, explain, audit, report."""
    configure_logging()
    LOGGER.info("=== Phase 4: Model Explainability and Proxy Feature Auditing ===")

    ensure_directory(REPORTS_DIR)
    ensure_directory(FIGURES_DIR)

    df = load_dataset(INPUT_FILE)
    train_df, test_df = split_dataset(df)

    model = train_baseline_model(train_df)

    x_test = test_df.select(FEATURE_COLUMNS).to_numpy()
    gender_test = test_df.select(PROTECTED_COLUMN).to_series().to_numpy()

    shap_matrix = compute_shap_values(model, x_test)

    global_importance = compute_global_importance(shap_matrix)
    importance_a = compute_group_importance(shap_matrix, gender_test, GROUP_A)
    importance_b = compute_group_importance(shap_matrix, gender_test, GROUP_B)
    structural_drift = compute_structural_drift(importance_a, importance_b)

    save_global_importance_plot(shap_matrix, x_test, GLOBAL_PLOT)
    save_beeswarm_plot(shap_matrix, x_test, BEESWARM_PLOT)

    report = assemble_report(
        test_df,
        global_importance,
        importance_a,
        importance_b,
        structural_drift,
    )
    save_report(report, OUTPUT_REPORT)

    summary = render_console_summary(report)
    print(summary)

    LOGGER.info("=== Phase 4 complete ===")


if __name__ == "__main__":
    main()
