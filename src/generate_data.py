from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Reproducibility and DGP configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 20260517
N_APPLICANTS: Final[int] = 50_000

DATA_DIR: Final[Path] = Path("data") / "raw"
OUTPUT_FILE: Final[Path] = DATA_DIR / "credit_applications.parquet"

# Demographic structure.
AGE_MIN: Final[float] = 22.0
AGE_MAX: Final[float] = 65.0
WORKING_AGE_FLOOR: Final[float] = 18.0  # no employment tenure accrues before 18

# Annual income: log-normal, positively correlated with age.
TARGET_MEAN_INCOME: Final[float] = 55_000.0
INCOME_AGE_COEF: Final[float] = 0.25  # loading of standardised age on log-income
INCOME_LOG_SD: Final[float] = 0.50    # idiosyncratic dispersion of log-income

# Employment tenure: per-row truncated normal on the support [0, age - 18].
EMP_MEAN_FRACTION: Final[float] = 0.45  # mean tenure as a fraction of the span
EMP_SD_FRACTION: Final[float] = 0.30    # sd of tenure as a fraction of the span

# Debt-to-income ratio: rescaled Beta variate, strictly inside (DTI_MIN, DTI_MAX).
DTI_MIN: Final[float] = 0.05
DTI_MAX: Final[float] = 0.65
DTI_BETA_A: Final[float] = 2.5
DTI_BETA_B: Final[float] = 4.0

# Baseline credit score: linear in a standardised merit composite plus noise.
SCORE_MIN: Final[float] = 300.0
SCORE_MAX: Final[float] = 850.0
SCORE_MIDPOINT: Final[float] = 575.0
SCORE_SPREAD: Final[float] = 85.0
SCORE_NOISE_SD: Final[float] = 0.35  # expressed in standardised-merit units

# Relative weights of the merit components feeding the credit score.
W_INCOME: Final[float] = 0.55
W_EMPLOYMENT: Final[float] = 0.30
W_DTI: Final[float] = 0.40  # enters negatively: higher leverage lowers the score

# Strategic systemic bias and the historical approval rule.
BIAS_PENALTY: Final[float] = 0.65        # latent-score penalty applied to gender == 1
APPROVAL_NOISE_SD: Final[float] = 0.55   # idiosyncratic noise in the historical rule
TARGET_APPROVAL_RATE: Final[float] = 0.50

LOGGER: Final[logging.Logger] = logging.getLogger("credit_dgp")


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
    LOGGER.info("Storage directory ready: %s", resolved)
    return resolved


# ---------------------------------------------------------------------------
# Primitive samplers (NumPy)
# ---------------------------------------------------------------------------
def sample_truncated_normal(
    rng: np.random.Generator,
    mean: np.ndarray,
    sd: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iterations: int = 200,
) -> np.ndarray:
    """Draw from a normal distribution truncated to per-row support [lower, upper].

    A vectorised rejection sampler is used: rows falling outside their bounds
    are repeatedly redrawn until valid. Any residual out-of-bound rows that
    survive ``max_iterations`` (numerically negligible for the configured
    parameters) are clipped to the boundary as a deterministic safeguard.

    Parameters
    ----------
    rng:
        The seeded NumPy generator driving all draws.
    mean, sd:
        Per-row location and scale of the latent (untruncated) normal.
    lower, upper:
        Per-row inclusive truncation bounds; broadcast to ``mean``'s shape.
    max_iterations:
        Maximum number of redraw passes before the clipping safeguard applies.

    Returns
    -------
    np.ndarray
        A float64 array of draws lying within [lower, upper] for every row.
    """
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), mean.shape).copy()
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), mean.shape).copy()

    samples = rng.normal(loc=mean, scale=sd)
    for iteration in range(max_iterations):
        invalid = (samples < lower) | (samples > upper)
        n_invalid = int(invalid.sum())
        if n_invalid == 0:
            LOGGER.info(
                "Truncated-normal sampler converged after %d iteration(s).",
                iteration + 1,
            )
            break
        samples[invalid] = rng.normal(loc=mean[invalid], scale=sd[invalid])
    else:
        LOGGER.warning(
            "Truncated-normal sampler reached the iteration cap; "
            "clipping residual out-of-bound rows to the boundary."
        )
        samples = np.clip(samples, lower, upper)
    return samples


def generate_gender(rng: np.random.Generator, n: int) -> np.ndarray:
    """Binary protected attribute with a 50/50 split (0 = Group A, 1 = Group B)."""
    return rng.binomial(n=1, p=0.5, size=n).astype(np.int8)


def generate_age(rng: np.random.Generator, n: int) -> np.ndarray:
    """Applicant age in years, uniform on the closed interval [AGE_MIN, AGE_MAX]."""
    return rng.uniform(low=AGE_MIN, high=AGE_MAX, size=n)


def generate_annual_income(rng: np.random.Generator, age: np.ndarray) -> np.ndarray:
    """Log-normal annual income, positively correlated with age.

    The mean of log-income is shifted by standardised age (loading
    ``INCOME_AGE_COEF``), inducing a positive age-income gradient. The
    intercept is solved in closed form from the log-normal mean identity
    ``E[X] = exp(mu + 0.5 * var)`` so that the *unconditional* mean income
    equals ``TARGET_MEAN_INCOME`` regardless of the chosen loadings.
    """
    n = age.shape[0]
    standardised_age = (age - age.mean()) / age.std()
    total_log_variance = INCOME_AGE_COEF**2 + INCOME_LOG_SD**2
    log_intercept = np.log(TARGET_MEAN_INCOME) - 0.5 * total_log_variance
    log_income = (
        log_intercept
        + INCOME_AGE_COEF * standardised_age
        + INCOME_LOG_SD * rng.standard_normal(size=n)
    )
    return np.exp(log_income)


def generate_employment_duration(
    rng: np.random.Generator, age: np.ndarray
) -> np.ndarray:
    """Employment tenure in years, truncated to [0, age - 18] for every applicant.

    Both the location and the scale of the latent normal are set as fractions
    of the applicant's available working span (``age - WORKING_AGE_FLOOR``).
    This keeps the truncation efficient and yields a tenure distribution that
    stays realistic across the full age range: a 22-year-old cannot record
    more than four years of tenure, while a 65-year-old can record many.
    """
    available_span = age - WORKING_AGE_FLOOR
    emp_mean = EMP_MEAN_FRACTION * available_span
    emp_sd = EMP_SD_FRACTION * available_span
    lower = np.zeros_like(available_span)
    return sample_truncated_normal(rng, emp_mean, emp_sd, lower, available_span)


def generate_debt_to_income(rng: np.random.Generator, n: int) -> np.ndarray:
    """Debt-to-income ratio: a rescaled Beta variate strictly inside (0.05, 0.65).

    The Beta support [0, 1] is affinely mapped onto [DTI_MIN, DTI_MAX]. A small
    epsilon clip removes the measure-zero boundary values that the Beta sampler
    can return numerically, guaranteeing the strict open-interval restriction.
    """
    beta_draws = rng.beta(a=DTI_BETA_A, b=DTI_BETA_B, size=n)
    dti = DTI_MIN + (DTI_MAX - DTI_MIN) * beta_draws
    epsilon = 1.0e-6
    return np.clip(dti, DTI_MIN + epsilon, DTI_MAX - epsilon)


# ---------------------------------------------------------------------------
# Frame assembly and Polars transformations
# ---------------------------------------------------------------------------
def assemble_raw_frame(rng: np.random.Generator) -> pl.DataFrame:
    """Sample every primitive column with NumPy and assemble them into a frame.

    Two auxiliary Gaussian noise columns are pre-sampled here and consumed by
    the downstream Polars pipeline, so that every stochastic draw originates
    from the single seeded generator and reproducibility is preserved.
    """
    n = N_APPLICANTS
    age = generate_age(rng, n)
    df = pl.DataFrame(
        {
            "applicant_id": np.arange(1, n + 1, dtype=np.int64),
            "gender": generate_gender(rng, n),
            "age": age,
            "annual_income": generate_annual_income(rng, age),
            "employment_duration_years": generate_employment_duration(rng, age),
            "debt_to_income_ratio": generate_debt_to_income(rng, n),
            "_score_noise": rng.normal(0.0, SCORE_NOISE_SD, size=n),
            "_approval_noise": rng.normal(0.0, APPROVAL_NOISE_SD, size=n),
        }
    )
    LOGGER.info(
        "Assembled raw feature frame: %d rows x %d columns.", df.height, df.width
    )
    return df


def _standardise(column: str) -> pl.Expr:
    """Return a Polars expression standardising ``column`` to mean 0, unit sd."""
    series = pl.col(column)
    return (series - series.mean()) / series.std()


def add_baseline_credit_score(df: pl.DataFrame) -> pl.DataFrame:
    """Construct the baseline credit score from a standardised merit composite.

    The merit composite is a weighted combination of (log) income, employment
    tenure and debt-to-income ratio. It depends only on financial fundamentals
    and never on the protected attribute, so the credit score itself carries
    no direct gender bias. The composite is restandardised before being mapped
    onto the [300, 850] range and perturbed by Gaussian measurement noise.
    """
    df = df.with_columns(pl.col("annual_income").log().alias("_log_income"))
    df = df.with_columns(
        (
            W_INCOME * _standardise("_log_income")
            + W_EMPLOYMENT * _standardise("employment_duration_years")
            - W_DTI * _standardise("debt_to_income_ratio")
        ).alias("_merit_raw")
    )
    df = df.with_columns(_standardise("_merit_raw").alias("_merit"))
    df = df.with_columns(
        (SCORE_MIDPOINT + SCORE_SPREAD * (pl.col("_merit") + pl.col("_score_noise")))
        .clip(SCORE_MIN, SCORE_MAX)
        .round(0)
        .cast(pl.Int64)
        .alias("baseline_credit_score")
    )
    return df


def add_biased_approval_decision(df: pl.DataFrame) -> pl.DataFrame:
    """Inject the systemic bias and derive the historical ``loan_approved`` label.

    The latent approval score is the standardised credit score plus
    idiosyncratic noise, minus a fixed penalty (``BIAS_PENALTY``) applied to
    every applicant in the disadvantaged group (``gender == 1``). The penalty
    is a flat additive constant: it is independent of income, employment
    tenure and debt-to-income ratio, and therefore constitutes pure,
    unjustified historical discrimination.

    The decision threshold is set to the empirical quantile of the latent score
    that delivers the configured overall approval rate. Anchoring the threshold
    to the aggregate rate ensures the injected bias surfaces as a disparity
    *between groups* rather than as a change in total acceptance volume.
    """
    df = df.with_columns(
        (
            _standardise("baseline_credit_score")
            + pl.col("_approval_noise")
            - BIAS_PENALTY * pl.col("gender")
        ).alias("_latent_approval")
    )
    threshold = df.select(
        pl.col("_latent_approval").quantile(
            1.0 - TARGET_APPROVAL_RATE, interpolation="linear"
        )
    ).item()
    LOGGER.info("Historical approval threshold (latent units): %.6f", threshold)
    df = df.with_columns(
        (pl.col("_latent_approval") > threshold).cast(pl.Int8).alias("loan_approved")
    )
    return df


def finalise_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Round, cast and select the published columns in their canonical order.

    ``age`` and ``employment_duration_years`` are rounded for readability.
    Because the two columns are rounded independently, rounding alone can let
    tenure marginally exceed ``age - 18``; the published tenure is therefore
    re-clipped to the rounded ``age - 18`` ceiling so the logical bound holds
    exactly in the persisted dataset.
    """
    df = df.with_columns(pl.col("age").round(2).alias("age"))
    df = df.with_columns(
        pl.col("employment_duration_years")
        .round(2)
        .clip(0.0, pl.col("age") - WORKING_AGE_FLOOR)
        .alias("employment_duration_years")
    )
    return df.select(
        pl.col("applicant_id").cast(pl.Int64),
        pl.col("gender").cast(pl.Int8),
        pl.col("age").cast(pl.Float64),
        pl.col("annual_income").round(2).cast(pl.Float64),
        pl.col("employment_duration_years").cast(pl.Float64),
        pl.col("debt_to_income_ratio").round(4).cast(pl.Float64),
        pl.col("baseline_credit_score").cast(pl.Int64),
        pl.col("loan_approved").cast(pl.Int8),
    )


# ---------------------------------------------------------------------------
# Diagnostics and persistence
# ---------------------------------------------------------------------------
def log_bias_diagnostics(df: pl.DataFrame) -> None:
    """Log group-conditional approval rates to confirm the injected disparity."""
    summary = (
        df.group_by("gender")
        .agg(
            pl.len().alias("n_applicants"),
            pl.col("loan_approved").mean().alias("approval_rate"),
            pl.col("baseline_credit_score").mean().alias("mean_credit_score"),
        )
        .sort("gender")
    )
    overall_rate = float(df.select(pl.col("loan_approved").mean()).item())
    LOGGER.info("Overall approval rate: %.4f", overall_rate)

    rates: dict[int, float] = {}
    for row in summary.to_dicts():
        gender = int(row["gender"])
        rates[gender] = float(row["approval_rate"])
        LOGGER.info(
            "Group %s | n=%d | approval rate=%.4f | mean credit score=%.1f",
            "A" if gender == 0 else "B",
            int(row["n_applicants"]),
            row["approval_rate"],
            row["mean_credit_score"],
        )

    if 0 in rates and 1 in rates:
        gap = rates[0] - rates[1]
        LOGGER.info(
            "Demographic-parity gap (Group A minus Group B): %.4f "
            "(%.2f percentage points).",
            gap,
            gap * 100.0,
        )


def export_parquet(df: pl.DataFrame, output_file: Path) -> None:
    """Write ``df`` to a Zstd-compressed Parquet file and log the artefact size."""
    df.write_parquet(output_file, compression="zstd")
    size_kib = output_file.stat().st_size / 1024.0
    LOGGER.info(
        "Wrote %d rows to %s (%.1f KiB).",
        df.height,
        output_file.resolve(),
        size_kib,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full Phase 1 pipeline: sample, transform, audit and persist."""
    configure_logging()
    LOGGER.info("=== Phase 1: Synthetic Credit Application Data Generation ===")
    LOGGER.info("Random seed: %d | Sample size: %d", RANDOM_SEED, N_APPLICANTS)

    rng = np.random.default_rng(RANDOM_SEED)
    ensure_directory(DATA_DIR)

    df = assemble_raw_frame(rng)
    df = add_baseline_credit_score(df)
    df = add_biased_approval_decision(df)
    final_df = finalise_frame(df)

    log_bias_diagnostics(final_df)
    export_parquet(final_df, OUTPUT_FILE)
    LOGGER.info("=== Data generation complete ===")


if __name__ == "__main__":
    main()
