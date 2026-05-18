from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parent

# The four phase scripts, listed in their mandatory execution order. Each tuple
# pairs a human-readable phase label with the script path relative to the
# repository root.
PHASES: Final[list[tuple[str, Path]]] = [
    ("Phase 1 - Synthetic Data Generation", Path("src") / "generate_data.py"),
    ("Phase 2 - Baseline Training and Bias Audit", Path("src") / "train_and_audit.py"),
    ("Phase 3 - Algorithmic Bias Mitigation", Path("src") / "mitigate_bias.py"),
    ("Phase 4 - Explainability and Proxy Auditing", Path("src") / "explainability.py"),
]

LOGGER: Final[logging.Logger] = logging.getLogger("pipeline")


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


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------
def run_phase(label: str, script_path: Path) -> None:
    """Execute a single phase script as a child process.

    The script is launched with the same Python interpreter that runs this
    orchestrator and with the repository root as the working directory, so the
    relative paths used inside each phase resolve correctly. Standard output
    and standard error are inherited by the child process, so each phase's own
    logging is streamed directly to the console in real time.

    Parameters
    ----------
    label:
        Human-readable phase name, used only in log messages.
    script_path:
        Path to the phase script, relative to the repository root.

    Raises
    ------
    FileNotFoundError
        If the phase script does not exist on disk.
    SystemExit
        If the phase process terminates with a non-zero exit code; the
        orchestrator aborts immediately with exit status 1.
    """
    absolute_script = (REPOSITORY_ROOT / script_path).resolve()
    if not absolute_script.is_file():
        LOGGER.error(
            "%s: script not found at %s. Aborting.", label, absolute_script
        )
        sys.exit(1)

    LOGGER.info("Starting %s (%s).", label, script_path.as_posix())
    start_time = time.monotonic()

    completed = subprocess.run(
        [sys.executable, str(absolute_script)],
        cwd=str(REPOSITORY_ROOT),
        check=False,
    )

    elapsed = time.monotonic() - start_time

    if completed.returncode != 0:
        LOGGER.error(
            "%s FAILED with exit code %d after %.1f s. Pipeline aborted.",
            label,
            completed.returncode,
            elapsed,
        )
        sys.exit(1)

    LOGGER.info("Completed %s in %.1f s.", label, elapsed)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute every phase in order, aborting on the first failure."""
    configure_logging()
    LOGGER.info("=== Algorithmic Fairness Pipeline: start ===")
    LOGGER.info("Repository root: %s", REPOSITORY_ROOT)
    LOGGER.info("Phases scheduled: %d", len(PHASES))

    pipeline_start = time.monotonic()
    for index, (label, script_path) in enumerate(PHASES, start=1):
        LOGGER.info("--- [%d/%d] %s ---", index, len(PHASES), label)
        run_phase(label, script_path)

    total_elapsed = time.monotonic() - pipeline_start
    LOGGER.info(
        "=== Algorithmic Fairness Pipeline: all %d phases completed in %.1f s ===",
        len(PHASES),
        total_elapsed,
    )


if __name__ == "__main__":
    main()
