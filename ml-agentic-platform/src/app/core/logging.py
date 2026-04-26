import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for terminal output.

    Uses ``force=True`` so this call always wins even if a third-party library
    (e.g. MLflow, PyTorch) attached handlers before us.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
