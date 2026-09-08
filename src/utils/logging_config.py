"""Central logging configuration.

Call ``setup_logging()`` once, as early as possible, in every entrypoint:
a notebook's first cell, the API's startup, the dashboard's top-level
script, or a CLI script. Every module then just does::

    import logging
    logger = logging.getLogger(__name__)
    logger.info("something happened")

and messages flow through the handlers configured here. Modules should
never call ``logging.basicConfig`` themselves and should never use
``print`` for anything other than a CLI's final human-facing output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger once for the whole application.

    Safe to call multiple times: subsequent calls are no-ops, so importing
    a module that also calls ``setup_logging()`` (e.g. in tests) will not
    duplicate handlers or double-log every message.

    Args:
        level: Root logging level, e.g. "DEBUG", "INFO", "WARNING".
        log_file: Optional path to also write logs to a file, in addition
            to stdout. Parent directories are created if missing.

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    global _configured
    if _configured:
        return

    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise TypeError(f"Unknown log level: {level!r}")

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )

    # Keep noisy third-party libraries at WARNING regardless of our level.
    for noisy_logger in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("Logging configured at level=%s", level)
