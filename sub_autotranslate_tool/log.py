"""Logging setup for sub-autotranslate-tool.

All loggers live under the 'sub_autotranslate_tool' namespace so the level
can be controlled centrally. Level is set from the config file
(log_level = "info") and can be overridden with --log-level on the CLI.

Log levels in use:
  DEBUG    - full LLM request/response payloads (the exact text sent to the
             model), ffmpeg/ffprobe commands, parsing details.
  INFO     - high-level progress (files, models, configuration).
  WARNING  - retries, malformed LLM responses, skipped files.
  ERROR    - per-file failures that don't abort the run.
  CRITICAL - unrecoverable failures.
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER_NAME = "sub_autotranslate_tool"

LEVELS = ("debug", "info", "warning", "error", "critical")


def setup_logging(level: str = "info") -> None:
    """Configure the project root logger to write to stderr."""
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(
            f"invalid log level {level!r} (choose from: {', '.join(LEVELS)})"
        )

    handler = logging.StreamHandler(sys.stderr)
    if lvl <= logging.DEBUG:
        fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(levelname)s: %(message)s"
        datefmt = None
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(lvl)
    root.handlers[:] = [handler]
    root.propagate = False
