"""Shared utilities for VocalTrace: logging, reproducibility and path helpers.

This module is intentionally dependency-light (numpy only) so that importing it
from any entry point is cheap.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np

PathLike = Union[str, Path]

#: Absolute path of the repository root (the directory containing ``main.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
LOG_FILENAME = "vocaltrace.log"

#: Audio formats accepted anywhere in the pipeline (kept dependency-free so
#: the CLI can enumerate files without importing librosa).
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".oga", ".mp3", ".aiff", ".aif", ".m4a"}

_LOGGER_CONFIGURED = False


def set_global_seed(seed: int = 42) -> None:
    """Seed ``random``, ``numpy`` and (when loaded) ``torch``.

    Reproducibility requirement #3 of the VocalTrace spec.

    Note: ``torch`` is only seeded **if it has already been imported**.  We
    deliberately never *trigger* the torch import from here: on some platforms
    (notably macOS/arm64) loading both PyTorch's bundled ``libomp`` and
    xgboost's OpenMP runtime into one process can crash the interpreter, and
    VocalTrace keeps the XGBoost and CNN code paths in mutually exclusive
    processes for exactly that reason (see README → Troubleshooting).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch_module = sys.modules.get("torch")
    if torch_module is not None and hasattr(torch_module, "manual_seed"):
        torch_module.manual_seed(seed)


def setup_logging(log_dir: Optional[PathLike] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the package-level ``vocaltrace`` logger.

    Records are written to ``<log_dir>/vocaltrace.log`` (default:
    ``outputs/vocaltrace.log``) *and* mirrored to **stderr**, keeping stdout
    reserved for machine-readable output (verdicts, CSV data).

    The function is idempotent: repeated calls return the same logger without
    stacking duplicate handlers.
    """
    global _LOGGER_CONFIGURED
    root = logging.getLogger("vocaltrace")
    if _LOGGER_CONFIGURED:
        if level is not None:
            root.setLevel(level)
        return root

    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    target_dir = Path(log_dir) if log_dir is not None else DEFAULT_OUTPUT_DIR
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # read-only install, bad permissions, ...
        root.warning("Could not open log file in %s: %s", target_dir, exc)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.propagate = False
    _LOGGER_CONFIGURED = True
    return root


def get_logger(name: str = "vocaltrace") -> logging.Logger:
    """Return a namespaced child logger (e.g. ``vocaltrace.train``).

    Child loggers propagate to the handlers installed by :func:`setup_logging`.
    """
    if not name or name == "vocaltrace" or name.startswith("vocaltrace."):
        logger_name = name or "vocaltrace"
    else:
        logger_name = f"vocaltrace.{name}"
    return logging.getLogger(logger_name)


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def require_file(path: PathLike, what: str = "file") -> Path:
    """Validate that ``path`` exists and is a regular file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{what} not found: {target}")
    if not target.is_file():
        raise ValueError(f"{what} is not a regular file: {target}")
    return target


def require_dir(path: PathLike, what: str = "directory") -> Path:
    """Validate that ``path`` exists and is a directory."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"{what} not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"{what} is not a directory: {target}")
    return target
