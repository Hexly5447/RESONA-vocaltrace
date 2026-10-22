"""VocalTrace — acoustic phase-discontinuity detector for vishing defence.

VocalTrace detects AI-generated / cloned speech (deepfakes) by looking for
signal-processing artefacts left behind by neural vocoders — abrupt
frame-to-frame phase jumps and high-frequency roll-off — and classifying them
with lightweight ML (XGBoost or a small 1-D CNN) executed locally through
ONNX Runtime.  No audio ever leaves the machine.

The package surface is intentionally small::

    from vocaltrace import FeatureExtractor, DatasetBuilder, VocalTraceDetector

Submodules are imported lazily so that ``import vocaltrace`` stays fast and
works even when optional dependencies (torch, sounddevice) are missing.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "FeatureExtractor",
    "DatasetBuilder",
    "VocalTraceDetector",
    "__version__",
]

_LAZY_IMPORTS = {
    "FeatureExtractor": ("vocaltrace.features", "FeatureExtractor"),
    "DatasetBuilder": ("vocaltrace.dataset", "DatasetBuilder"),
    "VocalTraceDetector": ("vocaltrace.detector", "VocalTraceDetector"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access (keeps import time low)."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr = _LAZY_IMPORTS[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - introspection helper
    return sorted(__all__)
