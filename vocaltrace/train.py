"""Module 3 — model training and ONNX export for VocalTrace.

Two classifiers are provided:

* **XGBoost** (default) — trained on the flattened mean/std/p90 feature
  vector; exported with ``onnxmltools`` to ``models/vocaltrace.onnx``.
* **1-D CNN** (``--model cnn``) — a 3-block Conv1d network over the raw
  ``(634, frames)`` feature stack; exported with ``torch.onnx.export``
  (dynamic time axis).  Requires the optional ``torch`` dependency.

Every export is verified by re-running the ONNX graph with *onnxruntime* and
comparing against the source model (tolerance ``1e-5``) before it is accepted.

Seeds are pinned to 42 (requirement #3) so runs are reproducible.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .dataset import DatasetBuilder
from .detector import (
    _probability_from_output,
    load_session,
    select_probability_output,
    session_probabilities,
)
from .features import FeatureExtractor
from .utils import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    ensure_dir,
    set_global_seed,
    setup_logging,
)

LOGGER = logging.getLogger("vocaltrace.train")

PathLike = Union[str, Path]
SEED = 42
#: Maximum accepted difference between the source model and its ONNX export.
ONNX_PARITY_TOLERANCE = 1e-5
#: Output file names (spec-mandated).
XGB_MODEL_FILENAME = "vocaltrace_xgb.json"
CNN_MODEL_FILENAME = "vocaltrace_cnn.pt"
ONNX_MODEL_FILENAME = "vocaltrace.onnx"


# --------------------------------------------------------------------------- #
# reproducibility + metrics
# --------------------------------------------------------------------------- #
def set_seed(seed: int = SEED) -> None:
    """Seed python / numpy / torch (thin wrapper over utils.set_global_seed)."""
    set_global_seed(seed)


def evaluate_binary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    ``roc_auc`` is ``None`` when the ground truth contains a single class
    (undefined) — callers should treat it as "not available" rather than 0.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        raise ValueError("y_true is empty — nothing to evaluate")
    if y_true.size != y_pred.size:
        raise ValueError(
            f"y_true ({y_true.size}) and y_pred ({y_pred.size}) must have the "
            "same length"
        )

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is None:
        metrics["roc_auc"] = None
    elif np.unique(y_true).size < 2:
        LOGGER.warning("ROC-AUC undefined for a single-class ground truth; skipping")
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, np.asarray(y_proba, dtype=np.float64))
        )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.astype(int).tolist()  # [[tn, fp], [fn, tp]]
    metrics["n_samples"] = int(y_true.size)
    return metrics


def format_metrics(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    """Render a metrics dict as a compact multi-line string."""
    auc = metrics.get("roc_auc")
    auc_txt = "n/a" if auc is None else f"{auc:.4f}"
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    lines = [
        f"{title}",
        f"  samples   : {metrics.get('n_samples', '?')}",
        f"  accuracy  : {metrics.get('accuracy', float('nan')):.4f}",
        f"  precision : {metrics.get('precision', float('nan')):.4f}",
        f"  recall    : {metrics.get('recall', float('nan')):.4f}",
        f"  f1        : {metrics.get('f1', float('nan')):.4f}",
        f"  roc_auc   : {auc_txt}",
        f"  confusion : TP={cm[1][1]} FP={cm[0][1]} FN={cm[1][0]} TN={cm[0][0]}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def make_xgb_classifier(**overrides: Any):
    """Build the spec-default :class:`~xgboost.XGBClassifier`.

    ``XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="binary:logistic", eval_metric="auc")`` with
    ``use_label_encoder=False`` added for xgboost < 2.0 (the parameter was
    removed in xgboost 2.x and would raise "unknown parameter").
    """
    import xgboost
    from xgboost import XGBClassifier

    params: Dict[str, Any] = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    try:
        major = int(str(xgboost.__version__).split(".")[0])
    except (ValueError, IndexError):  # pragma: no cover - dev versions
        major = 99
    if major < 2:
        # Spec-mandated flag; only understood by xgboost 1.x.
        params["use_label_encoder"] = False
    params.update(overrides)
    return XGBClassifier(**params)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: Optional[PathLike] = None,
    feature_names: Optional[Sequence[str]] = None,
    **xgb_params: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit XGBoost, evaluate on the test split and (optionally) save it.

    When ``feature_names`` is provided the model is fitted on a named
    ``pandas.DataFrame`` so that the ONNX/exported booster carries semantic
    feature names into the importance plot.
    """
    import pandas as pd

    if len(np.unique(np.asarray(y_train).ravel())) < 1:
        raise ValueError("y_train is empty — nothing to train on")

    names: Optional[List[str]] = None
    if feature_names is not None:
        if len(feature_names) != np.asarray(X_train).shape[1]:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but the data "
                f"has {np.asarray(X_train).shape[1]} columns"
            )
        names = list(feature_names)
        X_train_fit: Any = pd.DataFrame(np.asarray(X_train), columns=names)
        X_test_fit: Any = pd.DataFrame(np.asarray(X_test), columns=names)
    else:
        X_train_fit = np.asarray(X_train)
        X_test_fit = np.asarray(X_test)

    model = make_xgb_classifier(**xgb_params)
    started = time.perf_counter()
    model.fit(X_train_fit, np.asarray(y_train).ravel())
    elapsed = time.perf_counter() - started
    LOGGER.info("XGBoost fitted in %.2f s", elapsed)

    y_pred = model.predict(X_test_fit)
    y_proba = model.predict_proba(X_test_fit)[:, 1]
    metrics = evaluate_binary(y_test, y_pred, y_proba)
    metrics["train_seconds"] = round(elapsed, 3)

    if model_path is not None:
        target = Path(model_path)
        ensure_dir(target.parent)
        try:
            model.save_model(str(target))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to save XGBoost model to {target}: {exc}") from exc
        LOGGER.info("Saved XGBoost model to %s", target)

    return model, metrics


# --------------------------------------------------------------------------- #
# ONNX export helpers
# --------------------------------------------------------------------------- #
def _convert_xgb_to_onnx(model: Any, n_features: int, feature_names: Optional[Sequence[str]]):
    """Convert an XGBClassifier to ONNX with onnxmltools (skl2onnx backend).

    Important: ``FloatTensorType`` must be the class *onnxmltools* expects
    (``onnxmltools.convert.common.data_types``); using the identical-looking
    class from ``skl2onnx``/``onnxconverter_common`` fails the converter's
    type identity check.
    """
    del feature_names  # names live on the fitted booster; nothing to pass here

    convert_xgboost = None
    float_tensor_type = None
    try:
        from onnxmltools.convert import convert_xgboost as _cx  # type: ignore
        from onnxmltools.convert.common.data_types import (  # type: ignore
            FloatTensorType as _ftt,
        )

        convert_xgboost, float_tensor_type = _cx, _ftt
    except (ImportError, AttributeError):  # pragma: no cover - layout changes
        pass

    if float_tensor_type is None:  # pragma: no cover - defensive fallbacks
        try:
            from onnxconverter_common import FloatTensorType as float_tensor_type
        except ImportError:
            from skl2onnx.common.data_types import (  # type: ignore
                FloatTensorType as float_tensor_type,
            )

    initial_types = [("features", float_tensor_type([None, int(n_features)]))]

    if convert_xgboost is not None:
        try:
            return convert_xgboost(model, initial_types=initial_types)
        except TypeError:  # older/newer signatures differ
            return convert_xgboost(model, name="VocalTraceXGB",
                                   initial_types=initial_types)

    import onnxmltools  # type: ignore  # noqa: PLC0415

    return onnxmltools.convert(model, initial_types=initial_types)


def verify_onnx_parity(
    reference_proba: Sequence[float],
    onnx_path: PathLike,
    probe_input: np.ndarray,
    tolerance: float = ONNX_PARITY_TOLERANCE,
) -> float:
    """Re-run the ONNX graph and assert it matches ``reference_proba``.

    Returns the maximum absolute difference; raises :class:`RuntimeError`
    when it exceeds ``tolerance`` (spec: outputs must match within ``1e-5``).
    """
    reference = np.asarray(reference_proba, dtype=np.float64).ravel()
    session = load_session(onnx_path)
    onnx_proba = session_probabilities(session, probe_input)
    if onnx_proba.size != reference.size:
        raise RuntimeError(
            f"ONNX parity check failed: output has {onnx_proba.size} values, "
            f"source model produced {reference.size}"
        )
    max_diff = float(np.max(np.abs(onnx_proba - reference))) if reference.size else 0.0
    if not np.isfinite(max_diff):
        raise RuntimeError("ONNX parity check failed: non-finite outputs")
    if max_diff > tolerance:
        raise RuntimeError(
            f"ONNX parity check failed for {onnx_path}: max |Δ| = {max_diff:.3e} "
            f"> tolerance {tolerance:.0e}. The exported graph does not "
            "reproduce the source model — do NOT deploy this file."
        )
    LOGGER.info("ONNX parity verified for %s (max |Δ| = %.3e)", Path(onnx_path).name, max_diff)
    return max_diff


def _probe_input_for(model_kind: str, X: np.ndarray) -> np.ndarray:
    """Small deterministic probe batch used for parity checks."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    idx = np.linspace(0, X.shape[0] - 1, num=min(8, X.shape[0])).astype(int)
    probe = X[np.unique(idx)].astype(np.float32)
    if model_kind == "cnn" and probe.ndim == 2:  # (N, C, T) expected already
        raise ValueError("probe_input for the CNN must be (N, channels, frames)")
    return probe


def export_onnx_xgb(
    model: Any,
    n_features: int,
    onnx_path: PathLike,
    probe_input: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Export a trained XGBoost model to ONNX and verify parity (1e-5)."""
    try:
        import onnx  # type: ignore
    except ImportError as exc:  # pragma: no cover - onnx is a hard dep
        raise RuntimeError(
            "The `onnx` package is required for export: pip install onnx"
        ) from exc

    target = Path(onnx_path)
    ensure_dir(target.parent)

    # onnxmltools can only interpret XGBoost feature names matching 'f%d'.
    # The trained booster carries semantic names (mfcc_mean_0, ...), which are
    # ideal for the importance plot — so swap to positional names for the
    # duration of the conversion and restore them right after.
    booster = None
    original_names: List[str] = []
    try:
        booster = model.get_booster()
        original_names = list(booster.feature_names or [])
        if original_names:
            booster.feature_names = [f"f{i}" for i in range(int(n_features))]
    except AttributeError:  # pragma: no cover - not an XGB model
        booster = None

    try:
        try:
            model_onnx = _convert_xgb_to_onnx(model, n_features, feature_names)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ONNX conversion of the XGBoost model failed: "
                f"{exc}. Ensure `onnxmltools` and `skl2onnx` are installed "
                "(`pip install onnxmltools skl2onnx`)."
            ) from exc

        try:
            onnx.save(model_onnx, str(target))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to write ONNX model to {target}: {exc}") from exc

        # Structural sanity check (non-fatal: runtime + parity checks are authoritative).
        try:
            onnx.checker.check_model(str(target))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("onnx.checker reported an issue (non-fatal): %s", exc)

        reference = model.predict_proba(np.asarray(probe_input, dtype=np.float32))[:, 1]
        max_diff = verify_onnx_parity(reference, target, probe_input)
    finally:
        if booster is not None and original_names:
            booster.feature_names = original_names

    session = load_session(target)
    inp = session.get_inputs()[0]
    outp = select_probability_output(session)
    LOGGER.info(
        "ONNX export OK → %s | input %s%s %s | output '%s'",
        target, inp.name, inp.shape, inp.type, outp,
    )
    return {
        "onnx_path": str(target),
        "onnx_max_abs_diff": max_diff,
        "onnx_input_shape": [str(s) for s in inp.shape],
    }


# --------------------------------------------------------------------------- #
# 1-D CNN (optional torch)
# --------------------------------------------------------------------------- #
def _require_torch():
    """Import torch or raise a descriptive, actionable error."""
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed but --model cnn requires it. "
            "Install it with `pip install torch` (CPU build is enough), or "
            "train the default XGBoost model with `--model xgb`."
        ) from exc
    return torch


def build_cnn(in_channels: int, n_classes: int = 2):
    """Create the spec CNN: 3 Conv1d blocks (32→64→128, k=3, BN, ReLU, Pool)
    → GlobalAvgPool → Linear(n_classes).

    The first layer also applies **per-channel standardisation** using
    ``ch_mean``/``ch_inv_std`` buffers (populated from the training split by
    :func:`train_cnn`).  Raw channels live on incomparable scales — MFCC c0
    reaches ±600 while the phase-discontinuity channel sits near 0.4 — and a
    shared Conv1d would otherwise be dominated by whichever channel shouts
    loudest.  Because the statistics are buffers, they travel inside the
    ``state_dict`` and inside the exported ONNX graph, so inference (and
    export parity) needs no extra plumbing.
    """
    torch = _require_torch()

    class VocalTraceCNN(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "ch_mean", torch.zeros(1, in_channels, 1)
            )
            self.register_buffer(
                "ch_inv_std", torch.ones(1, in_channels, 1)
            )
            self.features = torch.nn.Sequential(
                self._block(in_channels, 32),
                self._block(32, 64),
                self._block(64, 128),
                torch.nn.AdaptiveAvgPool1d(1),
            )
            self.head = torch.nn.Linear(128, n_classes)

        @staticmethod
        def _block(ch_in: int, ch_out: int) -> torch.nn.Module:
            return torch.nn.Sequential(
                torch.nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(ch_out),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool1d(kernel_size=2, stride=2),
            )

        def set_channel_normalization(self, mean, std) -> None:
            """Populate standardisation buffers from training statistics."""
            mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1)
            std_t = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1, 1)
            if mean_t.shape[1] != in_channels or std_t.shape[1] != in_channels:
                raise ValueError(
                    f"normalisation vectors must have {in_channels} channels, "
                    f"got mean {tuple(mean_t.shape)} std {tuple(std_t.shape)}"
                )
            self.ch_mean.copy_(mean_t)
            self.ch_inv_std.copy_(1.0 / torch.clamp(std_t, min=1e-6))

        def forward(self, x):  # x: (batch, channels, time)
            x = (x - self.ch_mean) * self.ch_inv_std
            x = self.features(x)
            x = x.flatten(1)
            return self.head(x)

    model = VocalTraceCNN()
    model.__class__.__name__ = "VocalTraceCNN"  # readability in logs / ONNX
    return model


def _cnn_probabilities(logits):
    """Relative P(fake) derived from the two class logits.

    With one-hot BCE training, ``sigmoid(logit)`` gives per-class
    activations; normalising them yields a proper probability in (0, 1).
    """
    torch = _require_torch()
    act = torch.sigmoid(logits)
    return act[:, 1] / (act[:, 0] + act[:, 1] + 1e-9)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_path: Optional[PathLike] = None,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    seed: int = SEED,
    device: str = "cpu",
) -> Tuple[Any, Dict[str, Any]]:
    """Train the 1-D CNN on ``(N, channels, frames)`` sequences."""
    torch = _require_torch()
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    set_global_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    if X_train.ndim != 3:
        raise ValueError(
            f"CNN input must be (N, channels, frames), got shape {X_train.shape}. "
            "Build the dataset with vector='sequence'."
        )
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            f"Train {X_train.shape[1:]} and test {X_test.shape[1:]} shapes differ"
        )

    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    y_test = np.asarray(y_test, dtype=np.int64).ravel()

    dev = torch.device(device)
    model = build_cnn(in_channels=X_train.shape[1]).to(dev)

    # Equalise channel scales using *training-split* statistics only (no
    # leakage); buffers travel with state_dict and the exported ONNX graph.
    channel_mean = X_train.mean(axis=(0, 2))
    channel_std = X_train.std(axis=(0, 2))
    model.set_channel_normalization(channel_mean, channel_std)

    # One-hot targets for BCEWithLogitsLoss over the 2-way head (spec).
    targets = torch.nn.functional.one_hot(
        torch.from_numpy(y_train), num_classes=2
    ).float()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), targets
    )
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True,
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            LOGGER.info(
                "CNN epoch %2d/%d  loss=%.4f", epoch, epochs,
                total_loss / max(n_batches, 1),
            )
    elapsed = time.perf_counter() - started
    LOGGER.info("CNN trained in %.2f s", elapsed)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(dev))
        proba = _cnn_probabilities(logits).cpu().numpy()
    metrics = evaluate_binary(y_test, (proba >= 0.5).astype(int), proba)
    metrics["train_seconds"] = round(elapsed, 3)
    metrics["epochs"] = int(epochs)

    # Honest small-data warning: a Conv1d stack has far more parameters than
    # the flat XGBoost path; below a few hundred windows it memorises the
    # training split (observed train AUC 1.0 / test AUC 0.5).  Say so loudly
    # instead of silently shipping an overfit model.
    if len(y_train) < 200:
        LOGGER.warning(
            "CNN trained on only %d windows — deep models typically need "
            "several hundred to generalise. If test metrics look random, "
            "prefer `--model xgb` or enlarge the corpus.",
            len(y_train),
        )

    if model_path is not None:
        target = Path(model_path)
        ensure_dir(target.parent)
        try:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_channels": int(X_train.shape[1]),
                    "n_classes": 2,
                    "epochs": int(epochs),
                    "seed": int(seed),
                },
                str(target),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to save CNN weights to {target}: {exc}") from exc
        LOGGER.info("Saved CNN weights to %s", target)

    return model, metrics


def export_onnx_cnn(
    model: Any,
    probe_input: np.ndarray,
    onnx_path: PathLike,
    opset: int = 17,
) -> Dict[str, Any]:
    """Export the CNN to ONNX (dynamic batch/time axes) and verify parity."""
    torch = _require_torch()
    try:
        import onnx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The `onnx` package is required: pip install onnx") from exc

    target = Path(onnx_path)
    ensure_dir(target.parent)

    class _ExportWrapper(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, core: Any) -> None:
            super().__init__()
            self.core = core

        def forward(self, x):
            act = torch.sigmoid(self.core(x))
            return act / (act.sum(dim=1, keepdim=True) + 1e-9)

    core = model.eval()
    wrapper = _ExportWrapper(core).eval()
    probe = np.asarray(probe_input, dtype=np.float32)
    if probe.ndim == 2:
        probe = probe[None, :]
    dummy = torch.from_numpy(probe)

    export_kwargs: Dict[str, Any] = dict(
        input_names=["features"],
        output_names=["probabilities"],
        dynamic_axes={
            "features": {0: "batch", 2: "time"},
            "probabilities": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    if target.exists():
        target.unlink()

    # torch >= ~2.9 defaults to the dynamo exporter, which needs the optional
    # `onnxscript` package.  Prefer the legacy TorchScript exporter (pure
    # `onnx` dependency) and fall back to dynamo when legacy is unavailable.
    failures: List[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # legacy export emits FutureWarnings
            torch.onnx.export(wrapper, dummy, str(target),
                              dynamo=False, **export_kwargs)
    except TypeError as exc:
        if "dynamo" in str(exc):  # torch too old to know the `dynamo` flag
            try:
                torch.onnx.export(wrapper, dummy, str(target), **export_kwargs)
            except Exception as retry_exc:  # noqa: BLE001
                failures.append(f"legacy exporter: {retry_exc}")
        else:
            failures.append(f"legacy exporter: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"legacy exporter: {exc}")
        try:
            torch.onnx.export(wrapper, dummy, str(target),
                              dynamo=True, **export_kwargs)
        except Exception as dynamo_exc:  # noqa: BLE001
            failures.append(f"dynamo exporter: {dynamo_exc}")

    if not target.exists() or target.stat().st_size == 0:
        details = "; ".join(failures) or "no exporter produced a file"
        raise RuntimeError(
            "torch.onnx.export failed: "
            f"{details}. Install the optional exporter (`pip install "
            "onnxscript`), keep torch >= 2.4, or export the default model "
            "with `--model xgb`."
        )

    try:
        onnx.checker.check_model(str(target))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("onnx.checker reported an issue (non-fatal): %s", exc)

    with torch.no_grad():
        reference = _probability_from_output(
            _ExportWrapper(core)(torch.from_numpy(probe)).cpu().numpy()
        )
    max_diff = verify_onnx_parity(reference, target, probe)

    session = load_session(target)
    inp = session.get_inputs()[0]
    LOGGER.info("CNN ONNX export OK → %s | input %s", target, inp.shape)
    return {
        "onnx_path": str(target),
        "onnx_max_abs_diff": max_diff,
        "onnx_input_shape": [str(s) for s in inp.shape],
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_training(
    real_dir: PathLike,
    fake_dir: PathLike,
    model_type: str = "xgb",
    export_onnx: bool = True,
    test_size: float = 0.2,
    epochs: int = 30,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    learning_rate: Optional[float] = None,
    models_dir: PathLike = DEFAULT_MODEL_DIR,
    outputs_dir: PathLike = DEFAULT_OUTPUT_DIR,
    seed: int = SEED,
    make_plots: bool = True,
    **xgb_params: Any,
) -> Dict[str, Any]:
    """Full training pipeline: build → split → fit → evaluate → export.

    Returns a JSON-serialisable summary (metrics, model paths, ONNX parity).
    Raises descriptive exceptions for every recoverable-looking failure.
    """
    setup_logging()
    if model_type not in ("xgb", "cnn"):
        raise ValueError(f"model_type must be 'xgb' or 'cnn', got {model_type!r}")

    set_global_seed(seed)
    models_path = ensure_dir(Path(models_dir))
    outputs_path = ensure_dir(Path(outputs_dir))

    LOGGER.info(
        "Training VocalTrace (%s) | real=%s fake=%s test_size=%.2f seed=%d",
        model_type, real_dir, fake_dir, test_size, seed,
    )

    builder = DatasetBuilder(
        window_seconds=window_seconds, hop_seconds=hop_seconds, seed=seed
    )
    vector = "sequence" if model_type == "cnn" else "flat"
    X, y, _paths = builder.build_dataset(real_dir, fake_dir, vector=vector)

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError(
            "The dataset must contain both classes (0=real and 1=fake). "
            f"Found only class(es) {classes.tolist()}. Generate fakes first: "
            "`python scripts/make_samples.py`."
        )

    X_train, X_test, y_train, y_test = builder.split_train_test(
        X, y, test_size=test_size, random_state=seed
    )
    LOGGER.info(
        "Split: %d train / %d test samples", len(y_train), len(y_test)
    )

    summary: Dict[str, Any] = {
        "model_type": model_type,
        "seed": int(seed),
        "n_samples": int(X.shape[0]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
    }

    if model_type == "xgb":
        if "torch" in sys.modules:
            # Not fatal everywhere, but worth flagging: some platforms crash
            # when torch's bundled libomp and xgboost's OpenMP runtime coexist.
            LOGGER.warning(
                "PyTorch is already imported in this process while training "
                "XGBoost. If this machine crashes, run the xgb and cnn paths "
                "in separate processes (the CLI always does)."
            )
        feature_names = FeatureExtractor().flat_feature_names()
        if X.shape[1] != len(feature_names):  # custom extractor config
            LOGGER.warning(
                "Feature dim %d != default %d; training without feature names",
                X.shape[1], len(feature_names),
            )
            feature_names = None

        if learning_rate is not None:
            xgb_params.setdefault("learning_rate", learning_rate)

        model, metrics = train_xgb(
            X_train, y_train, X_test, y_test,
            model_path=models_path / XGB_MODEL_FILENAME,
            feature_names=feature_names,
            **xgb_params,
        )
        model_file = models_path / XGB_MODEL_FILENAME

        onnx_info: Dict[str, Any] = {}
        if export_onnx:
            probe = np.asarray(X_test if len(X_test) else X_train, dtype=np.float32)
            onnx_info = export_onnx_xgb(
                model,
                n_features=int(X.shape[1]),
                onnx_path=models_path / ONNX_MODEL_FILENAME,
                probe_input=probe,
                feature_names=feature_names,
            )
    else:
        model, metrics = train_cnn(
            X_train, y_train, X_test, y_test,
            model_path=models_path / CNN_MODEL_FILENAME,
            epochs=epochs,
            learning_rate=(learning_rate if learning_rate is not None else 1e-3),
            seed=seed,
        )
        model_file = models_path / CNN_MODEL_FILENAME

        onnx_info = {}
        if export_onnx:
            probe = np.asarray(X_test if len(X_test) else X_train, dtype=np.float32)[:4]
            onnx_info = export_onnx_cnn(
                model, probe, models_path / ONNX_MODEL_FILENAME
            )

    summary["metrics"] = metrics
    summary["model_path"] = str(model_file)
    summary.update(onnx_info)

    # Persist machine-readable metrics for CI / dashboards.
    metrics_path = outputs_path / "metrics.json"
    try:
        metrics_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        summary["metrics_path"] = str(metrics_path)
    except OSError as exc:  # pragma: no cover - permissions
        LOGGER.warning("Could not write metrics to %s: %s", metrics_path, exc)

    if make_plots:
        _generate_training_plots(
            model, model_type, X_test, y_test, metrics,
            outputs_path, model_file=model_file,
            feature_names=(feature_names if model_type == "xgb" else None),
        )

    LOGGER.info("Training finished: %s", format_metrics(metrics, title="test metrics"))
    return summary


def _generate_training_plots(
    model: Any,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Dict[str, Any],
    outputs_path: Path,
    model_file: Optional[Path] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> None:
    """Render the standard report plots; never let plotting kill a run."""
    from . import report

    y_true = np.asarray(y_test).ravel()
    y_pred: Optional[np.ndarray] = None
    y_proba: Optional[np.ndarray] = None
    try:
        if model_type == "xgb":
            y_pred = np.asarray(model.predict(X_test)).ravel()
            y_proba = np.asarray(model.predict_proba(X_test)[:, 1]).ravel()
        else:
            torch = _require_torch()
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(X_test, dtype=np.float32)))
                y_proba = _cnn_probabilities(logits).cpu().numpy().ravel()
            y_pred = (y_proba >= 0.5).astype(int)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not compute test predictions for plots: %s", exc)
        return

    # Confusion matrix + ROC curve from the test split.
    if y_pred is not None and y_true.size:
        try:
            report.plot_confusion_matrix(
                y_true, y_pred, outputs_path / "confusion_matrix.png"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("plot_confusion_matrix failed: %s", exc)
    if y_proba is not None and np.unique(y_true).size > 1:
        try:
            report.plot_roc_curve(
                y_true, y_proba, outputs_path / "roc_curve.png"
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("plot_roc_curve failed: %s", exc)

    # Feature importance (XGBoost only) — read from the saved model file.
    if model_type == "xgb" and model_file is not None and model_file.exists():
        try:
            report.plot_feature_importance(
                model_file, outputs_path / "feature_importance.png", top_n=20
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("plot_feature_importance failed: %s", exc)
