"""Module 6 — reporting and visualisation for VocalTrace.

All figures are written to PNG via the headless ``Agg`` backend —
``plt.show()`` is never called, so the module works on servers, CI and
desktops alike.

Functions
---------
* :func:`plot_waveform_and_spectrogram` — 3-panel view (waveform, MFCC
  heatmap, phase spectrogram).
* :func:`plot_feature_importance` — top-20 XGBoost gain importances.
* :func:`plot_confusion_matrix` — row-normalised heatmap.
* :func:`plot_roc_curve` — ROC with the AUC annotated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

try:
    import librosa
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "librosa is required for VocalTrace. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from .features import FeatureExtractor
from .utils import ensure_dir, require_file

LOGGER = logging.getLogger("vocaltrace.report")

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save_figure(fig: "plt.Figure", output_png: PathLike) -> Path:
    target = Path(output_png)
    if target.suffix.lower() != ".png":
        raise ValueError(
            f"output_png must end with .png, got {target.name!r}"
        )
    ensure_dir(target.parent)
    try:
        fig.savefig(target, dpi=150, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to save figure to {target}: {exc}") from exc
    finally:
        plt.close(fig)
    LOGGER.info("Wrote plot %s", target)
    return target


def _validate_pairs(
    y_true: Sequence[int], y_pred: Sequence[float], what: str = "inputs"
) -> tuple:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0:
        raise ValueError(f"{what} are empty — nothing to plot")
    if yt.size != yp.size:
        raise ValueError(
            f"{what} must have equal length, got {yt.size} vs {yp.size}"
        )
    return yt, yp


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_waveform_and_spectrogram(
    wav_path: PathLike,
    output_png: PathLike,
    sr: int = 16000,
    extractor: Optional[FeatureExtractor] = None,
) -> Path:
    """Render waveform + MFCC heatmap + phase spectrogram (3 panels)."""
    path = require_file(wav_path, what="Audio file")
    extractor = extractor or FeatureExtractor(sr=sr)

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    y = extractor.validate_audio(y, name=f"audio from {path.name}")
    feats = extractor.extract_all(y, sr=sr)

    times = np.arange(y.size) / float(sr)
    mfcc = feats["mfcc"]
    phase = feats["phase_spec"]
    n_fft = extractor.n_fft
    hop = extractor.hop_length
    spec_times = np.arange(phase.shape[1]) * hop / float(sr)
    freqs = np.linspace(0.0, sr / 2.0, phase.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(times, y, linewidth=0.6, color="#1f77b4")
    axes[0].set_title(f"Waveform — {path.name}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(times[0], times[-1])
    axes[0].grid(alpha=0.3)

    im1 = axes[1].imshow(
        mfcc, aspect="auto", origin="lower", cmap="magma",
        extent=[spec_times[0], spec_times[-1], 0, mfcc.shape[0]],
        interpolation="nearest",
    )
    axes[1].set_title("MFCC + delta + delta-delta (120 coefficients)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Coefficient")
    fig.colorbar(im1, ax=axes[1], pad=0.01, label="Amplitude")

    im2 = axes[2].imshow(
        phase, aspect="auto", origin="lower", cmap="twilight",
        vmin=-np.pi, vmax=np.pi,
        extent=[spec_times[0], spec_times[-1], freqs[0], freqs[-1]],
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Phase spectrogram (STFT n_fft={n_fft}, hop={hop}) — abrupt jumps "
        "mark vocoder artefacts"
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    fig.colorbar(im2, ax=axes[2], pad=0.01, label="Phase (rad, -π…π)")

    fig.tight_layout()
    return _save_figure(fig, output_png)


def plot_feature_importance(
    model_path: PathLike,
    output_png: PathLike,
    top_n: int = 20,
) -> Path:
    """Bar chart of the top-N XGBoost feature importances (gain)."""
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    path = require_file(model_path, what="XGBoost model")

    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise RuntimeError("xgboost is required: pip install xgboost") from exc

    booster = xgb.Booster()
    try:
        booster.load_model(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load XGBoost model {path}: {exc}. This plot only "
            "supports the XGBoost export (models/vocaltrace_xgb.json)."
        ) from exc

    scores = booster.get_score(importance_type="gain")
    if not scores or all(v == 0 for v in scores.values()):
        scores = booster.get_score(importance_type="weight")
    if not scores:
        raise ValueError(
            f"No feature-importance scores found in {path}. The model may "
            "have been trained without feature names or never split."
        )

    # f0/f1/... keys → fall back to positional labels.
    items = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top_n]
    labels = [k for k, _ in items]
    values = [float(v) for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#2ca02c", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {len(labels)} VocalTrace feature importances")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_png)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_png: PathLike,
    class_names: Sequence[str] = ("REAL", "FAKE"),
) -> Path:
    """Row-normalised confusion-matrix heatmap (percent of each true class)."""
    from sklearn.metrics import confusion_matrix

    yt, yp = _validate_pairs(y_true, y_pred, what="y_true/y_pred")
    if np.unique(yt).size > len(class_names):
        raise ValueError(
            f"Found {np.unique(yt).size} classes but only "
            f"{len(class_names)} labels: {list(class_names)}"
        )

    bad = np.isin(np.asarray(yt, dtype=np.int64), [0, 1], invert=True)
    if bad.any():
        raise ValueError(
            "y_true must contain only binary labels 0/1, found "
            f"{np.unique(np.asarray(yt)[bad]).tolist()}"
        )

    cm = confusion_matrix(yt, yp, labels=[0, 1])
    normalized = np.zeros_like(cm, dtype=np.float64)
    for i, row in enumerate(cm):
        total = row.sum()
        normalized[i] = row / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1], labels=list(class_names)[:2])
    ax.set_yticks([0, 1], labels=list(class_names)[:2])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("VocalTrace confusion matrix (row-normalised)")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            raw = cm[i, j]
            ax.text(
                j, i, f"{normalized[i, j] * 100:.1f}%\n(n={raw})",
                ha="center", va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")
    fig.tight_layout()
    return _save_figure(fig, output_png)


def plot_roc_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    output_png: PathLike,
) -> Path:
    """ROC curve with the AUC annotated in the legend."""
    from sklearn.metrics import roc_auc_score, roc_curve

    yt, yp = _validate_pairs(y_true, y_proba, what="y_true/y_proba")
    if np.unique(yt).size < 2:
        raise ValueError(
            "ROC curve needs both classes in y_true "
            f"(got {np.unique(yt).tolist()})"
        )
    yp = np.asarray(yp, dtype=np.float64)
    if not np.isfinite(yp).all():
        raise ValueError("y_proba contains NaN/Inf values")

    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, color="#d62728", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey",
            label="chance (AUC = 0.500)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("VocalTrace ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_png)
