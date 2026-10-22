"""Shared pytest fixtures for the VocalTrace test-suite.

* ``toy_corpus``   — a 5-file toy dataset (3 synthetic "real" + 2
  Griffin-Lim "fake" WAVs) generated inside a temp directory.
* ``trained_artifacts`` — a small XGBoost model trained on that corpus and
  exported to ONNX (parity-checked by the training code itself).

Everything is created on the fly, so the tests never depend on the shipped
``data/`` corpus and run identically on Windows/macOS/Linux.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 16000


def synth_voice(
    seconds: float = 2.4,
    seed: int = 42,
    sr: int = SR,
) -> np.ndarray:
    """Deterministic speech-like waveform: harmonic stack with formant-shaped
    spectrum, a moving pitch contour, amplitude modulation and breath noise.

    Being genuinely phase-coherent (all harmonics share an integrated pitch
    phase), it behaves like "real" speech for the phase-discontinuity
    detector, while a Griffin-Lim round-trip scrambles that phase — giving the
    toy classifier a real, physical signal to learn.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 115.0 + 35.0 * np.sin(2.0 * np.pi * 0.7 * t + seed)  # moving pitch
    pitch_phase = 2.0 * np.pi * np.cumsum(f0) / sr

    def formant_gain(freq: np.ndarray) -> np.ndarray:
        # Three broad formants (F1–F3) + gentle overall tilt ≈ vocal tract.
        g = np.full_like(freq, 0.05)
        for centre, width, height in ((700.0, 300.0, 1.0),
                                      (1300.0, 450.0, 0.7),
                                      (2600.0, 800.0, 0.35)):
            g = g + height * np.exp(-0.5 * ((freq - centre) / width) ** 2)
        return g

    y = np.zeros(n)
    harmonic = 1
    while harmonic * float(np.mean(f0)) < 7600.0:
        freq = harmonic * f0
        amplitude = formant_gain(harmonic * np.mean(f0)) / np.sqrt(harmonic)
        am = 1.0 + 0.15 * np.sin(pitch_phase)  # glottal-style modulation
        y += amplitude * am * np.sin(harmonic * pitch_phase)
        harmonic += 1

    y += 0.015 * rng.normal(size=n)  # breath noise
    # Syllabic loudness contour so silence-trimming/windowing looks realistic.
    envelope = 0.55 + 0.45 * np.abs(np.sin(2.0 * np.pi * 1.6 * t + 0.4))
    y *= envelope
    y = 0.35 * y / (np.max(np.abs(y)) + 1e-12)
    return y.astype(np.float32)


@pytest.fixture(scope="session")
def toy_corpus(tmp_path_factory) -> dict:
    """5 toy WAVs: 3 real (synthetic voice) + 2 fake (Griffin-Lim roundtrip).

    The clips are 6 s long on purpose: at the default 2 s window / 1 s hop
    that yields ~5 training windows per file (25 total), which is the minimum
    at which the split search finds the physical discriminative features
    (roll-off / phase statistics) instead of memorising 7 windows.
    """
    import soundfile as sf

    from vocaltrace.dataset import DatasetBuilder

    root = tmp_path_factory.mktemp("toy_corpus")
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()

    real_files = []
    for idx in range(1, 4):
        audio = synth_voice(seconds=6.0, seed=40 + idx)
        path = real_dir / f"real{idx}.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        real_files.append(path)

    builder = DatasetBuilder(sr=SR, seed=42)
    # Fakes come from the FIRST TWO real clips → 3 + 2 = 5 files total.
    fake_files = builder.generate_fake_audio(real_files[:2], fake_dir)

    assert len(real_files) == 3 and len(fake_files) == 2
    return {
        "root": root,
        "real_dir": real_dir,
        "fake_dir": fake_dir,
        "real_files": real_files,
        "fake_files": [Path(p) for p in fake_files],
    }


@pytest.fixture(scope="session")
def trained_artifacts(toy_corpus, tmp_path_factory) -> dict:
    """Train a compact XGBoost model on the toy corpus and export ONNX."""
    from vocaltrace.train import run_training

    out_root = tmp_path_factory.mktemp("artifacts")
    models_dir = out_root / "models"
    outputs_dir = out_root / "outputs"

    summary = run_training(
        real_dir=toy_corpus["real_dir"],
        fake_dir=toy_corpus["fake_dir"],
        model_type="xgb",
        export_onnx=True,
        test_size=0.3,
        n_estimators=25,
        max_depth=3,
        # The toy corpus has only 7 training windows; XGBoost's default
        # min_child_weight=1 would forbid every split (the Hessian sum of a
        # leaf at p=0.5 needs >= 4 samples) and the model would stay constant.
        min_child_weight=0,
        models_dir=models_dir,
        outputs_dir=outputs_dir,
        seed=42,
        make_plots=True,
    )
    summary["models_dir"] = models_dir
    summary["outputs_dir"] = outputs_dir
    # run_training returns JSON-friendly strings; tests and the detector
    # fixture work with Path objects.
    summary["model_path"] = Path(summary["model_path"])
    summary["onnx_path"] = Path(summary["onnx_path"])
    summary["metrics_path"] = Path(summary["metrics_path"])
    return summary


@pytest.fixture(scope="session")
def detector(trained_artifacts):
    """A warmed-up detector around the toy-trained ONNX model."""
    from vocaltrace.detector import VocalTraceDetector

    return VocalTraceDetector(onnx_path=trained_artifacts["onnx_path"])
