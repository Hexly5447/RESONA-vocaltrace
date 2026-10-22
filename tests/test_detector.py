"""Unit tests for Module 4 — inference engine + ONNX export parity.

Covers the spec requirements:
* ONNX export parity within 1e-5 (source model vs onnxruntime),
* the detector output schema,
* sliding-window streaming behaviour,
* graceful microphone degradation.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import SR, synth_voice

# --------------------------------------------------------------------------- #
# ONNX parity
# --------------------------------------------------------------------------- #
def test_onnx_parity_within_tolerance(trained_artifacts):
    """The exported graph must reproduce the source model to 1e-5 on fresh
    random inputs (parity is checked during export; assert it here too)."""
    import xgboost as xgb

    from vocaltrace.train import ONNX_PARITY_TOLERANCE, verify_onnx_parity

    max_diff = trained_artifacts.get("onnx_max_abs_diff")
    assert max_diff is not None, "export must report its parity error"
    assert max_diff <= ONNX_PARITY_TOLERANCE

    # Independent re-check on *unseen* random feature vectors.
    booster = xgb.Booster()
    booster.load_model(trained_artifacts["model_path"])
    n_features = int(trained_artifacts["onnx_input_shape"][1])
    rng = np.random.default_rng(0)
    probe = rng.normal(size=(16, n_features)).astype(np.float32)
    matrix = xgb.DMatrix(probe, feature_names=booster.feature_names)
    reference = booster.predict(matrix)
    diff = verify_onnx_parity(
        reference, trained_artifacts["onnx_path"], probe
    )
    assert diff <= ONNX_PARITY_TOLERANCE


def test_onnx_file_exists_and_is_loadable(trained_artifacts):
    from vocaltrace.detector import load_session

    path = trained_artifacts["onnx_path"]
    assert path.exists()
    session = load_session(path)
    assert session.get_inputs()[0].shape[1] == 1905
    output_names = [o.name for o in session.get_outputs()]
    assert any("prob" in name.lower() for name in output_names)


# --------------------------------------------------------------------------- #
# output schema
# --------------------------------------------------------------------------- #
def test_analyze_file_schema(detector, toy_corpus):
    result = detector.analyze_file(toy_corpus["real_files"][0])

    required = {
        "file", "verdict", "confidence", "fake_probability",
        "phase_discontinuity_score", "hf_rolloff_score", "latency_ms",
        "threshold",
    }
    assert required <= set(result), f"missing keys: {required - set(result)}"

    assert result["verdict"] in ("REAL", "FAKE")
    assert isinstance(result["file"], str) and result["file"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert 0.0 <= result["phase_discontinuity_score"] <= 1.0
    assert 0.0 <= result["hf_rolloff_score"] <= 1.0
    assert result["latency_ms"] > 0.0
    assert result["threshold"] == detector.threshold
    # Verdict and confidence must agree with fake_probability/threshold.
    expected = "FAKE" if result["fake_probability"] >= result["threshold"] else "REAL"
    assert result["verdict"] == expected
    assert result["confidence"] == pytest.approx(
        max(result["fake_probability"], 1.0 - result["fake_probability"])
    )


def test_analyze_file_detects_generated_fake(detector, toy_corpus):
    fake = detector.analyze_file(toy_corpus["fake_files"][0])
    real = detector.analyze_file(toy_corpus["real_files"][0])
    assert fake["verdict"] == "FAKE", (
        f"toy Griffin-Lim fake should be flagged, got {fake}"
    )
    assert fake["fake_probability"] > real["fake_probability"]


def test_analyze_file_steady_state_latency(detector, toy_corpus):
    """After warm-up, analysing a ~2.4 s window must stay well under 150 ms."""
    detector.analyze_file(toy_corpus["real_files"][1])  # ensure warm
    result = detector.analyze_file(toy_corpus["real_files"][1])
    assert result["latency_ms"] <= 150.0, result


def test_analyze_file_error_handling(detector, tmp_path, toy_corpus):
    with pytest.raises(FileNotFoundError):
        detector.analyze_file(tmp_path / "does_not_exist.wav")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("not audio")
    with pytest.raises(ValueError, match="format"):
        detector.analyze_file(bad_ext)

    with pytest.raises(ValueError, match="not a directory|regular file"):
        detector.analyze_file(tmp_path)


def test_feature_dimension_mismatch_is_clear(trained_artifacts, toy_corpus):
    """A detector built with a different extractor config must fail loudly."""
    from vocaltrace.detector import VocalTraceDetector
    from vocaltrace.features import FeatureExtractor

    wrong = VocalTraceDetector(
        onnx_path=trained_artifacts["onnx_path"],
        extractor=FeatureExtractor(n_mfcc=20),  # 3*20+513+1 != 634 channels
        warmup=False,
    )
    with pytest.raises(ValueError, match="mismatch"):
        wrong.analyze_file(toy_corpus["real_files"][0])


def test_invalid_threshold_and_model_path(trained_artifacts):
    from vocaltrace.detector import VocalTraceDetector

    with pytest.raises(ValueError, match="threshold"):
        VocalTraceDetector(trained_artifacts["onnx_path"], threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        VocalTraceDetector(trained_artifacts["onnx_path"], threshold=0.0)
    with pytest.raises(FileNotFoundError):
        VocalTraceDetector("/nonexistent/model.onnx")


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def test_analyze_stream_sliding_window(detector):
    chunk = synth_voice(seconds=6.0, seed=11)
    result = detector.analyze_stream(
        chunk, sr=SR, window_seconds=2.0, hop_seconds=0.5
    )

    # 6 s with 2 s windows / 0.5 s hop → 9 windows (incl. the tail).
    assert result["num_windows"] == 9
    scores = result["window_fake_probabilities"]
    assert len(scores) == 9
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["verdict"] in ("REAL", "FAKE")
    assert result["window_seconds"] == 2.0
    assert result["hop_seconds"] == 0.5
    # mean + max aggregation semantics.
    assert result["aggregate_fake_probability"] == pytest.approx(
        max(scores)
    )
    assert result["max_fake_probability"] == pytest.approx(max(scores))
    assert result["mean_fake_probability"] == pytest.approx(
        float(np.mean(scores))
    )
    assert result["windows_flagged"] == sum(
        1 for s in scores if s >= detector.threshold
    )
    assert result["latency_ms"] > 0.0


def test_analyze_stream_mean_aggregation(detector):
    chunk = synth_voice(seconds=5.0, seed=12)
    result = detector.analyze_stream(chunk, aggregation="mean")
    scores = result["window_fake_probabilities"]
    assert result["aggregate_fake_probability"] == pytest.approx(
        float(np.mean(scores))
    )
    assert result["aggregation"] == "mean"


def test_analyze_stream_short_chunk(detector):
    """Chunks shorter than the window are analysed as a single window."""
    short = synth_voice(seconds=1.0, seed=13)
    result = detector.analyze_stream(short)
    assert result["num_windows"] == 1


def test_analyze_stream_validation(detector):
    audio = synth_voice(seconds=3.0, seed=14)
    with pytest.raises(ValueError, match="aggregation"):
        detector.analyze_stream(audio, aggregation="median")
    with pytest.raises(ValueError, match="window_seconds"):
        detector.analyze_stream(audio, window_seconds=0)
    with pytest.raises(ValueError, match="hop_seconds"):
        detector.analyze_stream(audio, hop_seconds=-1)
    with pytest.raises(ValueError, match="sr"):
        detector.analyze_stream(audio, sr=0)
    with pytest.raises(ValueError, match="empty"):
        detector.analyze_stream(np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="NaN"):
        detector.analyze_stream(np.array([np.nan] + [0.0] * 100))


# --------------------------------------------------------------------------- #
# microphone (graceful degradation)
# --------------------------------------------------------------------------- #
def test_analyze_microphone_without_sounddevice(detector, monkeypatch):
    """Missing optional dependency → warning + None, never an exception."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # forces ImportError
    assert detector.analyze_microphone(duration_seconds=1) is None


def test_analyze_microphone_rejects_bad_duration(detector):
    with pytest.raises(ValueError, match="duration_seconds"):
        detector.analyze_microphone(duration_seconds=0)
