"""Unit tests for Module 1 — feature extraction (shapes + validation)."""

from __future__ import annotations

import numpy as np
import pytest

from vocaltrace.features import FeatureExtractor

from conftest import SR, synth_voice  # noqa: E402 - shared toy signal

N_MFCC = 40
N_FFT = 1024
HOP = 256


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor(sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    return synth_voice(seconds=1.5, seed=7)


@pytest.fixture(scope="module")
def features(extractor, audio):
    return extractor.extract_all(audio, sr=SR)


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def test_mfcc_shape(extractor, audio):
    mfcc = extractor.extract_mfcc(audio, sr=SR, n_mfcc=N_MFCC)
    assert mfcc.ndim == 2
    assert mfcc.shape[0] == 3 * N_MFCC, "MFCC must stack delta + delta-delta"
    assert mfcc.shape[1] > 0
    assert np.isfinite(mfcc).all()


def test_mfcc_custom_n_mfcc(extractor, audio):
    mfcc = extractor.extract_mfcc(audio, sr=SR, n_mfcc=20, n_fft=N_FFT,
                                  hop_length=HOP)
    assert mfcc.shape[0] == 60


def test_phase_spectrogram_shapes(extractor, audio):
    out = extractor.extract_phase_spectrogram(
        audio, sr=SR, n_fft=N_FFT, hop_length=HOP
    )
    assert set(out) >= {"phase_spec", "phase_diff", "phase_d2", "phase_disc"}
    frames = out["phase_spec"].shape[1]  # (freq_bins, time_frames)
    assert out["phase_spec"].shape == (N_FFT // 2 + 1, frames)
    assert out["phase_diff"].shape == out["phase_spec"].shape
    assert out["phase_d2"].shape == out["phase_spec"].shape
    assert out["phase_disc"].shape == (frames,)
    # Scores are bounded and finite by construction.
    assert out["phase_disc"].dtype == np.float32
    assert np.isfinite(out["phase_disc"]).all()
    assert (out["phase_disc"] >= 0.0).all() and (out["phase_disc"] <= 1.0).all()
    # Raw phase lives in [-pi, pi].
    assert (out["phase_spec"] >= -np.pi - 1e-6).all()
    assert (out["phase_spec"] <= np.pi + 1e-6).all()


def test_hf_rolloff_shape_and_range(extractor, audio):
    hf = extractor.extract_hf_rolloff(audio, sr=SR)
    assert hf.shape == (3,)
    assert np.isfinite(hf).all()
    roll85, roll95, ratio = hf
    assert 0.0 <= roll85 <= roll95 <= SR / 2
    assert 0.0 <= ratio <= 1.0


def test_extract_all_alignment(extractor, features):
    frames = features["mfcc"].shape[1]
    assert features["phase_spec"].shape[1] == frames, "time axes must align"
    assert features["phase_disc"].shape == (frames,)
    assert features["mfcc"].shape[0] == 3 * N_MFCC
    assert features["phase_spec"].shape[0] == N_FFT // 2 + 1
    assert features["hf_features"].shape == (3,)
    assert features["sr"] == SR
    for key in ("mfcc", "phase_spec", "phase_disc", "hf_features"):
        assert features[key].dtype == np.float32
        assert np.isfinite(features[key]).all(), key


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_to_flat_vector(extractor, features):
    vector = extractor.to_flat_vector(features)
    assert vector.ndim == 1
    assert vector.size == extractor.flat_dim
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_flat_feature_names_match_vector(extractor):
    names = extractor.flat_feature_names()
    assert len(names) == extractor.flat_dim
    assert names[0] == "mfcc_mean_0"
    assert names[-3:] == ["hf_roll85_hz", "hf_roll95_hz", "hf_ratio_above_6k"]
    # Deterministic, no duplicates.
    assert len(set(names)) == len(names)


def test_to_sequence(extractor, features):
    seq = extractor.to_sequence(features)
    assert seq.shape == (extractor.n_channels, features["mfcc"].shape[1])
    assert extractor.n_channels == 3 * N_MFCC + (N_FFT // 2 + 1) + 1  # 634
    assert np.isfinite(seq).all()


def test_pad_or_crop(extractor):
    seq = np.ones((4, 10), dtype=np.float32)
    assert extractor.pad_or_crop(seq, 10).shape == (4, 10)
    assert extractor.pad_or_crop(seq, 16).shape == (4, 16)   # edge-padded
    assert extractor.pad_or_crop(seq, 6).shape == (4, 6)     # cropped
    with pytest.raises(ValueError):
        extractor.pad_or_crop(seq, 0)
    with pytest.raises(ValueError):
        extractor.pad_or_crop(np.ones(5), 8)


# --------------------------------------------------------------------------- #
# input validation (spec requirement: clear ValueErrors, no cryptic tracebacks)
# --------------------------------------------------------------------------- #
def test_empty_audio_raises(extractor):
    with pytest.raises(ValueError, match="empty"):
        extractor.extract_all(np.zeros(0, dtype=np.float32))


def test_non_1d_audio_raises(extractor):
    with pytest.raises(ValueError, match="1-D mono"):
        extractor.extract_all(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="1-D mono"):
        extractor.extract_all(np.float32(1.0))  # scalar


def test_nan_audio_raises(extractor):
    audio = np.ones(SR // 10, dtype=np.float32)
    audio[5] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        extractor.extract_all(audio)


def test_inf_audio_raises(extractor):
    audio = np.ones(SR // 10, dtype=np.float32)
    audio[-1] = np.inf
    with pytest.raises(ValueError, match="NaN/Inf"):
        extractor.extract_mfcc(audio)


def test_none_and_string_audio_raise(extractor):
    with pytest.raises(ValueError, match="None"):
        extractor.extract_all(None)
    with pytest.raises(ValueError, match="numeric"):
        extractor.extract_all(["not", "audio"])


def test_missing_feature_keys_raise(extractor):
    with pytest.raises(ValueError, match="missing key"):
        extractor.to_flat_vector({"mfcc": np.zeros((120, 5))})
    with pytest.raises(ValueError, match="missing key"):
        extractor.to_sequence({"phase_disc": np.zeros(5)})
    with pytest.raises(ValueError, match="dict"):
        extractor.to_flat_vector(np.zeros(10))


def test_constructor_validates_configuration():
    with pytest.raises(ValueError, match="sr"):
        FeatureExtractor(sr=0)
    with pytest.raises(ValueError, match="n_mfcc"):
        FeatureExtractor(n_mfcc=1)
    with pytest.raises(ValueError, match="n_fft"):
        FeatureExtractor(n_fft=1023)
    with pytest.raises(ValueError, match="hop_length"):
        FeatureExtractor(hop_length=0)
    with pytest.raises(ValueError, match="hf_cutoff_hz"):
        FeatureExtractor(sr=8000, hf_cutoff_hz=9000)


def test_bad_method_arguments(extractor, audio):
    with pytest.raises(ValueError, match="sr"):
        extractor.extract_hf_rolloff(audio, sr=-1)
    with pytest.raises(ValueError, match="hop_length"):
        extractor.extract_phase_spectrogram(audio, hop_length=0)
    with pytest.raises(ValueError, match="n_mfcc"):
        extractor.extract_mfcc(audio, n_mfcc=1)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_very_short_audio_still_works(extractor):
    """0.01 s has fewer frames than the delta window — must not crash."""
    audio = synth_voice(seconds=0.01, seed=3)
    features = extractor.extract_all(audio, sr=SR)
    assert features["mfcc"].shape[1] == features["phase_spec"].shape[1] >= 1
    vector = extractor.to_flat_vector(features)
    assert np.isfinite(vector).all()


def test_silence_produces_finite_features(extractor):
    silence = np.zeros(SR // 4, dtype=np.float32)
    features = extractor.extract_all(silence, sr=SR)
    for key in ("mfcc", "phase_spec", "phase_disc", "hf_features"):
        assert np.isfinite(features[key]).all(), key
    vector = extractor.to_flat_vector(features)
    assert np.isfinite(vector).all()


def test_extract_all_default_sr_matches_constructor():
    fe = FeatureExtractor(sr=8000, hf_cutoff_hz=3000)
    audio = np.zeros(4000, dtype=np.float32)  # 0.5 s at 8 kHz
    audio[:100] = 0.1
    features = fe.extract_all(audio)  # sr defaults to the configured 8000
    assert features["sr"] == 8000
