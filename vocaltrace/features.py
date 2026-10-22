"""Module 1 — acoustic feature extraction for VocalTrace.

The extractor converts a mono waveform into four families of features:

``mfcc``
    MFCC + delta + delta-delta (shape ``(3 * n_mfcc, frames)``) — captures the
    spectral envelope / timbre that neural vocoders tend to smooth.
``phase_spec`` / ``phase_disc``
    The raw STFT phase spectrogram plus a per-frame **phase-discontinuity
    score**.  Neural vocoders synthesise magnitudes and phases somewhat
    independently, which leaves abrupt frame-to-frame phase jumps that natural
    glottal excitation does not produce.
``hf_features``
    85 % / 95 % spectral roll-off and the fraction of energy above 6 kHz —
    many vocoders run at reduced bandwidth and roll off early.
``to_flat_vector``
    Aggregates the time axis (mean / std / 90th percentile) into a single 1-D
    vector consumed by XGBoost.

All public methods validate their input and raise descriptive
:class:`ValueError` exceptions instead of letting numpy/scipy fail with a
cryptic traceback.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:  # pragma: no cover - import guard is defensive
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "librosa is required for VocalTrace. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

LOGGER = logging.getLogger("vocaltrace.features")

AudioLike = Union[np.ndarray, Sequence[float], List[float]]

# Width of the delta/delta-delta finite-difference window used by librosa.
_DELTA_WIDTH = 9
# Relative weights of the first / second wrapped phase derivative inside the
# per-frame discontinuity score (they sum to 1 so the score stays in [0, 1]).
_W_FIRST = 0.7
_W_SECOND = 0.3


class FeatureExtractor:
    """Extract VocalTrace features from a mono waveform.

    Parameters
    ----------
    sr:
        Default sample rate (Hz).  Audio is expected at 16 kHz mono, which is
        what :func:`librosa.load` produces with ``sr=16000, mono=True``.
    n_mfcc:
        Number of MFCC coefficients (deltas double the rows, delta-deltas
        triple them: ``3 * n_mfcc`` rows).
    n_fft, hop_length:
        STFT geometry shared by the phase and roll-off features.
    n_mels:
        Mel bands used internally when reconstructing audio from MFCCs (see
        :mod:`vocaltrace.dataset`).
    hf_cutoff_hz:
        Cut-off for the high-frequency energy ratio (default 6 kHz).
    roll_percents:
        Percentiles for the spectral roll-off features (85 % and 95 %).
    """

    STAT_NAMES: Sequence[str] = ("mean", "std", "p90")

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        hf_cutoff_hz: float = 6000.0,
        roll_percents: Sequence[float] = (0.85, 0.95),
    ) -> None:
        if not isinstance(sr, (int, float)) or sr <= 0:
            raise ValueError(f"sr must be a positive number, got {sr!r}")
        if n_mfcc < 2:
            raise ValueError(f"n_mfcc must be >= 2 (deltas require it), got {n_mfcc}")
        if n_fft < 16 or n_fft % 2:
            raise ValueError(f"n_fft must be an even integer >= 16, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if n_mels < 8:
            raise ValueError(f"n_mels must be >= 8, got {n_mels}")
        if len(roll_percents) != 2 or not all(0.0 < p < 1.0 for p in roll_percents):
            raise ValueError(
                "roll_percents must contain two values in (0, 1), got "
                f"{tuple(roll_percents)}"
            )
        if hf_cutoff_hz <= 0 or hf_cutoff_hz >= sr / 2:
            raise ValueError(
                f"hf_cutoff_hz must lie in (0, sr/2 = {sr / 2}), got {hf_cutoff_hz}"
            )

        self.sr = int(sr)
        self.n_mfcc = int(n_mfcc)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.hf_cutoff_hz = float(hf_cutoff_hz)
        self.roll_percents = tuple(float(p) for p in roll_percents)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_audio(audio: AudioLike, name: str = "audio") -> np.ndarray:
        """Validate and normalise a waveform argument.

        Raises
        ------
        ValueError
            If the input is not a numeric 1-D array, is empty, or contains
            NaN/Inf values.
        """
        if audio is None:
            raise ValueError(f"{name} is None — expected a 1-D mono waveform")
        try:
            arr = np.asarray(audio)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} could not be converted to a numpy array: {exc}"
            ) from exc

        if arr.dtype.kind not in "fiu":
            raise ValueError(
                f"{name} must be numeric, got dtype={arr.dtype!r} "
                f"(kind={arr.dtype.kind!r})"
            )
        if arr.ndim == 0:
            raise ValueError(
                f"{name} must be a 1-D mono waveform, got a scalar; "
                "wrap it in an array of samples"
            )
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D mono waveform, got shape {arr.shape}. "
                "Resample to mono first, e.g. "
                "librosa.load(path, sr=16000, mono=True)."
            )
        if arr.size == 0:
            raise ValueError(
                f"{name} is empty (0 samples). Provide at least ~50 ms of audio."
            )

        arr = arr.astype(np.float32, copy=False)
        bad = ~np.isfinite(arr)
        if bad.any():
            n_bad = int(bad.sum())
            first = int(np.argmax(bad))
            raise ValueError(
                f"{name} contains {n_bad} NaN/Inf value(s) "
                f"(first at index {first}). Clean the audio before extraction."
            )
        return arr

    def _resolve_sr(self, sr: Optional[float]) -> int:
        """Validate an explicit ``sr`` or fall back to the configured rate."""
        if sr is None:
            return self.sr
        if not isinstance(sr, (int, float)) or sr <= 0:
            raise ValueError(f"sr must be a positive number, got {sr!r}")
        return int(sr)

    @staticmethod
    def _wrap(phase_diff: np.ndarray) -> np.ndarray:
        """Wrap phase differences into ``(-pi, pi]``."""
        return (phase_diff + np.pi) % (2.0 * np.pi) - np.pi

    def _stft(self, y: np.ndarray) -> np.ndarray:
        return librosa.stft(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length, center=True
        )

    # ------------------------------------------------------------------ #
    # Module 1 API
    # ------------------------------------------------------------------ #
    def extract_mfcc(
        self,
        audio: AudioLike,
        sr: Optional[int] = None,
        n_mfcc: int = 40,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> np.ndarray:
        """Return MFCC + delta + delta-delta features.

        Shape: ``(3 * n_mfcc, time_frames)``.  ``sr`` defaults to the
        extractor's configured sample rate.
        """
        y = self.validate_audio(audio, name="audio")
        sr = self._resolve_sr(sr)
        if n_mfcc < 2:
            raise ValueError(f"n_mfcc must be >= 2, got {n_mfcc}")
        if n_fft < 16 or n_fft % 2:
            raise ValueError(f"n_fft must be an even integer >= 16, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")

        # Reflect-padding cannot be shorter than n_fft; pad with zeros instead
        # so that arbitrarily short clips still produce one analysis frame.
        work = y if y.size >= n_fft else np.pad(y, (0, n_fft - y.size))

        mfcc = librosa.feature.mfcc(
            y=work, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        n_frames = mfcc.shape[1]

        # librosa's delta estimator needs at least `width` frames; pad the
        # frame axis with edge replication for very short inputs and trim back.
        left = right = 0
        padded = mfcc
        if n_frames < _DELTA_WIDTH:
            missing = _DELTA_WIDTH - n_frames
            left, right = missing - missing // 2, missing // 2
            padded = np.pad(mfcc, ((0, 0), (left, right)), mode="edge")

        delta = librosa.feature.delta(padded, width=_DELTA_WIDTH, order=1)
        delta2 = librosa.feature.delta(padded, width=_DELTA_WIDTH, order=2)

        if left or right:
            keep = slice(left, left + n_frames)
            delta = delta[:, keep]
            delta2 = delta2[:, keep]

        features = np.vstack([mfcc, delta, delta2])
        expected_rows = 3 * n_mfcc
        if features.shape != (expected_rows, n_frames):  # pragma: no cover - defensive
            raise RuntimeError(
                "MFCC extraction produced an unexpected shape "
                f"{features.shape}, expected {(expected_rows, n_frames)}"
            )
        return features.astype(np.float32)

    def extract_phase_spectrogram(
        self,
        audio: AudioLike,
        sr: Optional[int] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
    ) -> Dict[str, np.ndarray]:
        """Compute the STFT phase spectrogram and discontinuity scores.

        Returns a dictionary with:

        * ``phase_spec``  — raw wrapped phase, shape ``(n_fft // 2 + 1, frames)``
        * ``phase_diff``  — first wrapped phase derivative, same shape
        * ``phase_d2``    — second wrapped phase derivative, same shape
        * ``phase_disc``  — scalar discontinuity score per frame,
          shape ``(frames,)``, values in ``[0, 1]``.  Higher means more
          abrupt (vocoder-like) phase jumps.
        """
        y = self.validate_audio(audio, name="audio")
        sr = self._resolve_sr(sr)
        if n_fft < 16 or n_fft % 2:
            raise ValueError(f"n_fft must be an even integer >= 16, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")

        work = y if y.size >= n_fft else np.pad(y, (0, n_fft - y.size))
        spectrum = librosa.stft(
            y=work, n_fft=n_fft, hop_length=hop_length, center=True
        )
        phase = np.angle(spectrum)
        magnitude = np.abs(spectrum)

        first = np.zeros_like(phase)
        first[:, 1:] = self._wrap(np.diff(phase, axis=1))
        second = np.zeros_like(phase)
        second[:, 1:] = self._wrap(np.diff(first, axis=1))

        # Magnitude-weighted mean of |d1| and |d2| (both normalised by pi) so
        # silent frequency bins cannot dominate the per-frame score.
        total_energy = magnitude.sum(axis=0, keepdims=True)
        weights = magnitude / (total_energy + 1e-12)
        score = (
            weights
            * (_W_FIRST * np.abs(first) + _W_SECOND * np.abs(second))
            / np.pi
        ).sum(axis=0)
        score = np.clip(score, 0.0, 1.0).astype(np.float32)

        return {
            "phase_spec": phase.astype(np.float32),
            "phase_diff": first.astype(np.float32),
            "phase_d2": second.astype(np.float32),
            "phase_disc": score,
        }

    def extract_hf_rolloff(
        self, audio: AudioLike, sr: Optional[int] = None
    ) -> np.ndarray:
        """Return ``[roll_off_85%, roll_off_95%, hf_energy_ratio]`` (shape ``(3,)``).

        * The first two entries are spectral roll-off frequencies in Hz.
        * The third is the fraction of total energy above ``hf_cutoff_hz``.
        * ``sr`` defaults to the extractor's configured sample rate.
        """
        y = self.validate_audio(audio, name="audio")
        sr = self._resolve_sr(sr)

        work = y if y.size >= self.n_fft else np.pad(y, (0, self.n_fft - y.size))
        power = np.abs(librosa.stft(y=work, n_fft=self.n_fft,
                                    hop_length=self.hop_length)) ** 2
        total = float(power.sum())
        if total <= 1e-12:
            # Digital silence: roll-offs are undefined, return zeros instead
            # of NaN so downstream vectors stay finite.
            LOGGER.debug("extract_hf_rolloff: silent input, returning zeros")
            return np.zeros(3, dtype=np.float32)

        roll85 = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    S=power, sr=sr, roll_percent=self.roll_percents[0],
                    n_fft=self.n_fft,
                )
            )
        )
        roll95 = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    S=power, sr=sr, roll_percent=self.roll_percents[1],
                    n_fft=self.n_fft,
                )
            )
        )

        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        hf_mask = freqs > self.hf_cutoff_hz
        hf_ratio = float(power[hf_mask].sum() / total)

        return np.array([roll85, roll95, hf_ratio], dtype=np.float32)

    def extract_all(
        self, audio: AudioLike, sr: Optional[int] = None
    ) -> Dict[str, object]:
        """Extract every feature family in one pass.

        Parameters
        ----------
        audio:
            1-D mono waveform (validated).
        sr:
            Sample rate of ``audio`` (defaults to the extractor's ``sr``).

        Returns
        -------
        dict
            ``mfcc`` ``(3*n_mfcc, T)``, ``phase_spec`` ``(n_fft//2+1, T)``,
            ``phase_disc`` ``(T,)``, ``hf_features`` ``(3,)`` and ``sr``.
            All time axes are aligned to a common frame count ``T``.
        """
        y = self.validate_audio(audio, name="audio")
        sr = self._resolve_sr(sr)

        mfcc = self.extract_mfcc(
            y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        phase = self.extract_phase_spectrogram(
            y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        hf = self.extract_hf_rolloff(y, sr=sr)

        n_frames = min(mfcc.shape[1], phase["phase_spec"].shape[1])
        if n_frames == 0:  # pragma: no cover - defensive
            raise RuntimeError("Feature extraction produced zero time frames")
        if mfcc.shape[1] != phase["phase_spec"].shape[1]:
            LOGGER.warning(
                "Frame-count mismatch between MFCC (%d) and phase (%d); "
                "cropping both to %d frames",
                mfcc.shape[1], phase["phase_spec"].shape[1], n_frames,
            )

        return {
            "mfcc": np.ascontiguousarray(mfcc[:, :n_frames], dtype=np.float32),
            "phase_spec": np.ascontiguousarray(
                phase["phase_spec"][:, :n_frames], dtype=np.float32
            ),
            "phase_disc": np.ascontiguousarray(
                phase["phase_disc"][:n_frames], dtype=np.float32
            ),
            "hf_features": np.ascontiguousarray(hf, dtype=np.float32),
            "sr": int(sr),
        }

    # ------------------------------------------------------------------ #
    # aggregation helpers
    # ------------------------------------------------------------------ #
    @property
    def n_channels(self) -> int:
        """Channel count of the CNN input (``3*n_mfcc + n_fft//2+1 + 1``)."""
        return 3 * self.n_mfcc + (self.n_fft // 2 + 1) + 1

    @property
    def flat_dim(self) -> int:
        """Length of the flat vector produced by :meth:`to_flat_vector`."""
        return 3 * (3 * self.n_mfcc + (self.n_fft // 2 + 1) + 1) + len(
            self.roll_percents
        ) + 1

    @staticmethod
    def _missing_keys(features: Dict[str, object]) -> List[str]:
        required = ("mfcc", "phase_spec", "phase_disc", "hf_features")
        return [key for key in required if key not in features]

    @staticmethod
    def _as_2d(value: object, key: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(
                f"features[{key!r}] must be 1-D or 2-D (channels, frames), "
                f"got shape {arr.shape}"
            )
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(
                f"features[{key!r}] has an empty axis: shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"features[{key!r}] contains NaN/Inf values — the input audio "
                "was probably corrupted"
            )
        return arr

    def to_flat_vector(self, features_dict: Dict[str, object]) -> np.ndarray:
        """Aggregate the time axis (mean/std/p90) into one 1-D float32 vector.

        The ordering of the entries matches :meth:`flat_feature_names`, so the
        vector can be fed to XGBoost (optionally as a named pandas frame).
        """
        if not isinstance(features_dict, dict):
            raise ValueError(
                "features_dict must be a dict produced by extract_all(), got "
                f"{type(features_dict).__name__}"
            )
        missing = self._missing_keys(features_dict)
        if missing:
            raise ValueError(
                f"features_dict is missing key(s) {missing}; call "
                "extract_all() first"
            )

        parts: List[np.ndarray] = []
        for key in ("mfcc", "phase_spec", "phase_disc"):
            arr = self._as_2d(features_dict[key], key)
            mean = arr.mean(axis=1)
            std = arr.std(axis=1)
            p90 = np.percentile(arr, 90, axis=1)
            parts.append(np.concatenate([mean, std, p90]))

        hf = np.asarray(features_dict["hf_features"], dtype=np.float64).ravel()
        if hf.size != len(self.roll_percents) + 1:
            raise ValueError(
                f"hf_features must have {len(self.roll_percents) + 1} entries "
                f"(roll-offs + ratio), got {hf.size}"
            )
        parts.append(hf)

        vector = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError(
                "The flat feature vector contains NaN/Inf — check the input audio"
            )
        expected = self.flat_dim
        if vector.size != expected:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Flat vector has size {vector.size}, expected {expected}"
            )
        return vector

    def flat_feature_names(self) -> List[str]:
        """Human-readable names matching :meth:`to_flat_vector` element-wise."""
        names: List[str] = []
        channels = (
            ("mfcc", 3 * self.n_mfcc),
            ("phase_spec", self.n_fft // 2 + 1),
            ("phase_disc", 1),
        )
        for prefix, n_ch in channels:
            for stat in self.STAT_NAMES:
                for idx in range(n_ch):
                    names.append(f"{prefix}_{stat}_{idx}")
        names.extend(["hf_roll85_hz", "hf_roll95_hz", "hf_ratio_above_6k"])
        if len(names) != self.flat_dim:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Feature-name count {len(names)} != flat_dim {self.flat_dim}"
            )
        return names

    def to_sequence(self, features_dict: Dict[str, object]) -> np.ndarray:
        """Stack features into the CNN layout ``(n_channels, frames)``.

        Channels are ``mfcc (120) + phase_spec (513) + phase_disc (1) = 634``
        with the default configuration.
        """
        if not isinstance(features_dict, dict):
            raise ValueError(
                "features_dict must be a dict produced by extract_all(), got "
                f"{type(features_dict).__name__}"
            )
        missing = self._missing_keys(features_dict)
        if missing:
            raise ValueError(
                f"features_dict is missing key(s) {missing}; call "
                "extract_all() first"
            )

        mfcc = self._as_2d(features_dict["mfcc"], "mfcc")
        phase = self._as_2d(features_dict["phase_spec"], "phase_spec")
        disc = np.asarray(features_dict["phase_disc"], dtype=np.float64).ravel()
        if disc.size == 0:
            raise ValueError("features['phase_disc'] is empty")

        n_frames = min(mfcc.shape[1], phase.shape[1], disc.size)
        sequence = np.concatenate(
            [mfcc[:, :n_frames], phase[:, :n_frames], disc[None, :n_frames]],
            axis=0,
        ).astype(np.float32)

        if sequence.shape[0] != self.n_channels:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Sequence has {sequence.shape[0]} channels, expected "
                f"{self.n_channels}"
            )
        if not np.isfinite(sequence).all():  # pragma: no cover - defensive
            raise ValueError("The feature sequence contains NaN/Inf values")
        return sequence

    @staticmethod
    def pad_or_crop(sequence: np.ndarray, target_frames: int) -> np.ndarray:
        """Edge-pad / crop ``(channels, frames)`` to exactly ``target_frames``.

        Edge padding keeps feature statistics plausible for the tail of the
        sequence (zero padding would inject a fake silent "phase = 0" region).
        """
        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(
                f"sequence must be 2-D (channels, frames), got shape {seq.shape}"
            )
        if target_frames <= 0:
            raise ValueError(f"target_frames must be positive, got {target_frames}")
        frames = seq.shape[1]
        if frames == target_frames:
            return seq
        if frames > target_frames:
            return np.ascontiguousarray(seq[:, :target_frames])
        pad = target_frames - frames
        return np.pad(seq, ((0, 0), (0, pad)), mode="edge")

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"FeatureExtractor(sr={self.sr}, n_mfcc={self.n_mfcc}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length})"
        )
