"""Module 4 — the VocalTrace inference engine.

:class:`VocalTraceDetector` loads an exported ONNX graph (XGBoost or 1-D CNN)
with onnxruntime and exposes three analysis entry points:

* :meth:`VocalTraceDetector.analyze_file` — one WAV file → REAL/FAKE verdict.
* :meth:`VocalTraceDetector.analyze_stream` — in-memory chunk with a sliding
  2 s window / 0.5 s hop, aggregated with mean + max.
* :meth:`VocalTraceDetector.analyze_microphone` — live capture via the
  optional ``sounddevice`` package; degrades gracefully (returns ``None``)
  when the package or a microphone is unavailable.

Module-level helpers (:func:`load_session`, :func:`select_probability_output`,
:func:`session_probabilities`) are shared with :mod:`vocaltrace.train` so the
export parity check exercises *exactly* the same inference path as
production detection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "librosa is required for VocalTrace. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from .features import FeatureExtractor
from .utils import require_file, AUDIO_EXTENSIONS

LOGGER = logging.getLogger("vocaltrace.detector")

PathLike = Union[str, Path]
DEFAULT_ONNX_PATH = Path(__file__).resolve().parents[1] / "models" / "vocaltrace.onnx"

# Keys guaranteed by analyze_file() (the detector output schema).
OUTPUT_SCHEMA_KEYS = (
    "file",
    "verdict",
    "confidence",
    "fake_probability",
    "phase_discontinuity_score",
    "hf_rolloff_score",
    "latency_ms",
    "threshold",
)


# --------------------------------------------------------------------------- #
# onnxruntime plumbing (shared with train.py)
# --------------------------------------------------------------------------- #
def load_session(
    onnx_path: PathLike, providers: Optional[Sequence[str]] = None
) -> Any:
    """Open an onnxruntime InferenceSession with clear failure messages."""
    path = require_file(onnx_path, what="ONNX model")
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - onnxruntime is a hard dep
        raise RuntimeError(
            "onnxruntime is required for inference: pip install onnxruntime"
        ) from exc
    try:
        session = ort.InferenceSession(
            str(path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load ONNX model {path}: {exc}. The file may be "
            "corrupt or exported for a newer opset — re-run "
            "`python main.py train ... --export_onnx`."
        ) from exc
    if not session.get_inputs():
        raise RuntimeError(f"ONNX model {path} exposes no inputs")
    if not session.get_outputs():
        raise RuntimeError(f"ONNX model {path} exposes no outputs")
    return session


def select_probability_output(session: Any) -> str:
    """Pick the output that carries P(fake) (skipping label outputs)."""
    outputs = session.get_outputs()
    # 1) names that clearly denote probabilities
    for out in outputs:
        if "prob" in out.name.lower():
            return out.name
    # 2) any floating-point output (labels are int64)
    float_outputs = [o for o in outputs if "float" in (o.type or "").lower()]
    if float_outputs:
        return float_outputs[-1].name
    # 3) last resort: final output
    LOGGER.warning(
        "No floating-point output found in ONNX graph; using output '%s'",
        outputs[-1].name,
    )
    return outputs[-1].name


def _numpy_dtype(onnx_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    return np.dtype(mapping.get(onnx_type, np.float32))


def _probability_from_output(raw: np.ndarray) -> np.ndarray:
    """Convert a raw ONNX output into a flat probability vector in [0, 1]."""
    arr = np.asarray(raw)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr.ravel()
        elif arr.shape[1] >= 2:
            arr = arr[:, 1]  # P(class=1) == P(fake)
        else:  # pragma: no cover - degenerate
            arr = arr.ravel()
    elif arr.ndim > 2:  # pragma: no cover - unexpected graph
        arr = arr.reshape(arr.shape[0], -1)[:, -1]
    return np.clip(arr.astype(np.float64), 0.0, 1.0)


def session_probabilities(
    session: Any,
    features: np.ndarray,
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Run one forward pass and return P(fake) for each row of ``features``.

    Validates rank/dimensionality against the graph signature and raises
    descriptive :class:`ValueError` messages on mismatch.
    """
    inp = session.get_inputs()[0]
    name = input_name or inp.name
    dtype = _numpy_dtype(inp.type)

    data = np.asarray(features, dtype=dtype)
    expected_rank = len(inp.shape)
    if expected_rank == 1:
        if data.ndim != 1:
            raise ValueError(
                f"ONNX input '{inp.name}' expects 1-D data, got shape {data.shape}"
            )
        data = data[None, :]
    elif expected_rank == 2:
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(
                f"ONNX input '{inp.name}' expects a 2-D matrix (batch, features), "
                f"got shape {data.shape}"
            )
        expected_features = inp.shape[1]
        if isinstance(expected_features, int) and data.shape[1] != expected_features:
            raise ValueError(
                f"Feature-dimension mismatch: ONNX model expects "
                f"{expected_features} features, got {data.shape[1]}. Was the "
                "detector built with a different feature configuration than "
                "the trained model?"
            )
    elif expected_rank == 3:
        if data.ndim == 2:
            data = data[None, :, :]
        if data.ndim != 3:
            raise ValueError(
                f"ONNX input '{inp.name}' expects a 3-D tensor "
                f"(batch, channels, time), got shape {data.shape}"
            )
        expected_channels = inp.shape[1]
        if isinstance(expected_channels, int) and data.shape[1] != expected_channels:
            raise ValueError(
                f"Channel mismatch: ONNX model expects {expected_channels} "
                f"feature channels, got {data.shape[1]}."
            )
    else:
        raise ValueError(
            f"Unsupported ONNX input rank {expected_rank} for '{inp.name}'"
        )

    if output_name is None:
        output_name = select_probability_output(session)
    if not np.isfinite(data.astype(np.float64)).all():
        raise ValueError("Feature tensor contains NaN/Inf — cannot run inference")

    try:
        raw = session.run([output_name], {name: data})[0]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"ONNX inference failed: {exc}") from exc
    return _probability_from_output(raw)


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class VocalTraceDetector:
    """Acoustic phase-discontinuity detector backed by an ONNX model.

    Parameters
    ----------
    onnx_path:
        Path to ``models/vocaltrace.onnx`` (exported by ``main.py train``).
    threshold:
        Decision threshold on P(fake); verdict is FAKE when
        ``P(fake) >= threshold`` (default 0.5).
    extractor:
        Optional pre-configured :class:`FeatureExtractor`.  Its configuration
        must match the one used during training.
    """

    def __init__(
        self,
        onnx_path: PathLike = DEFAULT_ONNX_PATH,
        threshold: float = 0.5,
        extractor: Optional[FeatureExtractor] = None,
        providers: Optional[Sequence[str]] = None,
        warmup: bool = True,
    ) -> None:
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError(f"threshold must lie in (0, 1), got {threshold}")

        self.onnx_path = Path(onnx_path)
        self.threshold = float(threshold)
        self.extractor = extractor or FeatureExtractor()

        self.session = load_session(self.onnx_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = list(inp.shape)
        self.input_rank = len(self.input_shape)
        self.input_dtype = _numpy_dtype(inp.type)
        self.output_name = select_probability_output(self.session)

        # Expected feature width / channel count when statically known.
        self.expected_features: Optional[int] = None
        self.expected_channels: Optional[int] = None
        if self.input_rank == 2 and isinstance(self.input_shape[1], int):
            self.expected_features = int(self.input_shape[1])
        if self.input_rank == 3 and isinstance(self.input_shape[1], int):
            self.expected_channels = int(self.input_shape[1])

        if warmup:
            self._warm_up()

        LOGGER.info(
            "VocalTraceDetector ready | model=%s input=%s output='%s' threshold=%.2f",
            self.onnx_path.name, self.input_shape, self.output_name, self.threshold,
        )

    def _warm_up(self, seconds: float = 0.3) -> None:
        """Run one tiny end-to-end pass at construction time.

        The very first feature extraction in a process triggers just-in-time
        compilation inside librosa/numba (~0.65 s), and the first ONNX run
        allocates its arena.  Paying that cost here (once, eagerly) keeps
        *reported* per-window analysis latency representative of steady-state
        real-time performance.  Warm-up failures never break construction.
        """
        started = time.perf_counter()
        try:
            n = max(int(seconds * self.extractor.sr), 512)
            t = np.arange(n, dtype=np.float64) / float(self.extractor.sr)
            probe = (
                0.2 * np.sin(2.0 * np.pi * 440.0 * t)
                + 0.05 * np.sin(2.0 * np.pi * 1730.0 * t)
            ).astype(np.float32)
            features = self.extractor.extract_all(probe, sr=self.extractor.sr)
            self._predict_proba(features)
        except Exception as exc:  # noqa: BLE001 - warm-up must never fail init
            LOGGER.debug("Detector warm-up skipped: %s", exc)
            return
        LOGGER.debug(
            "Detector warm-up took %.0f ms",
            (time.perf_counter() - started) * 1000.0,
        )

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _features_to_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert an extract_all() dict into the tensor the graph expects."""
        if self.input_rank == 2:
            vector = self.extractor.to_flat_vector(features)
            if self.expected_features and vector.size != self.expected_features:
                raise ValueError(
                    f"Feature-dimension mismatch: model expects "
                    f"{self.expected_features}, extractor produced "
                    f"{vector.size}. Train the model and the detector with "
                    "the same FeatureExtractor configuration."
                )
            return vector.reshape(1, -1).astype(self.input_dtype)
        if self.input_rank == 3:
            sequence = self.extractor.to_sequence(features)
            if self.expected_channels and sequence.shape[0] != self.expected_channels:
                raise ValueError(
                    f"Channel mismatch: model expects {self.expected_channels} "
                    f"channels, extractor produced {sequence.shape[0]}."
                )
            return sequence[None, :, :].astype(self.input_dtype)
        raise ValueError(  # pragma: no cover - guarded in __init__
            f"Unsupported ONNX input rank {self.input_shape}"
        )

    def _predict_proba(self, features: Dict[str, Any]) -> float:
        tensor = self._features_to_input(features)
        proba = session_probabilities(
            self.session, tensor,
            input_name=self.input_name, output_name=self.output_name,
        )
        if proba.size != 1:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Expected a single probability from the model, got {proba.size}"
            )
        return float(proba[0])

    @staticmethod
    def _verdict_from_proba(proba: float, threshold: float) -> str:
        return "FAKE" if proba >= threshold else "REAL"

    @staticmethod
    def _confidence_from_proba(proba: float, threshold: float) -> float:
        """How sure the model is about *its own* verdict.

        ``max(p, 1-p)`` — 0.5 at the decision boundary, 1.0 at certainty.
        """
        del threshold  # threshold only shifts the verdict, not the confidence
        return float(max(proba, 1.0 - proba))

    @staticmethod
    def _phase_score(features: Dict[str, Any]) -> float:
        disc = np.asarray(features["phase_disc"], dtype=np.float64).ravel()
        if disc.size == 0:  # pragma: no cover - defensive
            return 0.0
        return float(np.mean(disc))

    @staticmethod
    def _hf_score(features: Dict[str, Any]) -> float:
        """Normalised high-frequency presence in [0, 1] (low ⇒ vocoder-like)."""
        hf = np.asarray(features["hf_features"], dtype=np.float64).ravel()
        sr = int(features.get("sr", 16000))
        nyquist = max(sr / 2.0, 1.0)
        roll85, roll95, ratio = float(hf[0]), float(hf[1]), float(hf[2])
        score = np.mean(
            [np.clip(roll85 / nyquist, 0, 1), np.clip(roll95 / nyquist, 0, 1),
             np.clip(ratio, 0, 1)]
        )
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def analyze_file(self, wav_path: PathLike) -> Dict[str, Any]:
        """Analyse one audio file end-to-end.

        Returns the detector output schema::

            file, verdict (REAL/FAKE), confidence (0-1), fake_probability,
            phase_discontinuity_score, hf_rolloff_score, latency_ms, threshold
        """
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {path}")
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format {path.suffix!r} for {path}. "
                f"Supported: {sorted(AUDIO_EXTENSIONS)}"
            )

        started = time.perf_counter()
        y, _ = librosa.load(str(path), sr=self.extractor.sr, mono=True)
        y = self.extractor.validate_audio(y, name=f"audio from {path.name}")
        features = self.extractor.extract_all(y, sr=self.extractor.sr)
        proba = self._predict_proba(features)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "file": str(path),
            "verdict": self._verdict_from_proba(proba, self.threshold),
            "confidence": round(self._confidence_from_proba(proba, self.threshold), 6),
            "fake_probability": round(proba, 6),
            "phase_discontinuity_score": round(self._phase_score(features), 6),
            "hf_rolloff_score": round(self._hf_score(features), 6),
            "latency_ms": round(latency_ms, 3),
            "threshold": self.threshold,
        }

    def analyze_stream(
        self,
        audio_chunk: np.ndarray,
        sr: int = 16000,
        window_seconds: float = 2.0,
        hop_seconds: float = 0.5,
        aggregation: str = "max",
    ) -> Dict[str, Any]:
        """Slide a window over an in-memory chunk and aggregate the scores.

        Windows are ``window_seconds`` long with a ``hop_seconds`` step
        (defaults: 2 s / 0.5 s per spec).  Per-window P(fake) values are
        aggregated with **mean** and **max**; the reported
        ``aggregate_fake_probability`` is the max (a single synthetic-looking
        window flags the stream — the conservative choice for vishing
        defence) or the mean when ``aggregation="mean"``.
        """
        if aggregation not in ("max", "mean"):
            raise ValueError(
                f"aggregation must be 'max' or 'mean', got {aggregation!r}"
            )
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if hop_seconds <= 0:
            raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")
        if not isinstance(sr, (int, float)) or sr <= 0:
            raise ValueError(f"sr must be a positive number, got {sr!r}")

        y = self.extractor.validate_audio(audio_chunk, name="audio_chunk")
        win = int(round(window_seconds * sr))
        hop = int(round(hop_seconds * sr))
        if win <= 0:  # pragma: no cover - guarded by validation above
            raise ValueError("window length rounded to zero samples")

        if y.size <= win:
            LOGGER.warning(
                "Stream chunk (%.2f s) shorter than the %.1f s window — "
                "analysing it as a single window",
                y.size / sr, window_seconds,
            )
            windows = [y]
        else:
            starts = list(range(0, y.size - win + 1, hop))
            if starts[-1] != y.size - win:
                starts.append(y.size - win)  # always cover the tail
            windows = [y[s : s + win] for s in starts]

        started = time.perf_counter()
        window_scores: List[float] = []
        phase_scores: List[float] = []
        hf_scores: List[float] = []
        for window in windows:
            feats = self.extractor.extract_all(window, sr=int(sr))
            window_scores.append(self._predict_proba(feats))
            phase_scores.append(self._phase_score(feats))
            hf_scores.append(self._hf_score(feats))

        scores = np.asarray(window_scores, dtype=np.float64)
        mean_p = float(scores.mean())
        max_p = float(scores.max())
        aggregate_p = max_p if aggregation == "max" else mean_p
        latency_ms = (time.perf_counter() - started) * 1000.0

        return {
            "file": "<stream>",
            "verdict": self._verdict_from_proba(aggregate_p, self.threshold),
            "confidence": round(
                self._confidence_from_proba(aggregate_p, self.threshold), 6
            ),
            "fake_probability": round(aggregate_p, 6),
            "aggregate_fake_probability": round(aggregate_p, 6),
            "mean_fake_probability": round(mean_p, 6),
            "max_fake_probability": round(max_p, 6),
            "window_fake_probabilities": [round(p, 6) for p in window_scores],
            "windows_flagged": int(
                np.sum(scores >= self.threshold)
            ),
            "num_windows": len(window_scores),
            "phase_discontinuity_score": round(float(np.mean(phase_scores)), 6),
            "hf_rolloff_score": round(float(np.mean(hf_scores)), 6),
            "latency_ms": round(latency_ms, 3),
            "threshold": self.threshold,
            "window_seconds": float(window_seconds),
            "hop_seconds": float(hop_seconds),
            "aggregation": aggregation,
        }

    def analyze_microphone(
        self,
        duration_seconds: float = 10,
        sr: int = 16000,
        window_seconds: float = 2.0,
        hop_seconds: float = 0.5,
        verbose: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Capture live microphone audio and analyse it window by window.

        Returns the aggregated result dict, or ``None`` when the optional
        ``sounddevice`` package is missing, no input device exists, or the
        capture fails — always with a logged warning, never a traceback.
        """
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")

        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            LOGGER.warning(
                "Microphone analysis unavailable: the optional `sounddevice` "
                "package is not installed. Install it with "
                "`pip install sounddevice` (see README troubleshooting)."
            )
            return None

        n_samples = int(round(duration_seconds * sr))
        try:
            devices = sd.query_devices()
            input_count = int(
                sum(1 for d in devices if d.get("max_input_channels", 0) > 0)
            )
            if input_count == 0:
                LOGGER.warning(
                    "Microphone analysis unavailable: no input audio devices "
                    "were found by PortAudio."
                )
                return None
            LOGGER.info(
                "Recording %.1f s from the default microphone at %d Hz ...",
                duration_seconds, sr,
            )
            recording = sd.rec(
                n_samples, samplerate=int(sr), channels=1, dtype="float32"
            )
            sd.wait()
        except Exception as exc:  # noqa: BLE001 - PortAudioError et al.
            LOGGER.warning(
                "Microphone analysis unavailable: %s. Falling back to file "
                "analysis (e.g. `python main.py detect --file sample.wav`).",
                exc,
            )
            return None

        audio = np.asarray(recording, dtype=np.float32).ravel()
        if audio.size == 0 or not np.isfinite(audio).all():
            LOGGER.warning("Microphone capture returned empty/invalid audio")
            return None

        result = self.analyze_stream(
            audio, sr=sr, window_seconds=window_seconds, hop_seconds=hop_seconds
        )
        result["file"] = "<microphone>"

        if verbose:
            scores = result["window_fake_probabilities"]
            for idx, score in enumerate(scores, start=1):
                verdict = self._verdict_from_proba(score, self.threshold)
                start_t = (idx - 1) * result["hop_seconds"]
                print(
                    f"[window {idx:02d} | t={start_t:5.1f}s] {verdict:<4} "
                    f"P(fake)={score:.3f}"
                )
            print(
                f"[summary] {result['verdict']} "
                f"(confidence {result['confidence']:.3f}, "
                f"mean={result['mean_fake_probability']:.3f}, "
                f"max={result['max_fake_probability']:.3f}, "
                f"{result['windows_flagged']}/{result['num_windows']} windows "
                f"flagged, {result['latency_ms']:.1f} ms)"
            )
        return result
