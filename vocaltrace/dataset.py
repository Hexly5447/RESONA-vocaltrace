"""Module 2 — dataset preparation for VocalTrace.

Responsibilities
----------------
* Locate real speech in a directory tree (ASVspoof-2019-LA layout with its
  ``<root>/{train,dev,eval}/flac`` structure, LJSpeech's flat ``wavs/``
  folder, or any plain folder of ``.wav`` files).
* Synthesise *vocoder-like* fakes from those recordings (Griffin-Lim
  reconstruction from MFCCs by default, WORLD vocoder or Coqui TTS when the
  optional dependency is installed — with a documented fallback to
  Griffin-Lim otherwise).
* Window the corpus into fixed-length segments, extract features, and return
  stratified train/test splits ready for :mod:`vocaltrace.train`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "librosa is required for VocalTrace. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc
try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError(
        "soundfile is required for VocalTrace. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc

from .features import FeatureExtractor
from .utils import AUDIO_EXTENSIONS

LOGGER = logging.getLogger("vocaltrace.dataset")

PathLike = Union[str, Path]
#: Fake-generation backends.  ``griffinlim`` needs no extra dependency.
FAKE_METHODS = ("griffinlim", "world", "coqui")


def _progress(iterable, **kwargs):
    """tqdm progress bar when available, otherwise a plain iterable.

    Keeps import time low: tqdm is only touched when a loop actually runs.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is in requirements.txt
        return iterable
    kwargs.setdefault("disable", None)  # auto-disable on non-TTY output
    return tqdm(iterable, **kwargs)

# Human-readable guidance shown when a directory contains no audio at all.
_EMPTY_DIR_HINT = (
    "Expected .wav/.flac files. Typical layouts that work out of the box: "
    "ASVspoof 2019 LA (<root>/train/flac/*.flac, <root>/dev/flac, ...), "
    "LJSpeech (<root>/wavs/*.wav) or any flat folder of .wav recordings. "
    "Run `python scripts/make_samples.py` to generate the bundled toy corpus."
)


class DatasetBuilder:
    """Build the VocalTrace train/eval dataset from real + fake audio folders.

    Parameters
    ----------
    sr:
        Target sample rate; all audio is resampled to this rate as mono.
    extractor:
        Optional :class:`~vocaltrace.features.FeatureExtractor` (re)using an
        existing configuration instead of the defaults.
    window_seconds, hop_seconds:
        Sliding-window geometry used by :meth:`build_dataset`.  A 2.0 s
        window with a 1.0 s hop multiplies a small corpus without changing
        the per-window statistics the classifier sees at inference time.
    n_mfcc_fake:
        MFCC count used by the Griffin-Lim fake generator.
    seed:
        Seed for reproducible fake generation.
    """

    def __init__(
        self,
        sr: int = 16000,
        extractor: Optional[FeatureExtractor] = None,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        n_mfcc_fake: int = 40,
        seed: int = 42,
    ) -> None:
        if not isinstance(sr, (int, float)) or sr <= 0:
            raise ValueError(f"sr must be a positive number, got {sr!r}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        if hop_seconds <= 0:
            raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")
        if hop_seconds > window_seconds:
            raise ValueError(
                f"hop_seconds ({hop_seconds}) must not exceed window_seconds "
                f"({window_seconds}), otherwise windows would leave gaps"
            )
        if n_mfcc_fake < 2:
            raise ValueError(f"n_mfcc_fake must be >= 2, got {n_mfcc_fake}")

        self.sr = int(sr)
        self.extractor = extractor or FeatureExtractor(sr=int(sr))
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.n_mfcc_fake = int(n_mfcc_fake)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # loading helpers
    # ------------------------------------------------------------------ #
    @property
    def target_frames(self) -> int:
        """Frame count of one full window (used to pad CNN sequences)."""
        samples = int(round(self.window_seconds * self.sr))
        return 1 + samples // self.extractor.hop_length

    def load_audio(self, path: PathLike) -> np.ndarray:
        """Load one file as 16 kHz-class mono float32 (resampled as needed)."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Audio file not found: {target}")
        if target.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format {target.suffix!r} for {target}. "
                f"Supported: {sorted(AUDIO_EXTENSIONS)}"
            )
        try:
            y, _ = librosa.load(str(target), sr=self.sr, mono=True)
        except Exception as exc:  # noqa: BLE001 - re-raise with context
            raise RuntimeError(
                f"Failed to load audio file {target}: {exc}. If the file is "
                "MP3/OGG, install audioread (`pip install audioread`) or "
                "convert it to WAV with soundfile/ffmpeg."
            ) from exc
        # Raises a descriptive ValueError for empty / NaN / Inf payloads.
        return self.extractor.validate_audio(y, name=f"audio from {target.name}")

    @staticmethod
    def _list_audio_files(directory: PathLike, what: str = "directory") -> List[Path]:
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(
                f"{what} not found: {root}. Create it or pass an existing "
                "path (see README 'Quick start')."
            )
        if not root.is_dir():
            raise NotADirectoryError(f"{what} is not a directory: {root}")

        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No audio files found under {root}. {_EMPTY_DIR_HINT}")
        return files

    def load_real_audio(self, directory: PathLike) -> List[Path]:
        """Return every real-speech file under ``directory`` (recursive).

        Handles the ASVspoof 2019 LA tree (``*.flac``), LJSpeech (``wavs``),
        or any flat folder of ``*.wav`` recordings.
        """
        files = self._list_audio_files(directory, what="Real-audio directory")
        LOGGER.info("Found %d real audio file(s) under %s", len(files), Path(directory))
        return files

    def load_fake_audio(self, directory: PathLike) -> List[Path]:
        """Return every generated fake file under ``directory`` (recursive)."""
        files = self._list_audio_files(directory, what="Fake-audio directory")
        LOGGER.info("Found %d fake audio file(s) under %s", len(files), Path(directory))
        return files

    # ------------------------------------------------------------------ #
    # fake generation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_out_path(output_dir: Path, stem: str, used: set) -> Path:
        candidate = output_dir / f"{stem}.wav"
        counter = 2
        while str(candidate) in used or candidate.exists():
            candidate = output_dir / f"{stem}_{counter}.wav"
            counter += 1
        used.add(str(candidate))
        return candidate

    def _griffinlim_reconstruct(self, y: np.ndarray) -> np.ndarray:
        """MFCC -> mel -> linear magnitude -> Griffin-Lim phase estimation.

        The round-trip throws away the original phase and re-estimates it
        iteratively, which is exactly the class of phase artefacts VocalTrace
        is designed to detect.
        """
        n_fft = self.extractor.n_fft
        hop = self.extractor.hop_length
        n_mels = self.extractor.n_mels

        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc_fake, n_fft=n_fft, hop_length=hop
        )
        mel_power = librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=n_fft, n_mels=n_mels)
        # Approximate the inverse mel filter bank (least-squares pseudo-inverse)
        # to go back to a linear-frequency power spectrogram.
        linear_power = np.linalg.pinv(mel_basis) @ mel_power
        magnitude = np.sqrt(np.maximum(linear_power, 1e-12))

        gl_kwargs = dict(
            n_iter=32, hop_length=hop, n_fft=n_fft, length=len(y), momentum=0.99
        )
        try:  # librosa >= 1.0 exposes `rng`; older versions do not.
            reconstruction = librosa.griffinlim(
                magnitude, rng=np.random.default_rng(self.seed), **gl_kwargs
            )
        except TypeError:  # pragma: no cover - exercised on librosa < 1.0
            np.random.seed(self.seed)
            reconstruction = librosa.griffinlim(magnitude, **gl_kwargs)

        # Match loudness so the classifier cannot cheat on RMS alone.
        rms_orig = float(np.sqrt(np.mean(np.square(y))))
        rms_rec = float(np.sqrt(np.mean(np.square(reconstruction))))
        if rms_rec > 1e-8 and rms_orig > 1e-8:
            reconstruction = reconstruction * min(rms_orig / rms_rec, 20.0)
        return np.clip(reconstruction, -1.0, 1.0).astype(np.float32)

    def _world_roundtrip(self, y: np.ndarray) -> np.ndarray:
        """WORLD vocoder analysis->synthesis round-trip (optional ``pyworld``)."""
        try:
            import pyworld as pw  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyworld is not installed (it needs a C++ toolchain: "
                "`pip install pyworld`). Falling back to Griffin-Lim."
            ) from exc

        y64 = y.astype(np.float64)
        f0, t = pw.harvest(y64, self.sr, f0_floor=60.0, f0_ceil=400.0)
        f0 = pw.stonemask(y64, f0, t, self.sr)
        spectral = pw.cheaptrick(y64, f0, t, self.sr)
        aperiodic = pw.d4c(y64, f0, t, self.sr)
        synthesis = pw.synthesize(f0, spectral, aperiodic, self.sr, frame_period=5.0)
        synthesis = np.asarray(synthesis, dtype=np.float32)
        if synthesis.size == 0:
            raise RuntimeError("WORLD synthesis returned an empty waveform")
        if synthesis.size != len(y):
            # Align lengths (WORLD works on frames, output can differ by a few ms).
            n = min(synthesis.size, len(y))
            synthesis = synthesis[:n]
        rms_orig = float(np.sqrt(np.mean(np.square(y))))
        rms_rec = float(np.sqrt(np.mean(np.square(synthesis))))
        if rms_rec > 1e-8 and rms_orig > 1e-8:
            synthesis = synthesis * min(rms_orig / rms_rec, 20.0)
        return np.clip(synthesis, -1.0, 1.0).astype(np.float32)

    def _coqui_reconstruct(self, y: np.ndarray, transcript: Optional[str]) -> np.ndarray:
        """Coqui TTS synthesis (optional ``TTS`` package).

        Coqui is text-to-speech, so a transcript of the source clip is
        required; without one we raise a descriptive error that the caller
        turns into a documented fallback.
        """
        if not transcript:
            raise RuntimeError(
                "method='coqui' needs a transcript=... of the source clip "
                "(Coqui is a text-to-speech system)."
            )
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Coqui TTS is not installed (`pip install coqui-tts`)."
            ) from exc
        tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False)
        synthesis = np.asarray(tts.tts(text=transcript), dtype=np.float32)
        if synthesis.size == 0:
            raise RuntimeError("Coqui TTS returned an empty waveform")
        if synthesis.size > len(y):
            synthesis = synthesis[: len(y)]
        elif synthesis.size < len(y):
            synthesis = np.pad(synthesis, (0, len(y) - synthesis.size))
        return np.clip(synthesis, -1.0, 1.0)

    def generate_fake_audio(
        self,
        real_files: Iterable[PathLike],
        output_dir: PathLike,
        method: str = "griffinlim",
        transcript: Optional[str] = None,
    ) -> List[Path]:
        """Generate vocoder-like fakes from real files and save them as WAV.

        Parameters
        ----------
        real_files:
            Source recordings (1-D paths).
        output_dir:
            Destination directory (created when missing).
        method:
            One of ``griffinlim`` (default, no extra dependency), ``world``
            (needs ``pyworld``) or ``coqui`` (needs ``coqui-tts`` + a
            transcript).  ``world``/``coqui`` automatically fall back to
            Griffin-Lim when the optional dependency is unavailable.
        transcript:
            Optional transcript required by the ``coqui`` method.

        Returns
        -------
        list of pathlib.Path
            Paths of the written fake files (same stems as the sources).
        """
        if method not in FAKE_METHODS:
            raise ValueError(
                f"Unknown fake-generation method {method!r}. "
                f"Choose one of {FAKE_METHODS}."
            )
        sources = [Path(p) for p in real_files]
        if not sources:
            raise ValueError("real_files is empty — nothing to synthesise")
        for src in sources:
            if not src.exists():
                raise FileNotFoundError(f"Source audio not found: {src}")

        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        chosen = method
        if method in ("world", "coqui"):
            probe = None
            if method == "world":
                try:  # only check the import here; runtime errors fall back below
                    import pyworld  # noqa: F401
                except ImportError:
                    probe = "pyworld is not installed"
            else:
                try:
                    import TTS  # noqa: F401
                except ImportError:
                    probe = "coqui-tts is not installed"
            if probe:
                LOGGER.warning(
                    "%s — falling back to method='griffinlim'. Install the "
                    "optional dependency to enable method='%s' (see README).",
                    probe, method,
                )
                chosen = "griffinlim"

        written: List[Path] = []
        used_names: set = set()
        for src in _progress(sources, desc=f"fakes[{chosen}]"):
            y = self.load_audio(src)
            try:
                if chosen == "griffinlim":
                    fake = self._griffinlim_reconstruct(y)
                elif chosen == "world":
                    fake = self._world_roundtrip(y)
                else:
                    fake = self._coqui_reconstruct(y, transcript)
            except Exception as exc:  # noqa: BLE001 - documented fallback
                if chosen != "griffinlim":
                    LOGGER.warning(
                        "%s generation failed (%s) — falling back to "
                        "method='griffinlim'", chosen, exc,
                    )
                    chosen = "griffinlim"
                    fake = self._griffinlim_reconstruct(y)
                else:
                    raise RuntimeError(
                        f"Griffin-Lim fake generation failed for {src}: {exc}"
                    ) from exc

            out_path = self._resolve_out_path(dest, src.stem, used_names)
            try:
                sf.write(str(out_path), fake, self.sr, subtype="PCM_16")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to write fake audio to {out_path}: {exc}"
                ) from exc
            written.append(out_path)

        LOGGER.info("Wrote %d fake file(s) to %s", len(written), dest)
        return written

    # ------------------------------------------------------------------ #
    # dataset assembly
    # ------------------------------------------------------------------ #
    def _iter_windows(self, y: np.ndarray) -> List[np.ndarray]:
        """Split ``y`` into sliding windows (whole clip when shorter)."""
        n = y.size
        win = int(round(self.window_seconds * self.sr))
        hop = int(round(self.hop_seconds * self.sr))
        if n <= win:
            return [y]
        starts = list(range(0, n - win + 1, hop))
        if starts[-1] != n - win:
            starts.append(n - win)  # always cover the tail of the clip
        return [y[s : s + win] for s in starts]

    def build_dataset(
        self,
        real_dir: PathLike,
        fake_dir: PathLike,
        vector: str = "flat",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extract features for every file and return ``(X, y, file_paths)``.

        ``y`` uses ``0 = real`` and ``1 = fake`` as per the spec.
        ``file_paths`` holds one entry per returned sample (windowed clips
        repeat their source path).

        Parameters
        ----------
        vector:
            ``"flat"`` (default) returns the aggregated 1-D vector for
            XGBoost; ``"sequence"`` returns ``(n_channels, target_frames)``
            arrays for the 1-D CNN.
        """
        if vector not in ("flat", "sequence"):
            raise ValueError(
                f"vector must be 'flat' or 'sequence', got {vector!r}"
            )

        real_files = self.load_real_audio(real_dir)
        fake_files = self.load_fake_audio(fake_dir)

        samples: List[np.ndarray] = []
        labels: List[int] = []
        paths: List[str] = []

        for label, files in ((0, real_files), (1, fake_files)):
            desc = "features[real]" if label == 0 else "features[fake]"
            for file_path in _progress(files, desc=desc):
                y_audio = self.load_audio(file_path)
                for window in self._iter_windows(y_audio):
                    feats = self.extractor.extract_all(window, sr=self.sr)
                    if vector == "flat":
                        samples.append(self.extractor.to_flat_vector(feats))
                    else:
                        seq = self.extractor.to_sequence(feats)
                        samples.append(
                            self.extractor.pad_or_crop(seq, self.target_frames)
                        )
                    labels.append(label)
                    paths.append(str(file_path))

        if not samples:  # pragma: no cover - defensive (_list_audio_files raises first)
            raise ValueError(
                f"Dataset is empty — no samples extracted from {real_dir} / "
                f"{fake_dir}. {_EMPTY_DIR_HINT}"
            )

        X = np.stack(samples) if vector == "sequence" else np.vstack(samples)
        y = np.asarray(labels, dtype=np.int64)
        LOGGER.info(
            "Dataset ready: %d samples (%d real, %d fake), vector=%s, shape=%s",
            X.shape[0], int((y == 0).sum()), int((y == 1).sum()), vector, X.shape,
        )
        return X, y, paths

    @staticmethod
    def split_train_test(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified 80/20 train/test split (falls back gracefully).

        When stratification is impossible (e.g. a class with a single sample
        in a tiny toy dataset) a plain split is used with a logged warning
        instead of crashing.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if len(y) < 2:
            raise ValueError(
                f"Need at least 2 samples to split, got {len(y)}"
            )
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {test_size}")

        from sklearn.model_selection import train_test_split as _sk_train_test_split

        stratify = y
        try:
            return _sk_train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            LOGGER.warning(
                "Stratified split impossible (%s); using a plain split instead",
                exc,
            )
            return _sk_train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=None,
            )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DatasetBuilder(sr={self.sr}, window={self.window_seconds}s, "
            f"hop={self.hop_seconds}s)"
        )
