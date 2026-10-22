#!/usr/bin/env python3
"""Generate VocalTrace's bundled sample corpus (10 real + 10 fake WAVs).

Real clips
----------
Preferred source: **human speech recordings** downloaded at run time

* LibriSpeech narration via ``librosa.example("libri1"/"libri2"/"libri3")``
  (public-domain readings, 22.05 kHz OGG), plus
* the whisper.cpp ``jfk.wav`` clip (16 kHz) and the SpeechRecognition
  ``english.wav`` example (44.1 kHz).

Fallback (no network): the operating system speech synthesiser
(macOS ``say`` → Linux ``espeak``/``espeak-ng`` → optional ``pyttsx3``),
which produces full-band, properly-phased speech — i.e. the *non*-neural-vocoder
side of the comparison.

Clips are silence-trimmed, carved into non-overlapping ~3.5 s segments,
resampled to 16 kHz mono and written as ``data/real/sample{1..N}.wav``.

Fake clips
----------
``DatasetBuilder.generate_fake_audio`` reconstructs each real clip through an
MFCC → mel → linear → Griffin-Lim round-trip (phase is discarded and
re-estimated: exactly the artefact family VocalTrace detects) and writes
``data/fake/sample{1..N}.wav``.

Provenance for every clip is recorded in ``outputs/sample_sources.json``.

Usage::

    python scripts/make_samples.py                 # auto (network, then TTS)
    python scripts/make_samples.py --source network
    python scripts/make_samples.py --method world   # needs pyworld
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow running as `python scripts/make_samples.py`
    sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vocaltrace.dataset import DatasetBuilder  # noqa: E402
from vocaltrace.utils import ensure_dir, setup_logging  # noqa: E402

# LibriSpeech audiobook narrations shipped as librosa examples.
LIBROSA_SPEECH_KEYS = ("libri1", "libri2", "libri3")
# Extra human-speech WAVs (direct raw URLs, tried best-effort).
EXTRA_SPEECH_URLS = (
    (
        "whisper.cpp samples/jfk.wav (JFK address, 16 kHz)",
        "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav",
    ),
    (
        "SpeechRecognition examples/english.wav (44.1 kHz)",
        "https://raw.githubusercontent.com/Uberi/speech_recognition/master/examples/english.wav",
    ),
)
# Sentences used by the offline OS-TTS fallback.
TTS_SENTENCES = (
    "The quarterly numbers are ready for your review this afternoon.",
    "Please confirm your identity before we proceed with the transfer.",
    "Our records show an unusual login attempt from a new device.",
    "The conference call has been rescheduled to half past ten.",
    "Thank you for holding while I retrieve your account details.",
    "A verification code was just sent to your registered mobile number.",
    "The shipment should arrive before the end of the business week.",
    "Let's walk through the security checklist one more time.",
    "Your appointment with the specialist has been confirmed.",
    "We updated the policy document and shared it with the team.",
    "The weather forecast predicts heavy rain later this evening.",
    "Please read the following sentence at a normal speaking pace.",
)

Source = Tuple[str, np.ndarray]  # (origin label, 16 kHz mono float32)


def log(msg: str) -> None:
    print(f"[make_samples] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def _load_16k(path: Path, sr: int = 16000) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:  # noqa: BLE001 - best-effort source
        log(f"skip {path.name}: {exc}")
        return None
    if y.size < sr:  # ignore sub-second fragments
        return None
    if not np.isfinite(y).all():
        return None
    return y.astype(np.float32)


def collect_network_sources(sr: int = 16000) -> List[Source]:
    """Human speech from librosa examples + best-effort raw-URL downloads."""
    sources: List[Source] = []

    for key in LIBROSA_SPEECH_KEYS:
        try:
            path = Path(librosa.example(key))
        except Exception as exc:  # noqa: BLE001 - offline, cache miss, ...
            log(f"librosa example {key} unavailable: {exc}")
            continue
        y = _load_16k(path, sr)
        if y is not None:
            sources.append((f"librosa.example('{key}') — LibriSpeech narration", y))
            log(f"loaded {key}: {y.size / sr:.1f} s of human speech")

    for label, url in EXTRA_SPEECH_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = response.read()
        except Exception as exc:  # noqa: BLE001 - offline / moved URL
            log(f"download failed for {url}: {exc}")
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(payload)
            tmp_path = Path(handle.name)
        y = _load_16k(tmp_path, sr)
        tmp_path.unlink(missing_ok=True)
        if y is not None:
            sources.append((f"{label} ({url})", y))
            log(f"loaded {label}: {y.size / sr:.1f} s of human speech")

    return sources


def collect_tts_sources(sr: int = 16000) -> List[Source]:
    """Offline OS-speech-synthesiser fallback (macOS/Linux/Windows)."""
    sources: List[Source] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="vocaltrace_tts_"))

    say_bin = shutil.which("say")  # macOS
    espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")  # Linux

    for idx, text in enumerate(TTS_SENTENCES):
        wav_path: Optional[Path] = None
        try:
            if say_bin:
                aiff_path = tmp_dir / f"say_{idx}.aiff"
                subprocess.run(
                    [say_bin, "-o", str(aiff_path), "--data-format=LEI16@22050", text],
                    check=True, capture_output=True, timeout=60,
                )
                wav_path = aiff_path
                origin = f"macOS `say` synthesis (sentence {idx + 1})"
            elif espeak_bin:
                out = tmp_dir / f"espeak_{idx}.wav"
                subprocess.run(
                    [espeak_bin, "-w", str(out), text],
                    check=True, capture_output=True, timeout=60,
                )
                wav_path = out
                origin = f"espeak synthesis (sentence {idx + 1})"
            else:
                import pyttsx3  # type: ignore

                out = tmp_dir / f"pyttsx3_{idx}.wav"
                engine = pyttsx3.init()
                engine.save_to_file(text, str(out))
                engine.runAndWait()
                if not out.exists():
                    raise RuntimeError("pyttsx3 produced no file")
                wav_path = out
                origin = f"pyttsx3 synthesis (sentence {idx + 1})"
        except Exception as exc:  # noqa: BLE001 - fall through to next engine
            log(f"TTS clip {idx + 1} failed ({exc})")
            continue

        y = _load_16k(wav_path, sr)
        if y is not None:
            sources.append((origin, y))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if sources:
        log(f"generated {len(sources)} clip(s) with the OS speech synthesiser")
    return sources


# --------------------------------------------------------------------------- #
# carving
# --------------------------------------------------------------------------- #
def trim_silence(y: np.ndarray, sr: int, rel_threshold: float = 0.08) -> np.ndarray:
    """Drop leading/trailing silence using an energy threshold."""
    frame, hop = 2048, 512
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 1e-8:
        return y
    active = np.flatnonzero(rms > rel_threshold * peak)
    if active.size == 0:
        return y
    start = max(0, int(active[0]) * hop - hop)
    end = min(y.size, int(active[-1]) * hop + frame + hop)
    return y[start:end]


def carve_clips(
    sources: List[Source],
    count: int,
    clip_seconds: float,
    sr: int = 16000,
) -> List[Tuple[str, np.ndarray, float]]:
    """Cut non-overlapping ``clip_seconds`` segments until ``count`` are ready.

    Returns ``(origin, clip, offset_in_source_seconds)`` triples.  Segments
    that are almost pure silence are skipped (and the cursor advanced) so the
    corpus stays speech-dense.
    """
    clip_len = int(round(clip_seconds * sr))
    clips: List[Tuple[str, np.ndarray, float]] = []

    for origin, raw in sources:
        if len(clips) >= count:
            break
        y = trim_silence(raw, sr)
        if y.size < clip_len:
            continue
        source_peak = float(np.sqrt(np.mean(np.square(y)))) + 1e-12
        cursor = 0
        while len(clips) < count and cursor + clip_len <= y.size:
            candidate = y[cursor : cursor + clip_len]
            rms = float(np.sqrt(np.mean(np.square(candidate))))
            if rms >= 0.15 * source_peak:  # skip near-silent chunks
                offset_s = cursor / sr
                clips.append((origin, candidate.astype(np.float32), offset_s))
            cursor += clip_len
    return clips


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the VocalTrace sample corpus (real + fake WAVs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dir", default=str(ROOT / "data" / "real"))
    parser.add_argument("--fake-dir", default=str(ROOT / "data" / "fake"))
    parser.add_argument("--count", type=int, default=10, help="Clips per class")
    parser.add_argument("--clip-seconds", type=float, default=3.5,
                        help="Length of each carved real clip")
    parser.add_argument("--source", choices=("auto", "network", "tts"), default="auto",
                        help="auto = network first, OS synthesiser as fallback")
    parser.add_argument("--method", choices=("griffinlim", "world", "coqui"),
                        default="griffinlim", help="Fake-generation backend")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-existing", action="store_true",
                        help="Do not delete previously generated sample*.wav files")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    if args.count <= 0:
        log(f"--count must be positive, got {args.count}")
        return 1
    if args.clip_seconds <= 0:
        log(f"--clip-seconds must be positive, got {args.clip_seconds}")
        return 1

    real_dir = ensure_dir(args.real_dir)
    fake_dir = ensure_dir(args.fake_dir)

    if not args.keep_existing:
        for folder in (real_dir, fake_dir):
            for old in folder.glob("sample*.wav"):
                old.unlink()

    # ------------------------- collect sources -------------------------- #
    sources: List[Source] = []
    if args.source in ("auto", "network"):
        sources = collect_network_sources()
    if len(sources) < 3 and args.source in ("auto", "tts"):
        if args.source == "auto":
            log("network sources insufficient — falling back to the OS TTS engine")
        tts = collect_tts_sources()
        # Keep network speech first (prefer human recordings), TTS as filler.
        sources = sources + tts
    if not sources:
        log(
            "ERROR: no speech source available. Provide network access, "
            "install `espeak-ng` (Linux) or run on macOS with `say`."
        )
        return 1

    # --------------------------- carve clips ---------------------------- #
    clips = carve_clips(sources, args.count, args.clip_seconds)
    if len(clips) < args.count:
        log(
            f"ERROR: only carved {len(clips)}/{args.count} clips "
            f"(sources too short). Lower --clip-seconds or allow the TTS "
            "fallback (--source auto)."
        )
        return 1
    log(f"carved {len(clips)} real clips of {args.clip_seconds:.1f} s")

    # --------------------------- write real ----------------------------- #
    real_paths: List[Path] = []
    provenance = []
    for idx, (origin, clip, offset_s) in enumerate(clips, start=1):
        out_path = real_dir / f"sample{idx}.wav"
        sf.write(str(out_path), clip, 16000, subtype="PCM_16")
        real_paths.append(out_path)
        provenance.append(
            {
                "file": str(out_path.relative_to(ROOT)),
                "origin": origin,
                "source_offset_seconds": round(float(offset_s), 3),
                "length_seconds": round(clip.size / 16000, 3),
            }
        )
    log(f"wrote {len(real_paths)} real files → {real_dir}")

    # --------------------------- write fakes ---------------------------- #
    builder = DatasetBuilder(sr=16000, seed=args.seed)
    try:
        fake_paths = builder.generate_fake_audio(
            real_paths, fake_dir, method=args.method
        )
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: fake generation failed: {exc}")
        return 1
    log(f"wrote {len(fake_paths)} fake files → {fake_dir} (method={args.method})")

    # --------------------------- provenance ----------------------------- #
    report = {
        "generator": "scripts/make_samples.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": 16000,
        "n_real": len(real_paths),
        "n_fake": len(fake_paths),
        "fake_method": args.method,
        "seed": args.seed,
        "real_clips": provenance,
    }
    report_path = ensure_dir(ROOT / "outputs") / "sample_sources.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"provenance → {report_path}")

    log("done: corpus ready. Next: python main.py train --real_dir data/real "
        "--fake_dir data/fake --model xgb --export_onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
