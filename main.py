#!/usr/bin/env python3
"""VocalTrace command-line interface.

Commands
--------
Train + export ONNX (exit code 0 on success, 1 on any failure)::

    python main.py train --real_dir data/real --fake_dir data/fake \\
        --model xgb --export_onnx

Analyse one file / a directory / the live microphone::

    python main.py detect --file data/real/sample1.wav
    python main.py detect --directory data/fake --output results.csv
    python main.py detect --mic --duration 10

Launch the CODING COBRAS web GUI::

    python main.py gui --port 7860

Design rules (spec §5):
* every command exits **0** on success and **1** with a clear message on
  failure — including argparse usage errors;
* stdout carries machine-readable output (verdicts, CSV paths); logging goes
  to ``outputs/vocaltrace.log`` and to stderr;
* no uncaught exceptions ever reach the user.
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Sequence

from vocaltrace import __version__
from vocaltrace.utils import (
    AUDIO_EXTENSIONS,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    get_logger,
    setup_logging,
)

LOGGER = get_logger("cli")

DEFAULT_ONNX_PATH = str(DEFAULT_MODEL_DIR / "vocaltrace.onnx")


# --------------------------------------------------------------------------- #
# argparse (usage errors exit with code 1, not 2)
# --------------------------------------------------------------------------- #
class _VocalTraceArgumentParser(argparse.ArgumentParser):
    """ArgumentParser whose ``error()`` exits with code 1 (spec requirement)."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser with ``train``, ``detect`` and ``gui``."""
    parser = _VocalTraceArgumentParser(
        prog="main.py",
        description=(
            "VocalTrace — acoustic phase-discontinuity detector for vishing "
            "defence (detects neural-vocoder deepfake speech on-device)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"VocalTrace {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="{train,detect,gui}")

    # ---------------------------- train --------------------------------- #
    train_p = sub.add_parser(
        "train",
        help="Train a detector and (optionally) export it to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_p.add_argument("--real_dir", default=str(PROJECT_ROOT / "data" / "real"),
                         help="Directory with real speech (searched recursively)")
    train_p.add_argument("--fake_dir", default=str(PROJECT_ROOT / "data" / "fake"),
                         help="Directory with fake speech (searched recursively)")
    train_p.add_argument("--model", choices=("xgb", "cnn"), default="xgb",
                         help="Classifier to train (cnn needs the optional torch)")
    train_p.add_argument("--export_onnx", dest="export_onnx", action="store_true",
                         help="Export models/vocaltrace.onnx (on by default; "
                              "flag kept for spec compatibility)")
    train_p.add_argument("--no_onnx", dest="export_onnx", action="store_false",
                         help="Skip ONNX export")
    train_p.set_defaults(export_onnx=True)
    train_p.add_argument("--test_size", type=float, default=0.2,
                         help="Stratified test fraction")
    train_p.add_argument("--epochs", type=int, default=30,
                         help="CNN training epochs (ignored by xgb)")
    train_p.add_argument("--learning_rate", type=float, default=None,
                         help="Override the learning rate "
                              "(default: 0.1 for xgb, 1e-3 for cnn)")
    train_p.add_argument("--n_estimators", type=int, default=200,
                         help="XGBoost boosting rounds")
    train_p.add_argument("--max_depth", type=int, default=6,
                         help="XGBoost max tree depth")
    train_p.add_argument("--window_seconds", type=float, default=2.0,
                         help="Training window length")
    train_p.add_argument("--hop_seconds", type=float, default=1.0,
                         help="Training window hop")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--models_dir", default=str(DEFAULT_MODEL_DIR),
                         help="Where to write model files")
    train_p.add_argument("--outputs_dir", default=str(DEFAULT_OUTPUT_DIR),
                         help="Where to write metrics, plots and logs")
    train_p.add_argument("--no_plots", action="store_true",
                         help="Skip report plots (metrics/ONNX still produced)")
    train_p.set_defaults(func=_cmd_train)

    # ---------------------------- detect -------------------------------- #
    detect_p = sub.add_parser(
        "detect",
        help="Analyse a file, a directory, or the live microphone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = detect_p.add_mutually_exclusive_group()
    source.add_argument("--file", default=None, help="Path to one audio file")
    source.add_argument("--directory", default=None,
                        help="Directory of audio files (searched recursively)")
    source.add_argument("--mic", action="store_true",
                        help="Analyse the live microphone input")
    detect_p.add_argument("--duration", type=float, default=10.0,
                          help="Microphone capture duration in seconds")
    detect_p.add_argument("--output", default=None,
                          help="Optional CSV path for the results")
    detect_p.add_argument("--model", default=DEFAULT_ONNX_PATH,
                          help="Path to the ONNX model")
    detect_p.add_argument("--threshold", type=float, default=0.5,
                          help="P(fake) decision threshold (0-1)")
    detect_p.add_argument("--window_seconds", type=float, default=2.0,
                          help="Sliding window length (stream mode)")
    detect_p.add_argument("--hop_seconds", type=float, default=0.5,
                          help="Sliding window hop (stream mode)")
    detect_p.set_defaults(func=_cmd_detect)

    # ---------------------------- gui ----------------------------------- #
    gui_p = sub.add_parser(
        "gui",
        help="Launch the CODING COBRAS web interface (needs gradio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    gui_p.add_argument("--host", default="127.0.0.1",
                       help="Interface to bind (use 0.0.0.0 to share on LAN)")
    gui_p.add_argument("--port", type=int, default=7860,
                       help="TCP port for the local web server")
    gui_p.add_argument("--share", action="store_true",
                       help="Also create a public gradio.live link")
    gui_p.add_argument("--no_browser", action="store_true",
                       help="Do not open the browser automatically")
    gui_p.set_defaults(func=_cmd_gui)

    return parser


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> int:
    from vocaltrace.train import format_metrics, run_training

    kwargs: dict = dict(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        model_type=args.model,
        export_onnx=args.export_onnx,
        test_size=args.test_size,
        epochs=args.epochs,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        models_dir=args.models_dir,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
        make_plots=not args.no_plots,
    )
    if args.learning_rate is not None:
        kwargs["learning_rate"] = args.learning_rate
    if args.model == "xgb":
        kwargs.update(
            n_estimators=args.n_estimators, max_depth=args.max_depth
        )

    summary = run_training(**kwargs)

    print(format_metrics(summary["metrics"], title="VocalTrace test metrics"))
    print(f"model    : {summary['model_path']}")
    if summary.get("onnx_path"):
        print(
            f"onnx     : {summary['onnx_path']} "
            f"(max |Δ| vs source = {summary.get('onnx_max_abs_diff', 0.0):.2e}, "
            f"tolerance 1e-5)"
        )
    else:
        print("onnx     : skipped (--no_onnx)")
    print(f"metrics  : {summary.get('metrics_path', 'n/a')}")
    return 0


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #
_RESULT_FIELDS = (
    "file",
    "verdict",
    "confidence",
    "fake_probability",
    "phase_discontinuity_score",
    "hf_rolloff_score",
    "latency_ms",
    "threshold",
)


def _print_result(result: dict) -> None:
    verdict = result.get("verdict", "?")
    marker = "FAKE" if verdict == "FAKE" else "REAL"
    bar = "-" * 56
    print(bar)
    print(f"file      : {result.get('file')}")
    print(f"verdict   : {marker}  (confidence {result.get('confidence', 0.0):.3f})")
    print(f"P(fake)   : {result.get('fake_probability', 0.0):.4f} "
          f"  threshold {result.get('threshold', 0.5):.2f}")
    print(f"phase disc: {result.get('phase_discontinuity_score', 0.0):.4f} "
          f"(higher = more vocoder-like)")
    print(f"HF roll-off: {result.get('hf_rolloff_score', 0.0):.4f} "
          f"(lower = less high-frequency energy)")
    print(f"latency   : {result.get('latency_ms', 0.0):.1f} ms")
    print(bar)


def _write_csv(rows: List[dict], csv_path: str) -> Path:
    target = Path(csv_path)
    if target.parent and str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_RESULT_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in _RESULT_FIELDS})
    except OSError as exc:
        raise RuntimeError(f"Could not write results CSV {target}: {exc}") from exc
    return target


def _cmd_detect(args: argparse.Namespace) -> int:
    from vocaltrace.detector import VocalTraceDetector

    if not args.file and not args.directory and not args.mic:
        print(
            "error: choose an input source: --file WAV, --directory DIR or "
            "--mic",
            file=sys.stderr,
        )
        return 1

    detector = VocalTraceDetector(onnx_path=args.model, threshold=args.threshold)

    # ------------------------- single file ------------------------------ #
    if args.file:
        result = detector.analyze_file(args.file)
        _print_result(result)
        if args.output:
            path = _write_csv([result], args.output)
            print(f"results  : {path}")
        return 0

    # ------------------------- microphone ------------------------------- #
    if args.mic:
        result = detector.analyze_microphone(
            duration_seconds=args.duration,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        if result is None:
            print(
                "error: microphone analysis is unavailable "
                "(sounddevice missing, no input device, or capture failed). "
                "Details are in outputs/vocaltrace.log. Use "
                "`python main.py detect --file <wav>` instead.",
                file=sys.stderr,
            )
            return 1
        _print_result(result)
        if args.output:
            path = _write_csv([result], args.output)
            print(f"results  : {path}")
        return 0

    # ------------------------- directory -------------------------------- #
    root = Path(args.directory)
    if not root.exists():
        print(f"error: directory not found: {root}", file=sys.stderr)
        return 1
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        print(
            f"error: no audio files found under {root}. Supported formats: "
            f"{sorted(AUDIO_EXTENSIONS)}",
            file=sys.stderr,
        )
        return 1

    rows: List[dict] = []
    failures = 0
    for idx, file_path in enumerate(files, start=1):
        try:
            rows.append(detector.analyze_file(file_path))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            failures += 1
            LOGGER.warning("Skipping %s: %s", file_path, exc)
            print(f"[{idx}/{len(files)}] SKIP {file_path} ({exc})", file=sys.stderr)

    if not rows:
        print("error: every file failed to analyse — see outputs/vocaltrace.log",
              file=sys.stderr)
        return 1

    for row in rows:
        _print_result(row)

    n_real = sum(1 for r in rows if r["verdict"] == "REAL")
    n_fake = sum(1 for r in rows if r["verdict"] == "FAKE")
    mean_conf = sum(float(r["confidence"]) for r in rows) / len(rows)
    mean_lat = sum(float(r["latency_ms"]) for r in rows) / len(rows)
    print("=" * 56)
    print(f"files     : {len(rows)} analysed, {failures} skipped")
    print(f"verdicts  : REAL={n_real}  FAKE={n_fake}")
    print(f"mean conf : {mean_conf:.3f}   mean latency: {mean_lat:.1f} ms")

    if args.output:
        path = _write_csv(rows, args.output)
        print(f"results   : {path}")
    print("=" * 56)
    return 0


# --------------------------------------------------------------------------- #
# gui
# --------------------------------------------------------------------------- #
def _cmd_gui(args: argparse.Namespace) -> int:
    from vocaltrace.gui import run_gui

    return run_gui(
        host=args.host, port=args.port, share=args.share,
        inbrowser=not args.no_browser,
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit code (0 = success, 1 = failure)."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        print(
            "error: no command given — use `python main.py train ...`, "
            "`python main.py detect ...` or `python main.py gui` (see --help)",
            file=sys.stderr,
        )
        return 1

    try:
        return int(args.func(args))
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover - `| head` scenarios
        return 0
    except Exception as exc:  # noqa: BLE001 - last-resort CLI guard
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        LOGGER.debug("traceback:\n%s", traceback.format_exc())
        print(f"error: {exc}", file=sys.stderr)
        print("(full details in outputs/vocaltrace.log)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
