"""End-to-end pipeline tests: training artifacts, report plots and the CLI.

Runs the complete VocalTrace flow (dataset → features → XGBoost → ONNX →
detector) on the 5-file toy corpus and asserts the spec's behavioural
requirements: exit codes, CSV output, plot generation and reporting.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``python main.py <args>`` from the repository root."""
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# --------------------------------------------------------------------------- #
# training artifacts
# --------------------------------------------------------------------------- #
def test_training_produces_spec_metrics_and_artifacts(trained_artifacts):
    metrics = trained_artifacts["metrics"]
    for key in (
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "confusion_matrix", "n_samples",
    ):
        assert key in metrics, f"metric {key} missing"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["roc_auc"] is None or 0.0 <= metrics["roc_auc"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)

    model_path = Path(trained_artifacts["model_path"])
    assert model_path.exists() and model_path.stat().st_size > 0
    assert model_path.suffix == ".json", "XGBoost model must be saved as .json"

    onnx_path = Path(trained_artifacts["onnx_path"])
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    metrics_file = Path(trained_artifacts["metrics_path"])
    assert metrics_file.exists()
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert payload["model_type"] == "xgb"

    # Standard report plots generated during training.
    outputs_dir = trained_artifacts["outputs_dir"]
    for name in ("confusion_matrix.png", "roc_curve.png", "feature_importance.png"):
        plot = outputs_dir / name
        assert plot.exists(), f"{name} was not generated"
        assert plot.stat().st_size > 5_000


def test_pipeline_classifies_toy_corpus(detector, toy_corpus):
    """Full pipeline on the 5-file toy set: ≥ 80 % correct verdicts."""
    files = list(toy_corpus["real_files"]) + list(toy_corpus["fake_files"])
    labels = [0] * 3 + [1] * 2  # 0 = real, 1 = fake
    correct = 0
    for path, label in zip(files, labels):
        result = detector.analyze_file(path)
        predicted = 1 if result["verdict"] == "FAKE" else 0
        correct += int(predicted == label)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["latency_ms"] > 0
    assert correct >= 4, f"pipeline accuracy too low: {correct}/{len(files)}"


# --------------------------------------------------------------------------- #
# report module
# --------------------------------------------------------------------------- #
def test_plot_waveform_and_spectrogram(toy_corpus, tmp_path):
    from vocaltrace.report import plot_waveform_and_spectrogram

    out = tmp_path / "panels.png"
    written = plot_waveform_and_spectrogram(toy_corpus["real_files"][0], out)
    assert written.exists() and written.stat().st_size > 10_000


def test_plot_roc_and_confusion(tmp_path):
    from vocaltrace.report import plot_confusion_matrix, plot_roc_curve

    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.1, 0.3, 0.7, 0.9, 0.2, 0.8]

    roc = plot_roc_curve(y_true, y_score, tmp_path / "roc.png")
    assert roc.exists() and roc.stat().st_size > 5_000

    cm = plot_confusion_matrix(y_true, [0, 0, 1, 1, 1, 1], tmp_path / "cm.png")
    assert cm.exists() and cm.stat().st_size > 5_000


def test_report_input_validation(tmp_path):
    from vocaltrace.report import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_waveform_and_spectrogram,
    )

    with pytest.raises(FileNotFoundError):
        plot_waveform_and_spectrogram(tmp_path / "missing.wav", tmp_path / "x.png")
    with pytest.raises(ValueError, match="equal length"):
        plot_confusion_matrix([0, 1], [0], tmp_path / "x.png")
    with pytest.raises(ValueError, match="both classes"):
        plot_roc_curve([0, 0, 0], [0.1, 0.2, 0.3], tmp_path / "x.png")
    with pytest.raises(ValueError, match=r"\.png"):
        plot_confusion_matrix([0, 1], [0, 1], tmp_path / "not_png.jpg")
    with pytest.raises(ValueError, match="empty"):
        plot_roc_curve([], [], tmp_path / "x.png")


# --------------------------------------------------------------------------- #
# CLI — success paths (exit code 0)
# --------------------------------------------------------------------------- #
def test_cli_detect_file_prints_verdict(trained_artifacts, toy_corpus):
    fake = toy_corpus["fake_files"][0]
    proc = _run_cli(
        "detect", "--file", str(fake),
        "--model", str(trained_artifacts["onnx_path"]),
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert "verdict" in proc.stdout
    assert "FAKE" in proc.stdout
    assert "latency" in proc.stdout
    assert "Traceback" not in proc.stderr


def test_cli_detect_directory_writes_csv(trained_artifacts, toy_corpus, tmp_path):
    out_csv = tmp_path / "results.csv"
    proc = _run_cli(
        "detect", "--directory", str(toy_corpus["fake_dir"]),
        "--output", str(out_csv),
        "--model", str(trained_artifacts["onnx_path"]),
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert out_csv.exists()

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"file", "verdict", "confidence", "latency_ms"} <= set(rows[0])
    assert all(row["verdict"] == "FAKE" for row in rows), rows


def test_cli_train_end_to_end(toy_corpus, tmp_path):
    """`main.py train ... --export_onnx` must succeed with exit code 0."""
    models_dir = tmp_path / "models"
    outputs_dir = tmp_path / "outputs"
    proc = _run_cli(
        "train",
        "--real_dir", str(toy_corpus["real_dir"]),
        "--fake_dir", str(toy_corpus["fake_dir"]),
        "--model", "xgb",
        "--export_onnx",
        "--n_estimators", "15",
        "--max_depth", "3",
        "--test_size", "0.3",
        "--no_plots",
        "--models_dir", str(models_dir),
        "--outputs_dir", str(outputs_dir),
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "accuracy" in proc.stdout
    assert "onnx" in proc.stdout
    assert (models_dir / "vocaltrace_xgb.json").exists()
    assert (models_dir / "vocaltrace.onnx").exists()
    assert (outputs_dir / "metrics.json").exists()
    # Parity must be reported and within tolerance.
    payload = json.loads((outputs_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["onnx_max_abs_diff"] <= 1e-5


# --------------------------------------------------------------------------- #
# CLI — failure paths (exit code 1, clear message, no traceback)
# --------------------------------------------------------------------------- #
def test_cli_without_command_fails():
    proc = _run_cli()
    assert proc.returncode == 1
    assert "no command" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_detect_without_source_fails():
    proc = _run_cli("detect")
    assert proc.returncode == 1
    assert "input source" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_detect_missing_file_fails():
    proc = _run_cli("detect", "--file", "definitely_missing.wav")
    assert proc.returncode == 1
    assert "error" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_detect_empty_directory_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    proc = _run_cli("detect", "--directory", str(empty))
    assert proc.returncode == 1
    assert "no audio files" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_train_missing_real_dir_fails(tmp_path):
    proc = _run_cli(
        "train",
        "--real_dir", str(tmp_path / "nope"),
        "--fake_dir", str(tmp_path / "nope2"),
        "--models_dir", str(tmp_path / "models"),
        "--outputs_dir", str(tmp_path / "outputs"),
        "--no_plots",
    )
    assert proc.returncode == 1
    assert "not found" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_bad_flag_fails_with_code_1():
    proc = _run_cli("detect", "--definitely_not_a_flag")
    assert proc.returncode == 1  # spec: usage errors exit 1 (not 2)
    assert "Traceback" not in proc.stderr


def test_cli_mic_degrades_gracefully(trained_artifacts):
    """With or without a microphone, the command never crashes."""
    proc = _run_cli(
        "detect", "--mic", "--duration", "1",
        "--model", str(trained_artifacts["onnx_path"]),
    )
    assert proc.returncode in (0, 1), proc.stderr
    assert "Traceback" not in proc.stderr
    if proc.returncode == 1:
        assert "microphone" in (proc.stderr + proc.stdout).lower()
