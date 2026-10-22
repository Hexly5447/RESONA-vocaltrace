"""Tests for Module 5 — the CODING COBRAS web GUI.

Covers: logo assets, app construction (no server), the single-file handler
(including its never-traceback contract) and the ``main.py gui`` CLI wiring.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


# --------------------------------------------------------------------------- #
# logo assets
# --------------------------------------------------------------------------- #
def test_logo_banner_exists_and_is_valid_svg():
    path = ASSETS / "resona_logo.svg"
    assert path.exists(), "RESONA banner logo is missing"
    tree = ET.parse(path)  # raises if the SVG is malformed XML
    root = tree.getroot()
    assert root.tag.endswith("svg")
    text = path.read_text(encoding="utf-8")
    assert "RESONA" in text, "banner must carry the product name"
    assert "aria-label" in text and "coding cobras" in text


def test_logo_icon_exists_and_is_valid_svg():
    path = ASSETS / "resona_icon.svg"
    assert path.exists(), "RESONA icon mark is missing"
    root = ET.parse(path).getroot()
    assert root.tag.endswith("svg")
    assert "RESONA" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# app construction + handlers
# --------------------------------------------------------------------------- #
def test_build_app_builds_without_starting_a_server():
    from vocaltrace.gui import build_app

    app = build_app()
    assert app is not None
    # Gradio Blocks exposes launch() — our entry point to the server.
    assert callable(getattr(app, "launch", None))


def test_single_file_handler_without_audio_is_graceful():
    from vocaltrace.gui import _analyze_single

    card, scores, status = _analyze_single(None, 0.5)
    assert "no analysis yet" in card or "waiting" in card
    assert "Analyse" in scores or "appear" in scores


def test_single_file_handler_analyses_real_file(toy_corpus):
    from vocaltrace.gui import _analyze_single

    real = str(toy_corpus["real_files"][0])
    card, scores, status = _analyze_single(real, 0.5)
    assert "REAL" in card or "FAKE" in card, card
    assert "vt-verdict" in card
    assert "P(fake)" in scores and "Phase discontinuity" in scores
    assert "ms" in status, status


def test_single_file_handler_swallows_bad_threshold(toy_corpus):
    from vocaltrace.gui import _analyze_single

    card, scores, status = _analyze_single(str(toy_corpus["real_files"][0]), 1.5)
    assert "failed" in card.lower() or "REAL" in card
    assert "Traceback" not in status


def test_batch_handler_reports_missing_directory():
    from vocaltrace.gui import _analyze_directory

    table, csv_path, status = _analyze_directory("/definitely/not/here", None)
    assert table == []
    assert csv_path is None
    assert "error" in status and "directory not found" in status


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_cli_gui_help_lists_options():
    proc = subprocess.run(
        [sys.executable, "main.py", "gui", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in ("--port", "--host", "--share", "--no_browser"):
        assert flag in proc.stdout, f"{flag} missing from gui --help"


def test_cli_no_command_message_mentions_gui():
    proc = subprocess.run(
        [sys.executable, "main.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert "no command" in proc.stderr
    assert "gui" in proc.stderr
