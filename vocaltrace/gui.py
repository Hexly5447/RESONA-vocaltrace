"""Module 5 — **CODING COBRAS** web GUI for VocalTrace (Gradio 6).

Launch::

    python main.py gui                 # http://122.0.0.1:7860 (opens browser)
    python main.py gui --port 9000 --no_browser

The interface is a fully local **security-operations dashboard** (the browser
talks to a localhost server; no audio leaves the machine):

* **Header HUD** — CODING COBRAS branding, live status chips (model, runtime,
  uptime clock).
* **KPI strip** — total scans, threats flagged, clean verdicts, average
  latency, plus the shipped model's accuracy / AUC / ONNX parity.
* **Live terminal feed** — every scan appends a timestamped line.
* **Tabs** — ``SINGLE SCAN`` (upload or record), ``BATCH SCANNER`` (folder →
  CSV), ``MODEL LAB`` (train via the real CLI in a subprocess), ``INTEL``
  (explainer, limitations, privacy).

Everything degrades gracefully: a missing ``gradio`` install, a missing model
or a bad input path all surface as friendly messages, never tracebacks.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import PROJECT_ROOT, get_logger, setup_logging

LOGGER = get_logger("gui")

PathLike = Any

DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "vocaltrace.onnx"
LOGO_BANNER = PROJECT_ROOT / "assets" / "resona_logo.svg"
LOGO_ICON = PROJECT_ROOT / "assets" / "resona_icon.svg"
METRICS_FILE = PROJECT_ROOT / "outputs" / "metrics.json"

#: Same column order the CLI writes (main.py ``_RESULT_FIELDS``).
RESULT_FIELDS = (
    "file",
    "verdict",
    "confidence",
    "fake_probability",
    "phase_discontinuity_score",
    "hf_rolloff_score",
    "latency_ms",
    "threshold",
)
_TABLE_FIELDS = RESULT_FIELDS[:-1]  # drop threshold from the on-screen table

_START_TIME = time.time()

# --------------------------------------------------------------------------- #
# stylesheet — SOC / cyber HUD theme
# --------------------------------------------------------------------------- #
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@500;600;700&display=swap');

.gradio-container {
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(34,227,212,.07), transparent 60%),
        radial-gradient(700px 420px at -10% 110%, rgba(34,211,238,.06), transparent 60%),
        #04070d !important;
    color: #c9d6e8;
    font-family: 'Rajdhani', 'Segoe UI', system-ui, sans-serif;
}
.gradio-container::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(34,227,212,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,227,212,.035) 1px, transparent 1px);
    background-size: 46px 46px;
}

/* ------------------------------------------------------------------ header */
.vt-topbar { display: flex; align-items: center; justify-content: space-between;
             gap: 16px; flex-wrap: wrap; margin: 2px 2px 10px; }
.vt-brand  { display: flex; align-items: center; gap: 16px; }
.vt-brand svg { width: auto; height: auto; max-width: none;
                filter: drop-shadow(0 0 12px rgba(34,227,212,.4)); }
.vt-name   { font-family: 'JetBrains Mono', ui-monospace, monospace;
             font-size: 1.55rem; font-weight: 700; letter-spacing: .26em;
             color: #f0fdf8; line-height: 1.1; }
.vt-name b { color: #22e3d4; text-shadow: 0 0 14px rgba(34,227,212,.55); }
.vt-tag    { font-size: .7rem; letter-spacing: .3em; color: #5c7089;
             text-transform: uppercase; margin-top: 4px; }
.vt-chips  { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.vt-chip   { font-family: 'JetBrains Mono', ui-monospace, monospace;
             font-size: .7rem; letter-spacing: .08em; color: #8fe6e0;
             background: linear-gradient(180deg, #08131d, #060e16);
             border: 1px solid #123040; border-radius: 999px; padding: 6px 13px;
             box-shadow: 0 0 10px rgba(34,227,212,.06) inset; }
.vt-chip.cy { color: #7fd7ff; border-color: #123044; }
.vt-dot  { color: #22e3d4; animation: vt-blink 1.6s ease-in-out infinite; }
.vt-clock  { color: #e6fff5; }
@keyframes vt-blink { 0%,100% { opacity: 1 } 50% { opacity: .25 } }

/* -------------------------------------------------------------------- KPIs */
.vt-kpis { display: grid; gap: 12px; margin: 4px 0 12px;
           grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.vt-kpi  { background: linear-gradient(180deg, #0a121d, #060b12);
           border: 1px solid #12283a; border-radius: 12px; padding: 13px 16px;
           position: relative; overflow: hidden; }
.vt-kpi::after { content: ""; position: absolute; top: 0; left: 8%; width: 84%;
                 height: 2px; border-radius: 2px;
                 background: linear-gradient(90deg, transparent, #22e3d4, transparent);
                 opacity: .55; }
.vt-kpi .n { font-family: 'JetBrains Mono', ui-monospace, monospace;
             font-size: 1.75rem; font-weight: 700; color: #eafff6;
             text-shadow: 0 0 14px rgba(34,227,212,.35); line-height: 1.15; }
.vt-kpi .l { font-size: .64rem; letter-spacing: .22em; color: #5c7089;
             text-transform: uppercase; margin-top: 3px; }
.vt-kpi.bad::after { background: linear-gradient(90deg, transparent, #ff3b5c, transparent); }
.vt-kpi.bad .n { color: #ff6b85; text-shadow: 0 0 14px rgba(255,59,92,.4); }
.vt-kpi.cy::after { background: linear-gradient(90deg, transparent, #22d3ee, transparent); }
.vt-kpi.cy .n { color: #a5f3ff; text-shadow: 0 0 14px rgba(34,211,238,.4); }

/* ------------------------------------------------------------------ panels */
.vt-panel { background: linear-gradient(180deg, #081019, #05090f);
            border: 1px solid #12283a; border-radius: 14px; padding: 18px 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,.45); }
.vt-h2 { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: .78rem;
         letter-spacing: .3em; color: #22e3d4; text-transform: uppercase;
         margin: 0 0 12px; }
.vt-h2::before { content: "▸ "; color: #22d3ee; }

/* --------------------------------------------------------------- scan card */
.vt-screen { border: 1px solid #12283a; border-radius: 14px; padding: 30px 24px;
             text-align: center; position: relative; overflow: hidden;
             background:
               radial-gradient(120% 130% at 50% -20%, #0b1622 0%, #05090f 70%); }
.vt-screen.real { border-color: rgba(34,227,212,.45);
                  box-shadow: 0 0 44px rgba(34,227,212,.12) inset; }
.vt-screen.fake { border-color: rgba(255,59,92,.6);
                  animation: vt-pulse 1.7s ease-in-out infinite; }
.vt-screen.none { border-color: #12283a; }
@keyframes vt-pulse {
    0%,100% { box-shadow: 0 0 18px rgba(255,59,92,.18) inset, 0 0 12px rgba(255,59,92,.12); }
    50%     { box-shadow: 0 0 52px rgba(255,59,92,.34) inset, 0 0 30px rgba(255,59,92,.22); }
}
.vt-verdict { font-family: 'JetBrains Mono', ui-monospace, monospace;
              font-size: 3rem; font-weight: 700; letter-spacing: .3em;
              line-height: 1.05; margin: .2rem 0 .1rem; }
.vt-real .vt-verdict { color: #22e3d4; text-shadow: 0 0 22px rgba(34,227,212,.6); }
.vt-fake .vt-verdict { color: #ff3b5c; text-shadow: 0 0 22px rgba(255,59,92,.6); }
.vt-none .vt-verdict { color: #44586e; font-size: 1.15rem; letter-spacing: .12em; }
.vt-glyph { font-size: 1.6rem; line-height: 1; }
.vt-real .vt-glyph { color: #22e3d4; }
.vt-fake .vt-glyph { color: #ff3b5c; }
.vt-meta  { color: #7d93ad; font-family: 'JetBrains Mono', ui-monospace, monospace;
            font-size: .82rem; margin-top: .55rem; }

/* ------------------------------------------------------------------ meters */
.vt-meter { display: flex; align-items: center; gap: 12px; margin: 11px 0;
            font-size: .85rem; }
.vt-meter .vt-lbl { width: 11.5rem; color: #7d93ad; letter-spacing: .06em;
                    text-transform: uppercase; font-size: .68rem; }
.vt-meter .vt-track { flex: 1; height: 15px; border-radius: 4px; overflow: hidden;
                      background: #0a1420; border: 1px solid #12283a; }
.vt-meter .vt-fill { display: block; height: 100%; width: 0;
    background:
        repeating-linear-gradient(90deg, transparent 0 9px, rgba(2,6,12,.9) 9px 12px),
        linear-gradient(90deg, #0f9e93, #22e3d4);
    box-shadow: 0 0 12px rgba(34,227,212,.45); }
.vt-meter.warn .vt-fill {
    background:
        repeating-linear-gradient(90deg, transparent 0 9px, rgba(2,6,12,.9) 9px 12px),
        linear-gradient(90deg, #b45309, #ffb020);
    box-shadow: 0 0 12px rgba(255,176,32,.45); }
.vt-meter.bad .vt-fill {
    background:
        repeating-linear-gradient(90deg, transparent 0 9px, rgba(2,6,12,.9) 9px 12px),
        linear-gradient(90deg, #b91c3c, #ff3b5c);
    box-shadow: 0 0 12px rgba(255,59,92,.5); }
.vt-meter .vt-val { width: 4.6rem; text-align: right; color: #eafff6;
                    font-family: 'JetBrains Mono', ui-monospace, monospace;
                    font-size: .85rem; }
.vt-chips2 { color: #5c7089; font-family: 'JetBrains Mono', ui-monospace, monospace;
             font-size: .72rem; margin-top: 12px; letter-spacing: .05em; }

/* ----------------------------------------------------------------- terminal */
.vt-term { background: #03060b; border: 1px solid #0f2233; border-radius: 12px;
           padding: 13px 16px; font-family: 'JetBrains Mono', ui-monospace, monospace;
           font-size: .78rem; line-height: 1.75; color: #7fd7ff;
           max-height: 218px; overflow-y: auto; }
.vt-term .t   { color: #33465c; }
.vt-term .ok  { color: #22e3d4; }
.vt-term .bad { color: #ff6b85; }
.vt-term .hi  { color: #eafff6; }
.vt-term .dim { color: #44586e; }
.vt-status { font-family: 'JetBrains Mono', ui-monospace, monospace;
             font-size: .78rem; color: #7d93ad; margin-top: 10px; }
.vt-err { color: #ff6b85; }

/* ------------------------------------------------------------------ inputs */
input[type=text], textarea {
    background: #050b12 !important; border: 1px solid #12283a !important;
    color: #c9d6e8 !important; font-family: 'JetBrains Mono', ui-monospace,
    monospace !important; border-radius: 8px !important; }
input[type=range] { accent-color: #22e3d4; }
.vt-go, .vt-go button { background: linear-gradient(180deg, #22e3d4, #12b8ab) !important;
    color: #04211f !important; border: none !important; font-weight: 700 !important;
    letter-spacing: .24em !important; text-transform: uppercase !important;
    border-radius: 10px !important; box-shadow: 0 0 20px rgba(34,227,212,.35);
    transition: box-shadow .15s ease, transform .1s ease; }
.vt-go:hover, .vt-go button:hover { box-shadow: 0 0 34px rgba(34,227,212,.65);
                      transform: translateY(-1px); }
.vt-tab { background: transparent !important; }

/* ----------------------------------------------------------------- md / misc */
.vt-md { color: #a9bcd2; }
.vt-md h3 { color: #22e3d4; font-family: 'JetBrains Mono', monospace;
            letter-spacing: .12em; }
.vt-md b { color: #eafff6; }
.vt-md code { color: #7fd7ff; background: #08131d; border: 1px solid #123040;
              padding: .05rem .4rem; border-radius: 6px;
              font-family: 'JetBrains Mono', monospace; }
.vt-banner svg { width: min(640px, 92%); height: auto; display: block;
                 margin: 6px auto 2px; filter: drop-shadow(0 0 18px rgba(34,227,212,.18)); }
"""

_CLOCK_JS = """
function vtTick(){
  var el = document.getElementById('vt-clock');
  if (el) { el.textContent = new Date().toLocaleTimeString('en-GB'); }
}
setInterval(vtTick, 1000); vtTick();
"""

_FALLBACK_HEADER = (
    '<div class="vt-name">CODING <b>COBRAS</b> · VOCALTRACE</div>'
)


# --------------------------------------------------------------------------- #
# lazily-created singletons
# --------------------------------------------------------------------------- #
def _require_gradio():
    """Import gradio or raise a friendly, actionable error."""
    try:
        import gradio as gr  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "The CODING COBRAS GUI needs the optional `gradio` package. "
            "Install it with `pip install gradio` (Python 3.10+), or use the "
            "CLI: `python main.py detect --file <wav>`."
        ) from exc
    return gr


_DETECTOR: Optional[Any] = None
_DETECTOR_LOCK = threading.Lock()


def _get_detector(threshold: float = 0.5) -> Any:
    """Process-wide warmed-up detector (created on first analysis)."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            from .detector import VocalTraceDetector  # noqa: PLC0415

            _DETECTOR = VocalTraceDetector(
                onnx_path=DEFAULT_ONNX_PATH, threshold=float(threshold)
            )
        else:
            _DETECTOR.threshold = float(threshold)
    return _DETECTOR


def _icon_svg(size: int = 72) -> str:
    """Inline the cobra mark, scaled for the header."""
    try:
        svg = LOGO_ICON.read_text(encoding="utf-8")
    except OSError:
        return ""
    svg = svg.replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)
    return svg


def _logo_svg(width: int = 320) -> str:
    """Inline the full RESONA lockup (mark + wordmark), scaled to ``width``."""
    try:
        svg = LOGO_BANNER.read_text(encoding="utf-8")
    except OSError:
        return _icon_svg(width)
    # preserve the 920:300 aspect ratio of the lockup
    height = int(round(width * 300 / 920))
    svg = svg.replace("<svg ", f'<svg width="{width}" height="{height}" ', 1)
    return svg


# --------------------------------------------------------------------------- #
# dashboard state (KPI tiles + terminal feed)
# --------------------------------------------------------------------------- #
def _blank_state() -> Dict[str, Any]:
    return {"scans": 0, "fakes": 0, "reals": 0, "latencies": [], "feed": []}


def _shipped_metrics() -> Dict[str, str]:
    """Accuracy / AUC / parity of the shipped model (read once per build)."""
    out = {"accuracy": "—", "auc": "—", "parity": "—"}
    try:
        payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        m = payload.get("metrics", {})
        if m.get("accuracy") is not None:
            out["accuracy"] = f"{float(m['accuracy']):.4f}"
        if m.get("roc_auc") is not None:
            out["auc"] = f"{float(m['roc_auc']):.4f}"
        if payload.get("onnx_max_abs_diff") is not None:
            out["parity"] = f"{float(payload['onnx_max_abs_diff']):.1e}"
    except (OSError, ValueError):
        pass
    return out


def _kpi_html(state: Dict[str, Any]) -> str:
    """The KPI strip: session counters + shipped-model stats."""
    scans = int(state.get("scans", 0))
    fakes = int(state.get("fakes", 0))
    reals = int(state.get("reals", 0))
    lats = [float(x) for x in state.get("latencies", [])]
    avg = (sum(lats) / len(lats)) if lats else None
    shipped = _shipped_metrics()

    def tile(n: str, label: str, cls: str = "") -> str:
        return (f'<div class="vt-kpi {cls}"><div class="n">{n}</div>'
                f'<div class="l">{label}</div></div>')

    return (
        '<div class="vt-kpis">'
        + tile(str(scans), "files scanned")
        + tile(str(fakes), "threats flagged", "bad")
        + tile(str(reals), "verified human")
        + tile(f"{avg:.1f}" if avg is not None else "—", "avg latency ms", "cy")
        + tile(shipped["accuracy"], "model accuracy")
        + tile(shipped["auc"], "model roc-auc")
        + tile(shipped["parity"], "onnx parity |Δ|", "cy")
        + "</div>"
    )


def _feed_html(state: Dict[str, Any]) -> str:
    """Terminal-style activity feed (newest first)."""
    lines: List[str] = []
    for entry in list(state.get("feed", []))[:9]:
        kind = entry.get("kind", "dim")
        lines.append(
            f'<div><span class="t">[{entry.get("time", "--:--:--")}]</span> '
            f'<span class="{kind}">{entry.get("text", "")}</span></div>'
        )
    if not lines:
        lines = ['<div><span class="dim">awaiting first scan… _</span></div>']
    return (
        '<div class="vt-panel"><div class="vt-h2">Live activity feed</div>'
        f'<div class="vt-term">{"".join(lines)}</div></div>'
    )


def _remember(state: Dict[str, Any], text: str, kind: str,
              latency: Optional[float] = None) -> Dict[str, Any]:
    """Return an updated copy of the dashboard state (immutable in/out)."""
    new = dict(state or {})
    new["feed"] = [
        {"time": datetime.now().strftime("%H:%M:%S"), "text": text, "kind": kind}
    ] + list(new.get("feed", []))[:39]
    new["scans"] = int(new.get("scans", 0)) + 1
    if kind == "bad":
        new["fakes"] = int(new.get("fakes", 0)) + 1
    elif kind == "ok":
        new["reals"] = int(new.get("reals", 0)) + 1
    if latency is not None:
        lats = list(new.get("latencies", []))
        lats.append(float(latency))
        new["latencies"] = lats[-99:]
    return new


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def _card(verdict: str = "", meta: str = "", note: str = "") -> str:
    """Big verdict screen: REAL (green) / FAKE (red, pulsing) / standby."""
    if verdict in ("REAL", "FAKE"):
        cls = "vt-real" if verdict == "REAL" else "vt-fake"
        glyph = "✓ SECURE" if verdict == "REAL" else "⚠ THREAT"
        head = (
            f'<div class="vt-glyph">{glyph.split()[0]}</div>'
            f'<div class="vt-verdict">{verdict}</div>'
        )
    else:
        cls = "vt-none"
        head = f'<div class="vt-verdict">{note or "standby"}</div>'
    return (
        f'<div class="vt-screen {cls}">{head}'
        f'<div class="vt-meta">{meta}</div></div>'
    )


def _bar(label: str, value: float, hint: str = "") -> str:
    """Segmented neon meter; green < 0.4, amber < 0.7, red otherwise."""
    value = max(0.0, min(1.0, float(value)))
    cls = "" if value < 0.4 else ("warn" if value < 0.7 else "bad")
    pct = value * 100.0
    tail = f" <span class='dim'>({hint})</span>" if hint else ""
    return (
        f'<div class="vt-meter {cls}">'
        f'<span class="vt-lbl">{label}{tail}</span>'
        f'<span class="vt-track"><i class="vt-fill" style="width:{pct:.1f}%"></i></span>'
        f'<span class="vt-val">{value:.4f}</span>'
        f"</div>"
    )


def _scores_html(result: Dict[str, Any]) -> str:
    p_fake = float(result["fake_probability"])
    bars = "".join(
        (
            _bar("P(fake)", p_fake, "model output"),
            _bar("Phase discontinuity", result["phase_discontinuity_score"],
                 "higher = vocoder-like"),
            _bar("HF roll-off score", result["hf_rolloff_score"],
                 "lower = less HF energy"),
            _bar("Confidence", result["confidence"], "max(p, 1−p)"),
        )
    )
    chips = (
        f'<div class="vt-chips2">threshold {float(result["threshold"]):.2f} · '
        f'latency {float(result["latency_ms"]):.1f} ms · '
        "runtime onnx · 100% on-device</div>"
    )
    return bars + chips


# --------------------------------------------------------------------------- #
# tab 1 — single scan (core kept separate so its contract stays testable)
# --------------------------------------------------------------------------- #
def _analyze_single_core(
    audio_path: Optional[str], threshold: float
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Analyse one file. Returns (result dict, error message) — one is None."""
    if not audio_path:
        return None, None  # plain "waiting" state, not an error
    if not 0.0 < float(threshold) < 1.0:
        return None, "threshold must lie in (0, 1)"
    detector = _get_detector(threshold)
    return detector.analyze_file(audio_path), None


def _single_strings(result: Dict[str, Any]) -> Tuple[str, str, str]:
    """Build (card, meters, status) strings from a successful result dict."""
    verdict = result["verdict"]
    meta = (
        f'P(fake) = {float(result["fake_probability"]):.4f} · '
        f'confidence {float(result["confidence"]):.3f}'
    )
    status = (
        f'scanned <span class="hi">{Path(str(result["file"])).name}</span> '
        f'in {float(result["latency_ms"]):.1f} ms'
    )
    return _card(verdict, meta), _scores_html(result), status


def _analyze_single(audio_path: Optional[str], threshold: float) -> Tuple[str, str, str]:
    """Analyse one uploaded/recorded file; returns (card, meters, status)."""
    try:
        result, error = _analyze_single_core(audio_path, threshold)
        if error is not None:
            raise ValueError(error)
        if result is None:
            return (
                _card(note="waiting for input…"),
                '<div class="vt-status">Load audio (or record with the '
                "microphone) and press the <b>Analyse</b> button.</div>",
                "",
            )
        return _single_strings(result)
    except Exception as exc:  # noqa: BLE001 - GUI must never traceback
        LOGGER.warning("single-file analysis failed: %s", exc)
        return (
            _card(note="scan failed"),
            f'<div class="vt-status vt-err">error: {exc}</div>',
            "see outputs/vocaltrace.log for details",
        )


def _scan_single(
    audio_path: Optional[str], threshold: float, state: Optional[Dict[str, Any]]
):
    """Dashboard-aware wrapper: same analysis + KPI/feed/state updates."""
    try:
        result, error = _analyze_single_core(audio_path, threshold)
        if error is not None:
            raise ValueError(error)
        if result is None:
            card, scores, status = _analyze_single(audio_path, threshold)
            return (card, scores, status,
                    _kpi_html(state or _blank_state()),
                    _feed_html(state or _blank_state()),
                    dict(state or _blank_state()))
        card, scores, status = _single_strings(result)
    except Exception as exc:  # noqa: BLE001 - GUI must never traceback
        LOGGER.warning("single-file analysis failed: %s", exc)
        card, scores, status = _analyze_single(audio_path, threshold)
        return (card, scores, status,
                _kpi_html(state or _blank_state()),
                _feed_html(state or _blank_state()),
                dict(state or _blank_state()))

    verdict = result["verdict"]
    kind = "bad" if verdict == "FAKE" else "ok"
    label = "THREAT" if verdict == "FAKE" else "HUMAN "
    name = Path(str(result["file"])).name
    prob = float(result["fake_probability"])
    new_state = _remember(
        dict(state or _blank_state()),
        f"SCAN {name} → {label}  P(fake)={prob:.4f}",
        kind,
        float(result["latency_ms"]),
    )
    return card, scores, status, _kpi_html(new_state), _feed_html(new_state), new_state


# --------------------------------------------------------------------------- #
# tab 2 — batch scanner
# --------------------------------------------------------------------------- #
def _analyze_directory(
    dir_path: Optional[str], csv_path: Optional[str]
) -> Tuple[List[List[Any]], Optional[str], str]:
    """Screen a directory; returns (table rows, csv file path, status)."""
    try:
        from .utils import AUDIO_EXTENSIONS  # noqa: PLC0415

        if not str(dir_path or "").strip():
            return [], None, "enter a folder path and press <b>Screen folder</b>."

        root = Path(str(dir_path)).expanduser()
        if not root.exists():
            raise ValueError(f"directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"not a directory: {root}")

        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            raise ValueError(
                f"no audio files found under {root} "
                f"(supported: {sorted(AUDIO_EXTENSIONS)})"
            )

        detector = _get_detector(0.5)
        rows: List[Dict[str, Any]] = []
        skipped = 0
        for path in files:
            try:
                rows.append(detector.analyze_file(path))
            except Exception as exc:  # noqa: BLE001 - one bad file ≠ batch failure
                skipped += 1
                LOGGER.warning("Skipping %s: %s", path, exc)
        if not rows:
            raise RuntimeError("every file failed to analyse — see outputs/vocaltrace.log")

        target = Path(str(csv_path)).expanduser() if str(csv_path or "").strip() \
            else PROJECT_ROOT / "outputs" / "batch_results.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDS))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})

        table = [[row.get(k, "") for k in _TABLE_FIELDS] for row in rows]
        n_real = sum(1 for r in rows if r["verdict"] == "REAL")
        n_fake = len(rows) - n_real
        mean_conf = sum(float(r["confidence"]) for r in rows) / len(rows)
        mean_lat = sum(float(r["latency_ms"]) for r in rows) / len(rows)
        status = (
            f"<b>{len(rows)}</b> analysed, {skipped} skipped → "
            f"REAL={n_real} FAKE={n_fake} · mean confidence {mean_conf:.3f} · "
            f"mean latency {mean_lat:.1f} ms · CSV written to "
            f"<code>{target}</code>"
        )
        return table, str(target), status
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("batch analysis failed: %s", exc)
        return [], None, f'<span class="vt-err">error: {exc}</span>'


def _scan_batch(
    dir_path: Optional[str], csv_path: Optional[str],
    state: Optional[Dict[str, Any]],
):
    """Dashboard-aware wrapper around the batch scanner."""
    table, csv_file, status = _analyze_directory(dir_path, csv_path)
    new_state = dict(state or _blank_state())
    if table:
        fakes = sum(1 for row in table if len(row) > 1 and row[1] == "FAKE")
        reals = len(table) - fakes
        lats = [float(row[5]) for row in table if len(row) > 5]
        avg = sum(lats) / len(lats) if lats else None
        name = Path(str(dir_path)).name or str(dir_path)
        kind = "bad" if fakes else "ok"
        new_state["feed"] = [
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": (f"BATCH {name} → {len(table)} files · "
                         f"{fakes} threat{'s' if fakes != 1 else ''}, "
                         f"{reals} human"),
                "kind": kind,
            }
        ] + list(new_state.get("feed", []))[:39]
        new_state["scans"] = int(new_state.get("scans", 0)) + len(table)
        new_state["fakes"] = int(new_state.get("fakes", 0)) + fakes
        new_state["reals"] = int(new_state.get("reals", 0)) + reals
        if avg is not None:
            lats_hist = list(new_state.get("latencies", [])) + [avg]
            new_state["latencies"] = lats_hist[-99:]
    elif "error" in str(status):
        new_state["feed"] = [
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "text": f"BATCH FAILED — {str(status)[:90]}",
                "kind": "dim",
            }
        ] + list(new_state.get("feed", []))[:39]
    return table, csv_file, status, _kpi_html(new_state), _feed_html(new_state), new_state


# --------------------------------------------------------------------------- #
# tab 3 — model lab (runs the real CLI in a subprocess)
# --------------------------------------------------------------------------- #
def _train(
    real_dir: str,
    fake_dir: str,
    model_type: str,
    epochs: int,
    n_estimators: int,
    max_depth: int,
    export_onnx: bool,
) -> Tuple[str, List[str], str]:
    """Run ``main.py train`` as a subprocess; returns (metrics, plots, paths).

    A subprocess mirrors how the CLI isolates the xgb and torch/OpenMP
    runtimes, so pressing Train can never crash the GUI process itself.
    """
    try:
        if model_type not in ("xgb", "cnn"):
            raise ValueError(f"model must be 'xgb' or 'cnn', got {model_type!r}")

        outputs_dir = PROJECT_ROOT / "outputs"
        models_dir = PROJECT_ROOT / "models"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "main.py"), "train",
            "--real_dir", str(real_dir), "--fake_dir", str(fake_dir),
            "--model", str(model_type),
            "--epochs", str(int(epochs)),
            "--n_estimators", str(int(n_estimators)),
            "--max_depth", str(int(max_depth)),
            "--models_dir", str(models_dir),
            "--outputs_dir", str(outputs_dir),
        ]
        cmd.append("--export_onnx" if export_onnx else "--no_onnx")

        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(
                "training failed (exit "
                f"{proc.returncode}): {' | '.join(tail) or 'no output'}"
            )

        metrics_file = outputs_dir / "metrics.json"
        metrics_md = "training finished — see outputs/metrics.json"
        if metrics_file.exists():
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            m = payload.get("metrics", {})
            auc = m.get("roc_auc")
            auc_txt = "n/a" if auc is None else f"{auc:.4f}"
            cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
            metrics_md = "\n".join(
                (
                    f"### Training finished — {payload.get('model_type', '?')}",
                    "",
                    "| metric | value |",
                    "|---|---|",
                    f"| windows | {payload.get('n_samples', '?')} |",
                    f"| test accuracy | {m.get('accuracy', float('nan')):.4f} |",
                    f"| precision | {m.get('precision', float('nan')):.4f} |",
                    f"| recall | {m.get('recall', float('nan')):.4f} |",
                    f"| F1 | {m.get('f1', float('nan')):.4f} |",
                    f"| ROC-AUC | {auc_txt} |",
                    f"| confusion | TP={cm[1][1]} FP={cm[0][1]} "
                    f"FN={cm[1][0]} TN={cm[0][0]} |",
                    f"| ONNX parity (max \\|Δ\\|) | "
                    f"{payload.get('onnx_max_abs_diff', float('nan')):.2e} "
                    "(tol 1e-5) |",
                )
            )

        plots = [str(p) for p in sorted(outputs_dir.glob("*.png"))]
        paths_md = (
            f"model: `{payload.get('model_path', models_dir)}` · "
            f"onnx: `{payload.get('onnx_path', 'skipped')}` · "
            f"metrics: `{metrics_file}`"
        ) if metrics_file.exists() else ""
        return metrics_md, plots, paths_md
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("training failed: %s", exc)
        return (
            f"### Training failed\n\n<span class='vt-err'>error: {exc}</span>",
            [],
            "full details in outputs/vocaltrace.log",
        )


# --------------------------------------------------------------------------- #
# app assembly
# --------------------------------------------------------------------------- #
def build_app():
    """Build and return the Gradio ``Blocks`` app (no server started).

    The visual identity (CODING COBRAS HUD + dark CSS) is attached at launch
    time — Gradio 6 moved ``css``/``theme`` from the constructor to
    :meth:`gr.Blocks.launch`.
    """
    gr = _require_gradio()
    shipped = _shipped_metrics()

    with gr.Blocks(title="RESONA · VocalTrace") as app:
        # ---------------------------------------------------------- header --
        try:
            logo = _logo_svg(520)
        except OSError:  # pragma: no cover - defensive
            logo = ""
        gr.HTML(
            '<div class="vt-topbar">'
            f'<div class="vt-brand">{logo}<div>'
            '<div class="vt-tag">VocalTrace · acoustic threat detection '
            'console</div></div></div>'
            '<div class="vt-chips">'
            '<span class="vt-chip"><span class="vt-dot">●</span> SYSTEM ONLINE</span>'
            '<span class="vt-chip">MODEL vocaltrace.onnx</span>'
            '<span class="vt-chip cy">RUNTIME ONNX</span>'
            '<span class="vt-chip cy">DSP phase-discontinuity</span>'
            '<span class="vt-chip">UTC <span id="vt-clock">--:--:--</span></span>'
            "</div></div>"
        )

        state = gr.State(_blank_state())
        kpi = gr.HTML(_kpi_html(_blank_state()))
        feed = gr.HTML(_feed_html(_blank_state()))

        with gr.Tabs():
            # ------------------------------------------------- single scan --
            with gr.Tab("Single scan", elem_classes=["vt-tab"]):
                with gr.Row():
                    with gr.Column(scale=5):
                        audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="INPUT — upload a WAV or record live",
                            elem_classes=["vt-panel"],
                        )
                    with gr.Column(scale=1, min_width=220):
                        thr = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.5, step=0.05,
                            label="THRESHOLD — P(fake) ≥ t ⇒ FAKE",
                        )
                        run_btn = gr.Button("▶ Analyse", variant="primary",
                                            elem_classes=["vt-go"])
                with gr.Row():
                    with gr.Column():
                        card = gr.HTML(_card())
                    with gr.Column():
                        meters = gr.HTML(
                            '<div class="vt-panel"><div class="vt-h2">'
                            "Threat telemetry</div>"
                            '<div class="vt-status">Meters appear here after '
                            "the scan.</div></div>"
                        )
                status1 = gr.HTML('<div class="vt-status"></div>')

                run_btn.click(
                    fn=_scan_single,
                    inputs=[audio, thr, state],
                    outputs=[card, meters, status1, kpi, feed, state],
                    api_name="analyse",
                )

            # ---------------------------------------------- batch scanner --
            with gr.Tab("Batch scanner", elem_classes=["vt-tab"]):
                with gr.Row():
                    dir_in = gr.Textbox(
                        label="TARGET FOLDER (searched recursively)",
                        value=str(PROJECT_ROOT / "data" / "fake"),
                        scale=4,
                    )
                    csv_in = gr.Textbox(
                        label="CSV OUTPUT (blank = outputs/batch_results.csv)",
                        value="", scale=3,
                    )
                scan_btn = gr.Button("▶ Screen folder", variant="primary",
                                     elem_classes=["vt-go"])
                status2 = gr.HTML('<div class="vt-status">Load a folder to '
                                  "screen it.</div>")
                table = gr.Dataframe(
                    headers=list(_TABLE_FIELDS),
                    datatype=["str", "str", "number", "number", "number",
                              "number", "number"],
                    label="VERDICTS",
                    interactive=False,
                )
                csv_out = gr.File(label="RESULTS CSV — click to download")

                scan_btn.click(
                    fn=_scan_batch,
                    inputs=[dir_in, csv_in, state],
                    outputs=[table, csv_out, status2, kpi, feed, state],
                    api_name="screen_folder",
                )

            # -------------------------------------------------- model lab --
            with gr.Tab("Model lab", elem_classes=["vt-tab"]):
                gr.Markdown(
                    '<div class="vt-md">Runs the **real training pipeline** '
                    "(same code as `python main.py train`) in an isolated "
                    "subprocess — the GUI process itself can never crash — "
                    "then shows metrics, plots and the ONNX parity check.</div>"
                )
                with gr.Row():
                    real_in = gr.Textbox(label="REAL SPEECH DIRECTORY",
                                         value=str(PROJECT_ROOT / "data" / "real"))
                    fake_in = gr.Textbox(label="FAKE SPEECH DIRECTORY",
                                         value=str(PROJECT_ROOT / "data" / "fake"))
                with gr.Row():
                    model_dd = gr.Dropdown(
                        choices=["xgb", "cnn"], value="xgb", label="CLASSIFIER",
                        info="cnn needs the optional torch and lots of data",
                    )
                    epochs_n = gr.Number(value=30, minimum=1, label="CNN EPOCHS")
                    est_n = gr.Number(value=200, minimum=1,
                                      label="XGBOOST N_ESTIMATORS")
                    depth_n = gr.Number(value=6, minimum=1, label="MAX_DEPTH")
                    onnx_cb = gr.Checkbox(value=True, label="EXPORT ONNX")
                train_btn = gr.Button("▶ Train", variant="primary",
                                      elem_classes=["vt-go"])
                train_md = gr.Markdown("Press **Train** to start.")
                plots = gr.Gallery(type="filepath", label="TRAINING REPORT",
                                   columns=3, height="auto")
                paths_md = gr.Markdown("")

                train_btn.click(
                    fn=_train,
                    inputs=[real_in, fake_in, model_dd, epochs_n, est_n,
                            depth_n, onnx_cb],
                    outputs=[train_md, plots, paths_md],
                    api_name="train",
                )

            # ----------------------------------------------------- intel --
            with gr.Tab("Intel", elem_classes=["vt-tab"]):
                gr.HTML(f'<div class="vt-banner">{LOGO_BANNER.read_text(encoding="utf-8") if LOGO_BANNER.exists() else ""}</div>')
                gr.Markdown(
                    f"""
<div class="vt-md">

### HOW IT WORKS

Real vocal folds produce **smoothly evolving phase** across short-time frames;
neural vocoders re-synthesise magnitude but leave **abrupt phase jumps**.
VocalTrace measures that artefact (plus spectral roll-off effects) and feeds
MFCC+Δ+ΔΔ, phase and HF features to XGBoost — about **11 ms** per 2-second
window, no GPU, no internet, nothing leaves this machine.

**Shipped model** — accuracy `{shipped['accuracy']}` · ROC-AUC `{shipped['auc']}`
· ONNX parity `{shipped['parity']}` (tol 1e-5).

### HONEST LIMITATIONS

* detects **vocoder artefacts**, not "AI" — weak against replays and 8 kHz
  phone-band audio;
* the shipped model is trained on a 20-clip demo corpus — retrain on your own
  data (ASVspoof/LibriSpeech) before relying on it;
* the CNN path overfits below ~200 windows — prefer `xgb`;
* one risk signal among many — **never** auto-decline on it alone; keep a
  human in the loop.

### PRIVACY

GDPR / DPDP-friendly by design — audio is processed in this process only;
store verdict rows, not recordings. Full details in `README.md`.

*CODING COBRAS — designed & built with signal processing, XGBoost, ONNX and a
lot of coffee.*

</div>
"""
                )
    return app


def run_gui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> int:
    """Launch the GUI server (blocks until Ctrl+C). Returns an exit code."""
    setup_logging()
    app = build_app()
    LOGGER.info(
        "Starting CODING COBRAS GUI on http://%s:%d (browser=%s, share=%s)",
        host, port, inbrowser, share,
    )
    try:
        app.launch(
            server_name=host,
            server_port=int(port),
            inbrowser=bool(inbrowser),
            share=bool(share),
            favicon_path=str(LOGO_ICON) if LOGO_ICON.exists() else None,
            allowed_paths=[str(PROJECT_ROOT)],
            show_error=True,
            css=_CSS,
            js=_CLOCK_JS,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nGUI server stopped.")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import argparse

    _p = argparse.ArgumentParser(description="CODING COBRAS VocalTrace GUI")
    _p.add_argument("--host", default="127.0.0.1")
    _p.add_argument("--port", type=int, default=7860)
    _p.add_argument("--share", action="store_true")
    _p.add_argument("--no_browser", action="store_true")
    _a = _p.parse_args()
    raise SystemExit(
        run_gui(host=_a.host, port=_a.port, share=_a.share,
                inbrowser=not _a.no_browser)
    )
