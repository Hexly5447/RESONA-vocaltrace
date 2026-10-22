#!/usr/bin/env bash
# VocalTrace — guided live demo for a viva / presentation.
#
# Runs the full story in order with narration between steps:
#   1. Detect a fake file        5. Retrain from scratch (+ ONNX parity)
#   2. Detect a real file        6. Test suite
#   3. Batch screen -> CSV       7. Graceful failure / exit codes
#   4. Show trained metrics      8. CODING COBRAS web GUI health-check
#
# Usage:
#   ./demo.sh              # interactive: pauses before each step
#   RUN=auto ./demo.sh     # no pauses (CI, rehearsals, screen recordings)
set -uo pipefail

cd "$(dirname "$0")"

# Prefer the project venv; fall back to whatever python is on PATH.
if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python    # Git-Bash on Windows
else
    PY=python3
fi

AUTO="${RUN:-}"
pause() {
    [ -n "$AUTO" ] && return 0
    printf '\n  \033[2m── press Enter to continue ──\033[0m '
    read -r _ || true
}

banner() {
    printf '\n\033[1;36m%s\033[0m\n' "$1"
    printf '\033[2m%s\033[0m\n\n' "$2"
}

# A failing command that is EXPECTED to fail — show its output but not die.
expect_fail() {
    "$@"
    local code=$?
    printf '  \033[2m(exit code %d — this is the designed behaviour)\033[0m\n' "$code"
    return 0
}

printf '\033[1;36m'
cat <<'EOF'
 ┌──────────────────────────────────────────────────────────────┐
 │  VocalTrace — acoustic phase-discontinuity detector          │
 │  Live demo (~5 minutes)                                      │
 └──────────────────────────────────────────────────────────────┘
EOF
printf '\033[0m'
cat <<'EOF'
The pitch (say this out loud, 30 seconds):

  Criminals clone a voice from a 20-second recording and call finance
  with a fake wire instruction — a Hong Kong firm lost US $25M in 2024.
  VocalTrace detects AI-cloned speech from one PHYSICAL artifact:
  neural vocoders cannot reconstruct the phase of a waveform, so their
  phase tracks contain abrupt jumps. Deterministic DSP + XGBoost on
  ONNX Runtime. No GPU, no internet, nothing leaves the machine.
EOF

# ---------------------------------------------------------------- 1. fake
banner "STEP 1 — Detect an AI-cloned file (live)" \
       "Watch: verdict FAKE with high confidence, plus the two INTERPRETABLE scores."
$PY main.py detect --file data/fake/sample1.wav
pause

# ---------------------------------------------------------------- 2. real
banner "STEP 2 — Detect a genuine human recording (live)" \
       "Same command, real speech from the LibriSpeech corpus. Scores shift down."
$PY main.py detect --file data/real/sample3.wav
pause

# ---------------------------------------------------------------- 3. batch
banner "STEP 3 — Batch screening → CSV (the bank / KYC story)" \
       "A call centre would screen every recorded segment; verdicts land in a CSV."
$PY main.py detect --directory data/fake --output outputs/fake_results.csv
$PY main.py detect --directory data/real  --output outputs/real_results.csv
printf '  Result on this corpus: \033[1m10/10 fakes flagged, 9/10 reals passed\033[0m (criterion was >= 8/10 each).\n'
printf '  CSVs: outputs/fake_results.csv, outputs/real_results.csv\n'
pause

# ---------------------------------------------------------------- 4. metrics
banner "STEP 4 — What training produced" \
       "Held-out metrics + ONNX parity: the shipped file is PROVABLY the trained model."
if [ -f outputs/metrics.json ]; then
    $PY - <<'EOF'
import json
m = json.load(open("outputs/metrics.json"))
t = m["metrics"]
print(f"  model        : {m['model_type']}  (seed {m['seed']}, {m['n_samples']} windows)")
print(f"  test accuracy: {t['accuracy']:.4f}    ROC-AUC: {t['roc_auc']:.4f}")
print(f"  ONNX parity  : max |Δ| = {m['onnx_max_abs_diff']:.3e}  (tolerance 1e-5 — export aborts if exceeded)")
print(f"  artifacts    : models/vocaltrace_xgb.json (audit) + models/vocaltrace.onnx (runtime)")
EOF
else
    echo "  (no metrics yet — run step 5 first)"
fi
pause

# ---------------------------------------------------------------- 5. train
banner "STEP 5 — Retrain the whole pipeline from scratch" \
       "Seed pinned to 42 → reproducible. Features → XGBoost → ONNX export → parity check."
$PY main.py train --real_dir data/real --fake_dir data/fake --model xgb --export_onnx
pause

# ---------------------------------------------------------------- 6. tests
banner "STEP 6 — Test suite (58 tests)" \
       "Tests synthesize their own corpus and train a model on the fly — nothing depends on this machine."
$PY -m pytest tests/ -q
pause

# ---------------------------------------------------------------- 7. errors
banner "STEP 7 — Graceful failure (engineering maturity)" \
       "Every failure exits 1 with a clean message. No tracebacks, ever."
echo "\$ python main.py detect --mic --duration 1"
expect_fail $PY main.py detect --mic --duration 1 2>&1 | grep -E "error|WARNING" | tail -2
echo
echo "\$ python main.py detect --file missing.wav"
expect_fail $PY main.py detect --file missing.wav 2>&1 | grep "error"
echo
echo "\$ python main.py            # no command given"
expect_fail $PY main.py 2>&1 | grep "error"

# ---------------------------------------------------------------- 8. GUI
banner "STEP 8 — CODING COBRAS web GUI" \
       "Launching the local web interface headless to prove it starts, then shutting it down."
GUI_PORT="${GUI_PORT:-7865}"
$PY main.py gui --no_browser --port "$GUI_PORT" >/tmp/vocaltrace_gui.log 2>&1 &
GUI_PID=$!
GUI_OK=0
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$GUI_PORT/" >/dev/null 2>&1; then
        GUI_OK=1; break
    fi
    kill -0 "$GUI_PID" 2>/dev/null || break
    sleep 1
done
if [ "$GUI_OK" = "1" ]; then
    printf '  \033[1;32mGUI is live\033[0m at http://127.0.0.1:%s (CODING COBRAS branding, 4 tabs)\n' "$GUI_PORT"
else
    printf '  \033[2m(GUI health-check skipped — see /tmp/vocaltrace_gui.log)\033[0m\n'
fi
kill "$GUI_PID" 2>/dev/null
wait "$GUI_PID" 2>/dev/null
printf '  To open it interactively:  \033[1mpython main.py gui\033[0m\n'

printf '\n\033[1;36mDemo complete.\033[0m  Full logs: outputs/vocaltrace.log\n'
printf '\033[2mBack-up talking points if asked:\033[0m\n'
cat <<'EOF'
  • "Why phase?"     — vocoders match the magnitude spectrum well enough to fool
                       humans; they don't reconstruct phase. README "How it works".
  • "Why XGBoost?"   — data-frugal + auditable. The 1-D CNN is also implemented
                       (--model cnn) but honestly measured to overfit below ~200
                       windows; the trainer warns about it. README limitation #3.
  • "Limitations?"   — detects vocoder artefacts, not "AI": weak on replays and
                       8 kHz phone audio; retrain on ASVspoof for real use;
                       one risk signal among many, human in the loop. README #1-7.
  • "Privacy?"       — fully on-device; store verdict rows, not audio. GDPR/DPDP.
  • "The GUI?"       — `python main.py gui`: local-only web interface, CODING
                       COBRAS branding, single/batch/train tabs, ~11 ms verdicts.
EOF
