# VocalTrace — Demonstration Guide (Simple English)

**Made by CODING COBRAS**

This document shows you how to demonstrate VocalTrace, step by step. Every
sentence is short and simple. You can read it out loud while you show the
project.

---

## 1. What is the problem?

Scammers can now copy a person's voice. They only need a short recording —
even a 20-second voice note is enough. Then they call a company and pretend
to be the boss. They ask the finance team to send money fast.

This already happened. In 2024, a company in Hong Kong lost **25 million US
dollars** because of one fake voice call.

Voice calls are used everywhere — banks, offices, hospitals. So we need a
tool that can tell: **"Is this a real human voice, or is it a computer copy?"**

---

## 2. What does VocalTrace do?

VocalTrace listens to a short piece of audio and gives one word back:

- **REAL** — this sounds like a real human, or
- **FAKE** — this sounds like a computer-generated copy.

It also gives a **confidence number** (how sure it is) and two simple scores
that explain *why*. It works on your own computer. No internet is needed. No
audio is sent anywhere.

It is very fast: about **11 milliseconds** for a 2-second audio window. A
human needs several seconds to even listen to it.

---

## 3. How does it work? (the simple explanation)

When a real person speaks, air comes out of the lungs, the vocal folds vibrate,
and the mouth shapes the sound. The sound waves are **smooth** — one moment
flows into the next.

A computer that clones a voice does not really "speak". It builds the sound
from numbers, piece by piece. The loudness part looks almost perfect. But the
**timing part (called "phase")** gets mixed up. Between two moments in time,
the phase jumps suddenly instead of moving smoothly.

VocalTrace measures these jumps:

```
real voice   →  smooth phase  →  low jump score  →  verdict: REAL
cloned voice →  jumpy phase   →  high jump score →  verdict: FAKE
```

It looks at three groups of measurements:

| # | What we measure | What it catches |
|---|-----------------|-----------------|
| 1 | Shape of the voice (MFCC) | How the mouth and throat shape the sound |
| 2 | **Phase jumps** (the main one) | The broken timing that clones leave behind |
| 3 | High-frequency energy | Clones often lose the sharp, bright sounds |

A small model called **XGBoost** looks at these measurements and decides REAL
or FAKE. The model runs on **ONNX Runtime**, which is fast and works on any
computer — no GPU needed.

---

## 4. Before the demo — prepare (1 minute)

Open Terminal and type:

```bash
cd vocaltrace
source .venv/bin/activate
```

Check that everything works:

```bash
python -m pytest tests/ -q
```

You should see: **58 passed**.

> Tip: do this once before the demo so the first run is not slow. The very
> first audio check takes about 0.6 seconds extra (the program warms up).
> After that it is instant.

---

## 5. The live demonstration (7 steps)

Do these steps in order. Say the sentence next to each one.

### Step 1 — Show a FAKE voice

```bash
python main.py detect --file data/fake/sample1.wav
```

**Say:** *"This is a computer-made copy of a voice. Watch the verdict."*

**You will see:**

```
verdict   : FAKE  (confidence 0.961)
P(fake)   : 0.9609   threshold 0.50
phase disc: 0.4424 (higher = more vocoder-like)
latency   : 11.1 ms
```

**Say:** *"It is 96% sure this is fake. The phase score is high — that is the
broken timing I talked about. And it took 11 milliseconds."*

### Step 2 — Show a REAL voice

```bash
python main.py detect --file data/real/sample3.wav
```

**Say:** *"Same command, but this time it is a real human recording."*

**You will see:** `verdict : REAL (confidence 0.974)`

**Say:** *"The verdict changed. The scores went down. Same model, same
command — only the audio is different."*

### Step 3 — Show the batch mode (the bank use-case)

```bash
python main.py detect --directory data/fake --output outputs/fake_results.csv
```

**Say:** *"A bank records thousands of calls. It cannot check them one by one.
So we screen a whole folder and save the results into a CSV file. The bank
system can read that file."*

**You will see:** `verdicts : REAL=0  FAKE=10` — all 10 fake files were found.

Then try the real folder:

```bash
python main.py detect --directory data/real --output outputs/real_results.csv
```

**You will see:** `verdicts : REAL=9  FAKE=1`

**Say:** *"10 out of 10 fakes were caught, and 9 out of 10 real voices passed.
Our target was 8 out of 10 for both — so we passed the test. One real file was
borderline, and I report that honestly instead of hiding it."*

### Step 4 — Show the graph (open the GUI)

```bash
python main.py gui
```

The browser opens at `http://127.0.0.1:7860`. You will see the **CODING
COBRAS** logo and four tabs.

**Say:** *"This is the graphical interface. It runs only on my computer — no
audio goes to any server."*

Show these tabs:

1. **Single file / mic** — drag a WAV file in (or record your own voice with
   the microphone) and click **Analyse**. The big card turns green (REAL) or
   red (FAKE).
2. **Batch folder** — type a folder path, click **Screen folder**, and
   download the CSV.
3. **Train model** — trains the model with one click and shows the results.
4. **About** — the explanation and the limits of the tool.

**Say:** *"The microphone tab is important for vishing defence — a bank agent
can speak live into the tool during a call."*

Close the GUI with `Ctrl + C` in the Terminal.

### Step 5 — Train the model in front of them

```bash
python main.py train --real_dir data/real --fake_dir data/fake --model xgb --export_onnx
```

**Say:** *"I am not shipping a model I downloaded. Let me train it from
scratch, right now, in front of you."*

**You will see:**

```
accuracy  : 1.0000
roc_auc   : 1.0000
onnx     : .../models/vocaltrace.onnx (max |Δ| vs source = 7.64e-08, tolerance 1e-5)
```

**Say:** *"Perfect scores on the test data. And look at this line — the ONNX
file is checked against the trained model. The difference is 0.00000007, way
below our tolerance. So the file we ship is provably the same model. That
matters for audit and compliance."*

Also mention: **the random seed is fixed at 42**, so anyone who runs the same
command gets the same model. This makes the result *reproducible*.

### Step 6 — Run the test suite

```bash
python -m pytest tests/ -q
```

**You will see:** `58 passed`

**Say:** *"58 automatic tests. They check the features, the detector, the
command line, the error handling, the GUI, and the logo files. The tests
create their own tiny dataset — they do not depend on my machine, so they pass
on any computer."*

### Step 7 — Show that errors are handled cleanly

```bash
python main.py detect --file missing.wav
python main.py detect --mic --duration 1
python main.py
```

**Say:** *"Bad input never crashes the program. Each one exits with code 1 and
shows a clear message. No ugly error tracebacks. The full details always go
to the log file. This is what you need in production."*

---

## 6. Summary table (show this on a slide if you have one)

| Result | Value |
|--------|-------|
| Fake voices caught | **10 / 10** |
| Real voices passed | **9 / 10** (target was ≥ 8/10) |
| Speed | **~11 ms** per 2-second window (budget: 150 ms) |
| Model match (ONNX parity) | **7.6 × 10⁻⁸** (limit: 1 × 10⁻⁵) |
| Tests | **58 passed** |
| Internet needed | **None** — runs fully offline |

---

## 7. Questions they may ask (short answers)

**Q: Why do we look at phase?**
A: Clone programs copy the *loudness* of speech very well — that is why people
get fooled. But they break the *timing* (phase) between sound pieces. We
measure exactly that. It is a physical fact, not a guess.

**Q: Is this an AI fighting AI?**
A: No. The measurement part is pure maths (signal processing). Only the final
decision uses a small, simple model. No large AI, no cloud, no GPU.

**Q: What are the limits?**
A: Three honest ones:
1. It finds *computer-made* voices. It does not find a *recording* of a real
   person played back.
2. Old phone lines cut the sound at 8 kHz and remove the high frequencies, so
   accuracy drops there. Test it on your own channel first.
3. The demo model was trained on only 20 clips. For real use, train it on a
   bigger dataset (we support ASVspoof and LibriSpeech).

**Q: Can a clever attacker beat it?**
A: Maybe — no detector is perfect. So this tool is *one layer* among many:
call-back verification, transaction limits, employee training. We never
auto-reject a payment based on this tool alone. A human always decides.

**Q: What about privacy laws (GDPR, India's DPDP)?**
A: The audio never leaves the machine. We recommend storing only the verdict
row (file name, REAL/FAKE, score) and deleting the recording. The README has
a full privacy section.

**Q: Why XGBoost and not deep learning?**
A: We built both. On small data, deep learning memorises instead of learning
— we measured it: perfect scores on training data, random scores on test
data. XGBoost is honest, fast, and explainable. The GUI's Train tab shows a
warning about this.

---

## 8. One-paragraph version (memorise this)

> *"Voice cloning is now good enough to fool people, and it has already cost
> companies millions. VocalTrace detects cloned speech by measuring a physical
> artefact that clone programs cannot hide: they break the smooth timing — the
> phase — of a real voice. The tool converts audio into three groups of
> measurements, a small model makes the REAL/FAKE decision, and everything runs
> offline on the user's own machine in about 11 milliseconds. On our demo
> corpus it catches 10 out of 10 fakes and passes 9 out of 10 real voices. It
> is built for banks and call centres — with a web interface, batch CSV
> output, reproducible training, and honest documentation of its limits."*

---

## 9. Quick command card (keep this open in a second Terminal tab)

```bash
cd vocaltrace && source .venv/bin/activate

python main.py detect --file data/fake/sample1.wav     # fake → FAKE
python main.py detect --file data/real/sample3.wav      # real → REAL
python main.py detect --directory data/fake --output outputs/results.csv
python main.py train --real_dir data/real --fake_dir data/fake --model xgb --export_onnx
python main.py gui                                      # web interface
python -m pytest tests/ -q                              # 58 passed
./demo.sh                                               # guided demo, 7+1 steps
```

**If anything breaks:** everything is logged in `outputs/vocaltrace.log`.
Stop the GUI with `Ctrl+C`. Restart the environment with the first two
commands above.
