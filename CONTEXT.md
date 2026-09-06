# Session Context — COL774 Assignment 1

Working directory: `D:\Study\7Sem\COL341\Assignments\Assignment1`
This file is a snapshot of where things stand, so a future session (or you) can pick
up without re-deriving everything.

## What's in this folder

- `26M_COL774_Assignment_1__Linear_Logistic_Regression_ v2.pdf` — the assignment spec
  (Part 1: Linear Regression, Part 2: Logistic Regression). Kept local only — **not**
  pushed anywhere (see Copyright note below).
- `part2_logistic_regression/` — all Part 2 work (this session's focus). Part 1
  (linear regression) is reportedly already done elsewhere, not in this folder.
- `.git/`, `.gitignore` — this folder is a git repo, pushed to GitHub (see below).
- `CONTEXT.md` — this file.

## Environment set up this session

- Installed **Python 3.12.10** via `winget` (`Python.Python.3.12`), plus `numpy`,
  `pandas`, `scikit-learn`, `gdown`. Now on the user PATH as `python` / `python3`.
- Installed **GitHub CLI** (`gh` 2.100.0) via `winget`. Authenticated as GitHub user
  **`Pianissimo-3115`** (SSH protocol, key at `~/.ssh/id_ed25519`).

## GitHub repo

**https://github.com/Pianissimo-3115/COL341_Assignment_1.2** — public, code-only.

Contains exactly: `part_a.py`, `part_b.py`, `part_c.py`, `verify_weights.py`,
`.gitignore` (from `part2_logistic_regression/`).

**Deliberately excluded** from the repo (per copyright/academic-integrity discussion —
the assignment PDF explicitly forbids replicating assignment content anywhere):
the assignment PDF itself, and all downloaded datasets
(`part2_logistic_regression/drive_data/`). These remain local-only.

## Part 2 (Logistic Regression) — status

All code lives in `part2_logistic_regression/`.

### Part (a) — `part_a.py` — DONE, verified correct
Multinomial softmax logistic regression, 4 optimizers (full-batch, mini-batch, SGD,
AdaGrad) exactly per spec (float64, stabilized softmax clipped to [-60,0],
`default_rng(774)` per-epoch permutation, per-batch gradient averaging).
CLI: `python3 part_a.py train.csv test.csv {full_batch,mini_batch,sgd,adagrad}
predictions.txt weights.txt [val.csv] [loss_curve.csv]` (last two args optional, for
report plots only, not used by the autograder).

### Part (b) — `part_b.py` — DONE, verified correct
Class-imbalance handling on the AdaGrad protocol: `baseline`, `classweight`
(inverse-frequency), `classweight2` (power 0.3), `focal` (γ=2, α'=α^0.5).
CLI: `python3 part_b.py train.csv test.csv {baseline,classweight,classweight2,focal}
predictions.txt weights.txt`.

### Verification — `verify_weights.py` (dev tool, not part of the graded submission)
Trains all 8 methods on the real `part_ab_train.csv` and diffs epoch-1..5 weight
snapshots against the reference traces in `drive_data/weight_traces/`.
**Result: 40/40 snapshots match to ~1e-10** (essentially exact). Minor
self-correcting loss-value drift (≤0.007, gone by epoch 4-5) seen only for the 4
AdaGrad-based methods in early epochs — expected AdaGrad floating-point sensitivity
near-zero `G`, not a correctness issue (weights themselves match exactly).
Run it with: `python verify_weights.py` (from `part2_logistic_regression/`, needs
`drive_data/` present locally).

### Part (c) — five separate versions in `variants/`, validated on synthetic data
The submission is a single file, so each version is **complete and self-contained**;
you pick one by copying it over the top-level `part_c.py`. See
`part2_logistic_regression/variants/README.md` for the full comparison.

| # | file | phi(x) | synthetic test M |
|---|---|---|---|
| 1 | `part_c_v1_baseline.py` | the 392 supplied columns, cleaned | -0.21 |
| 2 | `part_c_v2_selected.py` | + out-of-fold AUC ranking to the best 220 | -0.44 |
| 3 | `part_c_v3_raw.py` | + features computed from `raw_signals/` | **1.00** |
| 4 | `part_c_v4_gam.py` | + cubic-spline basis on the 45 strongest | **1.00** |
| 5 | `part_c_v5_auto.py` | picks among 1-4 by patient-grouped CV | **1.00** |

Top-level `part_c.py` is currently the multi-variant build (defaults to `raw`, and
still accepts an optional 4th arg to switch); `variants/part_c_v3_raw.py` is the
same strategy as a clean single-purpose file. Swap in a version with
`cp variants/part_c_v3_raw.py part_c.py`.

**Excluded patients.** Per the course announcement (validation looks unexpectedly
poor "particularly for patient 051 in val.csv and 045 in train.csv"), all six files
define `EXCLUDE_PATIENTS = ("045", "051")`, dropped from **training and threshold
selection only** — never from `test.csv`, where every row must still be scored. ID
matching is format-tolerant (`45`, `"045"`, `"P045"`, `"patient_045"` all match;
unit-tested). Set the tuple to `()` to keep them. Note the asymmetry: dropping 045
from *training* is explicitly invited by the assignment's "noisy patients" section,
but dropping 051 from *val* tunes the threshold on an easier distribution than the
private test set, which at 100x per false positive is the expensive direction to be
wrong in — worth comparing both ways before submitting.

Raw-signal features (the part that earns marks) re-implement published AF detectors:
Pan-Tompkins QRS detection; RR irregularity (COSEn of Lake & Moorman, symbolic-
dynamics Shannon entropy of Zhou et al., Sarkar-style Lorenz-plot dRR occupancy,
Poincare SD1/SD2, TPR); atrial activity via zero-padding QRST cancellation, giving
P-wave amplitude/beat-to-beat reproducibility and 4-9 Hz fibrillatory-wave band
power in the TQ segments; plus **signal-quality** features, which exist specifically
to stop noisy strips producing spurious QRS detections that mimic AF irregularity.

Design driven by the metric `M = TPR - 100*FPR`: one FP costs ~100/N, so the
pipeline winsorises to train quantiles (an outlier must not create a confident FP),
uses patient-grouped CV everywhere, and picks the *largest* threshold within 0.01 of
the best M rather than the raw arg-max.

**Validated end-to-end** with `dev/make_synthetic_partc_data.py`, which fabricates a
dataset in the exact Part (c) layout (the real data is a private Kaggle Dataset that
has not been downloaded). On 60 synthetic patients / 4015 windows: `given` scored
M = -1.96 on held-out test, `raw` and `gam` scored M = 1.00; whole run took **22 s**,
versus the 40-minute Kaggle limit. Contract checks pass: `d < 500`,
`len(weights) == d`, finite features, and `evaluate_partc.py` reproduces the score.
Graceful fallbacks verified for missing `raw_signals/` and missing `val.csv`.

**Caveat:** those numbers come from *synthetic* data and say only that the pipeline
is correct and discriminative — they are not an estimate of real leaderboard
performance and must not be quoted in the report.

**Still TODO:** run against the real Kaggle data, then investigate noisy patients /
label quality via out-of-fold errors, and re-tune `SELECT_DIMS` / `C_GRID`.

### Dev tools (not submitted)
- `dev/make_synthetic_partc_data.py out_dir/ [n_patients]` — synthetic Part (c)
  dataset (train/val/test + `raw_signals/*.npz` + `test_labels.csv`).
- `dev/partc_report.py dataset_dir/ out_dir/ [variant]` — produces every report
  deliverable: `m_vs_threshold.png`, `epsilon_sweep.png` (1/eps swept over
  [10,1000]) and `report_tables.md` (top-10 features, class-imbalance comparison,
  final metrics vs the 392-column baseline). Needs `matplotlib` (installed).

### Not started
- `report.pdf` (required alongside the code; max 4 pages — loss plots for Part (a),
  feature pipeline + required analysis/plots for Part (c)). `dev/partc_report.py`
  already generates the Part (c) figures and tables; they just need writing up, and
  must be regenerated on the real data first.
- Part 1 (Linear Regression) — said to be complete already, but not present in this
  folder; location unknown to this session.

**Report must disclose AI use** (the assignment explicitly permits AI tools and
open-source code "as long as you disclose it"). `part_c.py`'s module docstring
carries the disclosure text to copy in.

## `drive_data/` contents (local only, gitignored)

Downloaded from the course Google Drive folder
(`https://drive.google.com/drive/folders/1Y0wh-2al6jr_kvt3ZSjEsF54TMhJlUyN`):
- `weight_traces/part_a/`, `weight_traces/part_b/` — reference weight snapshots
  (epochs 1-5) for all 8 methods, used by `verify_weights.py`.
- `weight_traces/loss_by_epoch.csv` — reference training loss per epoch (1-5).
- `part_ab_train.csv`, `part_ab_val.csv`, `part_ab_test_public.csv` — the real Part
  (a)/(b) data.
- `evaluate_partc.py`, `partc_kaggle.py`, `FEATURE_DESCRIPTIONS.md`,
  `kaggle_sample_submission.csv` — Part (c) helper files.

## Known gotchas for next time

- `python3` (no suffix) still resolves to the Windows Store stub on this machine;
  use `python` instead, or the full path
  `C:\Users\Nihal\AppData\Local\Programs\Python\Python312\python.exe`.
- `gh` needed `C:\Program Files\GitHub CLI` on PATH — already machine-wide via
  winget, but a terminal opened *before* the install won't see it until restarted.
- Git push over SSH needed GitHub's host key added to `~/.ssh/known_hosts`
  (`ssh-keyscan -t ed25519 github.com`) — already done.
