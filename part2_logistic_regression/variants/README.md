# Part 2(c) — solution versions

Five independent solutions to the AF-detection task. Each file is **complete and
self-contained**: the assignment submits a single `part_c.py`, so a version is
used by copying it over the top-level file.

```bash
cp variants/part_c_v3_raw.py part_c.py     # pick a version to submit
python3 part_c.py dataset_dir/ model.pkl final_features.csv
```

Every version takes the same three arguments, writes the same two outputs, and
is scored the same way: `sigmoid(phi(x) . weights + bias) >= threshold`.

## Commands

Run from `part2_logistic_regression/`. Use `python` locally — `python3` resolves
to the Windows Store stub on this machine.

```bash
python variants/part_c_v1_baseline.py dataset_dir/ model_v1.pkl final_features_v1.csv
python variants/part_c_v2_selected.py dataset_dir/ model_v2.pkl final_features_v2.csv
python variants/part_c_v3_raw.py      dataset_dir/ model_v3.pkl final_features_v3.csv
python variants/part_c_v4_gam.py      dataset_dir/ model_v4.pkl final_features_v4.csv
python variants/part_c_v5_auto.py     dataset_dir/ model_v5.pkl final_features_v5.csv
```

Compare them on val, then build the Kaggle submission from the winner:

```bash
python dev/partc_report.py dataset_dir/ report_out/ raw   # table (iv) = val metrics
python drive_data/partc_kaggle.py model_v3.pkl final_features_v3.csv submission.csv
```

See [`../COMMANDS.md`](../COMMANDS.md) for the full reference, including Parts
(a) and (b), the synthetic-data workflow, and a trap in `evaluate_partc.py` that
silently reports `M = 0.0000` when the feature rows do not match the label rows.

| # | file | phi(x) | when it wins |
|---|---|---|---|
| 1 | `part_c_v1_baseline.py` | the 392 supplied columns, cleaned | few training patients; the supplied columns already separate the classes |
| 2 | `part_c_v2_selected.py` | + out-of-fold AUC ranking to the best 220 | many redundant/noisy columns relative to training rows |
| 3 | `part_c_v3_raw.py` | + ECG features computed from `raw_signals/` | **the expected winner** — the supplied columns miss atrial activity and cannot separate noise from genuine irregularity |
| 4 | `part_c_v4_gam.py` | + cubic-spline basis on the 45 strongest features | strong features that act non-monotonically (e.g. heart rate, where both fast and slow are suspicious) |
| 5 | `part_c_v5_auto.py` | picks among 1–4 by patient-grouped CV | when you would rather not guess; costs the sum of the others' runtime |

## What versions 3–5 actually compute

Versions 1 and 2 only reshuffle the supplied columns. The supplied 392 come from
the published Goodfellow / `ecg-features` library and are dominated by RR/HRV
statistics, so they largely miss AF's *other* defining sign — the loss of
organised atrial activity — and they cannot tell a genuinely irregular rhythm
from a noisy recording. Versions 3–5 go back to the raw signal for:

- **Rhythm irregularity** — Pan-Tompkins QRS detection, then COSEn (Lake &
  Moorman), symbolic-dynamics Shannon entropy (Zhou et al.), Lorenz-plot dRR
  occupancy (Sarkar et al.), Poincaré SD1/SD2, turning-point ratio.
- **Atrial activity** — zero-padding QRST cancellation: subtract an average beat
  at every R peak, then measure P-wave amplitude and beat-to-beat
  reproducibility, plus 4–9 Hz fibrillatory-wave band power in the TQ intervals.
- **Signal quality** — these exist to *prevent* false positives. Noise causes
  spurious QRS detections, which look exactly like the irregular RR series that
  means AF. At 100× cost per false positive, letting the model see that a
  recording is untrustworthy matters more than any classifier tweak.

## Design constraints all five respect

`M = TP/P − FP/(N·ε)` with `ε = 1/100`, i.e. **`M = TPR − 100·FPR`**. One false
positive costs as much as `100/N` of recall, so every version:

- winsorises features to train-set quantiles, so one wild outlier at test time
  cannot produce a confident false positive;
- uses **patient-grouped** CV everywhere (rows within a patient are correlated);
- picks the *largest* threshold whose M is within `0.01` of the best rather than
  the raw arg-max, which sits exactly where a couple of negatives happen to fall
  below the cut;
- fits every statistic on train only, keeps `d < 500`, and never touches
  `test.csv` row membership.

## Excluded patients

Per the course announcement that validation performance looks unexpectedly poor
"particularly for patient 051 in val.csv and 045 in train.csv", all five define:

```python
EXCLUDE_PATIENTS = ("045", "051")
```

These rows are dropped from **training and threshold selection only** — never
from `test.csv`, where every row must still be scored. ID matching is
format-tolerant (`45`, `"045"`, `"P045"`, `"patient_045"` all match). Set the
tuple to `()` to keep them.

Two caveats worth weighing before submitting:

1. **Dropping 045 from training is the safe half.** The assignment explicitly
   invites it ("some patients or recordings may behave as systematic outliers …
   out-of-fold predictions can be useful for identifying patients that
   consistently contribute a disproportionate number of errors").
2. **Dropping 051 from val is the risky half.** val is what the threshold is
   tuned on, and the private test set will still contain hard patients. Tuning
   on an easier val can yield a threshold that is too permissive, and at 100×
   per false positive that is the expensive direction to be wrong in. If the
   final M looks better only because the hard patient is gone, that gain is not
   real. Compare both ways before committing.

## Validation status

Verified end-to-end on a synthetic dataset in the exact Part (c) layout
(`../dev/make_synthetic_partc_data.py`), because the real data is a private
Kaggle dataset. On 60 synthetic patients / 4015 windows, held-out test M was
−0.21 (v1), −0.44 (v2), 1.00 (v3), 1.00 (v4), 1.00 (v5), with the whole v3 run
taking ~22 s against a 40-minute limit.

**These numbers are synthetic.** They show the pipeline is correct and that the
raw-signal features carry real discriminative power. They are *not* an estimate
of leaderboard performance and must not be quoted in the report.
