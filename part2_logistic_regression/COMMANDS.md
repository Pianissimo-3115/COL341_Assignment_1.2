# Part 2 — command reference

Run everything from `part2_logistic_regression/`:

```bash
cd D:/Study/7Sem/COL341/Assignments/Assignment1/part2_logistic_regression
```

**`python` vs `python3` on this machine.** `python3` resolves to the Windows
Store stub here and will fail — use `python` locally. The assignment spec and
the graders use `python3`; both work on Kaggle/Linux, so the submitted files are
written against `python3` and only the local commands below say `python`.

If `python` is not found, use the full path:
`C:/Users/Nihal/AppData/Local/Programs/Python/Python312/python.exe`

---

## Part (a) — gradient-descent variants

Four methods, each producing its own predictions/weights pair:

```bash
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv full_batch predictions_full_batch.txt weights_full_batch.txt
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv mini_batch predictions_mini_batch.txt weights_mini_batch.txt
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv sgd        predictions_sgd.txt        weights_sgd.txt
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv adagrad    predictions_adagrad.txt    weights_adagrad.txt
```

Two extra optional arguments (**not** used by the autograder) dump the
train/validation loss curve needed for the report's Part (a) plots:

```bash
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv adagrad predictions_adagrad.txt weights_adagrad.txt drive_data/part_ab_val.csv loss_curve_adagrad.csv
```

## Part (b) — class imbalance

```bash
python part_b.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv baseline      predictions_baseline.txt      weights_baseline.txt
python part_b.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv classweight   predictions_classweight.txt   weights_classweight.txt
python part_b.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv classweight2  predictions_classweight2.txt  weights_classweight2.txt
python part_b.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv focal         predictions_focal.txt         weights_focal.txt
```

## Verify Parts (a) and (b) against the reference weights

Trains all 8 methods and diffs epochs 1–5 against `drive_data/weight_traces/`:

```bash
python verify_weights.py
```

Expected: `40/40 snapshots matched`. (Small transient loss differences in the
four AdaGrad-based methods' first epochs are expected and explained in
`../CONTEXT.md`; the weights themselves match to ~1e-10.)

---

## Part (c) — the five versions

Each version is self-contained and takes the **same three arguments**. Replace
`dataset_dir/` with the real Kaggle data folder (the one containing
`train.csv`, `val.csv`, `test.csv` and `raw_signals/`).

```bash
# v1  392 supplied columns, cleaned
python variants/part_c_v1_baseline.py dataset_dir/ model_v1.pkl final_features_v1.csv

# v2  + out-of-fold AUC feature selection
python variants/part_c_v2_selected.py dataset_dir/ model_v2.pkl final_features_v2.csv

# v3  + raw-signal ECG features            <- expected best
python variants/part_c_v3_raw.py      dataset_dir/ model_v3.pkl final_features_v3.csv

# v4  + cubic-spline basis (GAM)
python variants/part_c_v4_gam.py      dataset_dir/ model_v4.pkl final_features_v4.csv

# v5  picks among v1–v4 by patient-grouped CV
python variants/part_c_v5_auto.py     dataset_dir/ model_v5.pkl final_features_v5.csv
```

The top-level `part_c.py` is the multi-variant build. It runs with the graded
three-argument form and additionally accepts an optional 4th argument:

```bash
python part_c.py dataset_dir/ model.pkl final_features.csv            # defaults to raw
python part_c.py dataset_dir/ model.pkl final_features.csv given
python part_c.py dataset_dir/ model.pkl final_features.csv gam
```

### Choosing a version to submit

The submission is a single `part_c.py`, so copy the winner over it:

```bash
cp variants/part_c_v3_raw.py part_c.py
python part_c.py dataset_dir/ model.pkl final_features.csv
```

### Score a version

`evaluate_partc.py` aligns features to the label file **by `release_id`**, so it
only works when `final_features.csv` covers exactly the rows you are scoring.
`part_c.py` writes features for `test.csv` rows only, which has a trap:

```bash
# WRONG - features are for test rows, labels are val rows.
python drive_data/evaluate_partc.py dataset_dir/val.csv model_v3.pkl final_features_v3.csv
#   WARNING: 827 test release_ids from val.csv have no matching row
#   M = 0.0000        <- not a score, just zero overlap
```

That `M = 0.0000` looks like a real result but means nothing matched. So:

**To get validation metrics on the real data, use the report tool** — it scores
val properly, through the same pipeline:

```bash
python dev/partc_report.py dataset_dir/ report_out/ raw
# then read the "(iv) Final metrics on val.csv" table in report_out/report_tables.md
```

**`evaluate_partc.py` is for the synthetic dataset**, which ships matching
`test_labels.csv`, and for confirming the model.pkl / final_features.csv pair is
self-consistent before submitting:

```bash
python drive_data/evaluate_partc.py ../synth_data/test_labels.csv m.pkl f.csv
```

Real test labels are never released; the leaderboard score comes from Kaggle.

### Build the Kaggle submission

```bash
python drive_data/partc_kaggle.py model_v3.pkl final_features_v3.csv submission.csv
```

### Keep or drop the flagged patients

All six Part (c) files define, near the top:

```python
EXCLUDE_PATIENTS = ("045", "051")
```

Set it to `()` to keep them, then rerun. Compare both ways before submitting —
dropping 051 from val tunes the threshold on an easier distribution than the
private test set.

---

## Dev tools (not submitted)

### Synthetic Part (c) dataset

Builds data in the exact Part (c) layout so the pipeline can be exercised
without the private Kaggle data. Second argument is the patient count:

```bash
python dev/make_synthetic_partc_data.py ../synth_data/ 60
```

Produces `train.csv`, `val.csv`, `test.csv`, `test_labels.csv` and
`raw_signals/*.npz`. Then, for example:

```bash
python variants/part_c_v3_raw.py ../synth_data/ m.pkl f.csv
python drive_data/evaluate_partc.py ../synth_data/test_labels.csv m.pkl f.csv
```

### Report figures and tables

Generates every Part (c) report deliverable — `m_vs_threshold.png`,
`epsilon_sweep.png` and `report_tables.md` (top-10 features, class-imbalance
comparison, final metrics vs the 392-column baseline):

```bash
python dev/partc_report.py dataset_dir/ report_out/ raw
```

Requires `matplotlib` (already installed). The variant argument is optional and
defaults to `raw`; pass `given` to produce the baseline comparison figures.

---

## Full sequence for a fresh run on the real data

```bash
cd D:/Study/7Sem/COL341/Assignments/Assignment1/part2_logistic_regression

# 1. sanity-check parts (a) and (b) still reproduce the reference weights
python verify_weights.py

# 2. run all five Part (c) versions on the real Kaggle folder
for v in v1_baseline v2_selected v3_raw v4_gam v5_auto; do
  python "variants/part_c_$v.py" dataset_dir/ "model_$v.pkl" "features_$v.csv"
done

# 3. compare them on val (report_out/report_tables.md, table "(iv)")
python dev/partc_report.py dataset_dir/ report_out/ raw
python dev/partc_report.py dataset_dir/ report_out_given/ given

# 4. pick the winner and build the submission
cp variants/part_c_v3_raw.py part_c.py
python drive_data/partc_kaggle.py model_v3_raw.pkl features_v3_raw.csv submission.csv
```

Step 2's loop is bash (works in Git Bash here). In PowerShell:

```powershell
foreach ($v in "v1_baseline","v2_selected","v3_raw","v4_gam","v5_auto") {
  python "variants/part_c_$v.py" dataset_dir/ "model_$v.pkl" "features_$v.csv"
}
```
