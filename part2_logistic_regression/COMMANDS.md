# Part 2 — command reference

Every command below is in the **exact form the assignment PDF specifies**. Where
a command runs one of the alternative Part (c) versions, only the *script
filename* differs from the PDF; the arguments are unchanged.

**Two local adaptations**, and nothing else:

1. `python3` resolves to the Windows Store stub on this machine, so substitute
   `python` when running locally. The graders and Kaggle use `python3`, which is
   what the commands are written as here.
2. The PDF assumes the input CSVs sit in the working directory. Here they live
   in `drive_data/`, so either `cd drive_data` first or prefix those two
   filenames with `drive_data/`. Everything else is identical.

Run from `part2_logistic_regression/`.

---

## Part (a) — gradient-descent variants

The evaluator runs it once per method (PDF p. 17):

```bash
python3 part_a.py part_ab_train.csv part_ab_test_public.csv full_batch \
predictions_full_batch.txt weights_full_batch.txt
python3 part_a.py part_ab_train.csv part_ab_test_public.csv mini_batch \
predictions_mini_batch.txt weights_mini_batch.txt
python3 part_a.py part_ab_train.csv part_ab_test_public.csv sgd \
predictions_sgd.txt weights_sgd.txt
python3 part_a.py part_ab_train.csv part_ab_test_public.csv adagrad \
predictions_adagrad.txt weights_adagrad.txt
```

`weights.txt` must contain exactly 79 lines: the bias (3 comma-separated values
for classes N, A, O) followed by the 78 rows of `W`.

Locally, with the CSVs under `drive_data/`:

```bash
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv adagrad \
predictions_adagrad.txt weights_adagrad.txt
```

### Loss curves for the report

`part_a.py` accepts two **optional** extra arguments beyond the graded
interface. They are ignored by the evaluator and only exist to dump the
train/validation loss curve the report's Part (a) plots need:

```bash
python part_a.py drive_data/part_ab_train.csv drive_data/part_ab_test_public.csv adagrad \
predictions_adagrad.txt weights_adagrad.txt drive_data/part_ab_val.csv loss_curve_adagrad.csv
```

## Part (b) — class imbalance

The evaluator runs it once per method (PDF p. 18):

```bash
python3 part_b.py part_ab_train.csv part_ab_test_public.csv baseline \
predictions_baseline.txt weights_baseline.txt
python3 part_b.py part_ab_train.csv part_ab_test_public.csv classweight \
predictions_classweight.txt weights_classweight.txt
python3 part_b.py part_ab_train.csv part_ab_test_public.csv classweight2 \
predictions_classweight2.txt weights_classweight2.txt
python3 part_b.py part_ab_train.csv part_ab_test_public.csv focal \
predictions_focal.txt weights_focal.txt
```

Output file formats are identical to Part (a).

## Part (c) — feature engineering

The submitted file is run as (PDF p. 19):

```bash
python3 part_c.py dataset_dir/ model.pkl final_features.csv
```

`dataset_dir` is the folder holding `train.csv`, `val.csv`, `test.csv` and
`raw_signals/`.

### The five versions

Each takes the same arguments in the same order — only the filename changes:

```bash
python3 variants/part_c_v1_baseline.py dataset_dir/ model.pkl final_features.csv
python3 variants/part_c_v2_selected.py dataset_dir/ model.pkl final_features.csv
python3 variants/part_c_v3_raw.py      dataset_dir/ model.pkl final_features.csv
python3 variants/part_c_v4_gam.py      dataset_dir/ model.pkl final_features.csv
python3 variants/part_c_v5_auto.py     dataset_dir/ model.pkl final_features.csv
```

The output paths are ordinary command-line arguments, so when comparing versions
side by side substitute distinct names rather than overwriting one file:

```bash
python3 variants/part_c_v3_raw.py dataset_dir/ model_v3.pkl final_features_v3.csv
```

Whichever version you submit must be named `part_c.py`:

```bash
cp variants/part_c_v3_raw.py part_c.py
python3 part_c.py dataset_dir/ model.pkl final_features.csv
```

### Kaggle submission

Per the PDF (p. 20), from the `model.pkl` and `final_features.csv` just written:

```bash
python3 partc_kaggle.py model.pkl final_features.csv submission.csv
```

The copy shipped with the course files is at `drive_data/partc_kaggle.py`:

```bash
python drive_data/partc_kaggle.py model.pkl final_features.csv submission.csv
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

## Helper scripts (course-supplied and dev; not submitted)

These are not specified in the PDF's command-line requirements, so they keep
their own interfaces. Written with `python` since they only ever run locally.

### Score a model

`evaluate_partc.py` (course-supplied) aligns features to labels **by
`release_id`**, and `part_c.py` writes features for `test.csv` rows only. So
pointing it at `val.csv` matches nothing and prints a score that is not a score:

```bash
# WRONG - features are for test rows, labels are val rows
python drive_data/evaluate_partc.py dataset_dir/val.csv model.pkl final_features.csv
#   WARNING: 827 test release_ids from val.csv have no matching row
#   M = 0.0000        <- zero overlap, not a result
```

Use it where the rows do line up — the synthetic dataset, which ships a matching
`test_labels.csv`, or to confirm a `model.pkl` / `final_features.csv` pair is
self-consistent before submitting:

```bash
python drive_data/evaluate_partc.py ../synth_data/test_labels.csv model.pkl final_features.csv
```

For validation metrics on the real data use the report tool below; real test
labels are never released.

### Verify Parts (a) and (b) against the reference weights

Trains all 8 methods and diffs epochs 1–5 against `drive_data/weight_traces/`:

```bash
python verify_weights.py
```

Expected: `40/40 snapshots matched`.

### Report figures and tables

Produces `m_vs_threshold.png`, `epsilon_sweep.png` and `report_tables.md` (top-10
features, class-imbalance comparison, final val metrics vs the 392-column
baseline):

```bash
python dev/partc_report.py dataset_dir/ report_out/ raw
```

### Synthetic Part (c) dataset

Builds data in the exact Part (c) layout, so the pipeline can be exercised
before the real Kaggle data is available:

```bash
python dev/make_synthetic_partc_data.py ../synth_data/ 60
python variants/part_c_v3_raw.py ../synth_data/ model.pkl final_features.csv
python drive_data/evaluate_partc.py ../synth_data/test_labels.csv model.pkl final_features.csv
```

---

## Full sequence on the real data

```bash
cd D:/Study/7Sem/COL341/Assignments/Assignment1/part2_logistic_regression

# 1. confirm parts (a) and (b) still reproduce the reference weights
python verify_weights.py

# 2. run all five Part (c) versions, keeping their outputs separate
for v in v1_baseline v2_selected v3_raw v4_gam v5_auto; do
  python "variants/part_c_$v.py" dataset_dir/ "model_$v.pkl" "final_features_$v.csv"
done

# 3. compare on val (report_out/report_tables.md, table "(iv)")
python dev/partc_report.py dataset_dir/ report_out/ raw

# 4. submit the winner under the required name, then build the Kaggle file
cp variants/part_c_v3_raw.py part_c.py
python part_c.py dataset_dir/ model.pkl final_features.csv
python drive_data/partc_kaggle.py model.pkl final_features.csv submission.csv
```

PowerShell equivalent of step 2:

```powershell
foreach ($v in "v1_baseline","v2_selected","v3_raw","v4_gam","v5_auto") {
  python "variants/part_c_$v.py" dataset_dir/ "model_$v.pkl" "final_features_$v.csv"
}
```
