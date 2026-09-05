"""
Dev tool (NOT part of the submission).

Builds a synthetic dataset in the exact Part (c) layout so part_c.py can be
run end-to-end before the real, private Kaggle data is available:

    dataset_dir/
        train.csv  val.csv          release_id, patient, label, f000..f391
        test.csv                    same, without label
        test_labels.csv             release_id, label  (for evaluate_partc.py)
        raw_signals/{patient}.npz   release_id (n,), signal (n, 7500) float32

The simulation is deliberately built to punish a careless pipeline the same way
the real task does:

  * patient-level AF propensity, so ungrouped CV over-estimates performance;
  * AF windows differ from normal ones by irregular RR *and* absent P waves
    plus ~6 Hz fibrillatory waves - the two real discriminators;
  * a slice of very noisy non-AF windows whose spurious QRS detections mimic
    AF irregularity, which is exactly how false positives happen in practice;
  * of the 392 "given" columns only ~30 carry signal, the rest are noise, and
    a few percent of all entries are NaN.

    python make_synthetic_partc_data.py out_dir/ [n_patients]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FS = 250
WINDOW_S = 30
N_SAMPLES = FS * WINDOW_S            # 7500
N_GIVEN = 392
N_INFORMATIVE = 30
SEED = 20260905


def beat_waveform(fs, include_p=True):
    """One PQRST complex sampled at `fs`, R peak at the returned index."""
    def gauss(t, centre, width, amp):
        return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)

    t = np.arange(-0.30, 0.45, 1.0 / fs)
    wave = np.zeros_like(t)
    if include_p:
        wave += gauss(t, -0.16, 0.022, 0.16)          # P
    wave += gauss(t, -0.022, 0.0075, -0.11)           # Q
    wave += gauss(t, 0.0, 0.0090, 1.00)               # R
    wave += gauss(t, 0.028, 0.0100, -0.24)            # S
    wave += gauss(t, 0.22, 0.048, 0.30)               # T
    return wave, int(round(0.30 * fs))


def make_window(rng, is_af, noisy):
    """One 30 s single-lead ECG window."""
    if is_af:
        # Irregularly irregular ventricular response.
        mean_rr = rng.uniform(0.55, 0.95)
        rr = rng.gamma(shape=12.0, scale=mean_rr / 12.0, size=80)
        rr = np.clip(rr * rng.uniform(0.85, 1.15, size=rr.size), 0.32, 1.9)
    else:
        mean_rr = rng.uniform(0.70, 1.10)
        rr = np.clip(rng.normal(mean_rr, 0.025, size=80), 0.40, 1.8)
        if rng.random() < 0.25:                        # occasional ectopy
            k = rng.integers(1, 4)
            pos = rng.choice(rr.size - 2, size=k, replace=False) + 1
            rr[pos] *= 0.62
            rr[pos + 1] *= 1.34

    template, r_index = beat_waveform(FS, include_p=not is_af)
    x = np.zeros(N_SAMPLES + template.size)
    cursor = rng.integers(0, int(0.6 * FS))
    for gap in rr:
        cursor += int(round(gap * FS))
        if cursor >= N_SAMPLES:
            break
        amp = rng.normal(1.0, 0.11 if is_af else 0.05)
        lo = cursor - r_index
        if lo < 0:
            continue
        x[lo:lo + template.size] += amp * template

    x = x[:N_SAMPLES]
    t = np.arange(N_SAMPLES) / FS

    if is_af:
        # Fibrillatory waves replacing organised atrial activity.
        f0 = rng.uniform(5.0, 7.5)
        x += rng.uniform(0.03, 0.09) * (
            np.sin(2 * np.pi * f0 * t + rng.uniform(0, 6.28))
            + 0.5 * np.sin(2 * np.pi * (f0 * 1.7) * t + rng.uniform(0, 6.28)))

    x += rng.uniform(0.05, 0.20) * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
    x += rng.normal(0, 0.05 if not noisy else rng.uniform(0.25, 0.55), N_SAMPLES)
    if noisy:
        # Motion artefacts: bursts that a QRS detector happily mistakes for
        # beats, producing AF-looking RR series on a non-AF recording.
        for _ in range(rng.integers(6, 16)):
            start = rng.integers(0, N_SAMPLES - 60)
            width = rng.integers(10, 55)
            x[start:start + width] += rng.normal(0, rng.uniform(0.8, 2.0), width)
    return x.astype(np.float32), rr


def given_columns(rng, rows):
    """392 supplied-style columns: a few informative, the rest noise.

    Crucially these are functions of *measured* RR irregularity, never of the
    label. That mirrors the real 392 columns, which are dominated by RR/HRV
    statistics computed by a detector that noise can fool - so a noisy non-AF
    window looks irregular here and only the raw signal reveals it is not AF.
    """
    n = len(rows)
    X = rng.normal(0, 1, size=(n, N_GIVEN))
    measured = np.array([r["rr_cv_measured"] for r in rows], dtype=float)
    measured = (measured - measured.mean()) / (measured.std() + 1e-9)
    for j in range(N_INFORMATIVE):
        strength = rng.uniform(0.30, 0.75)
        X[:, j] = strength * measured + rng.normal(0, 1.0, size=n)
    X[rng.random(X.shape) < 0.02] = np.nan            # sprinkle NaNs
    return X


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    n_patients = int(sys.argv[2]) if len(sys.argv) == 3 else 60

    rng = np.random.default_rng(SEED)
    (out_dir / "raw_signals").mkdir(parents=True, exist_ok=True)

    rows, signals_by_patient = [], {}
    for pid in range(n_patients):
        patient = f"P{pid:03d}"
        # Patients are mostly-AF or mostly-not, so rows within a patient are
        # correlated - the reason grouped CV is mandatory.
        propensity = rng.beta(0.6, 1.6)
        n_windows = int(rng.integers(35, 110))
        ids, sigs = [], []
        for w in range(n_windows):
            is_af = bool(rng.random() < propensity)
            noisy = bool(rng.random() < (0.16 if not is_af else 0.08))
            sig, rr = make_window(rng, is_af, noisy)
            # What a detector would *measure*: artefact bursts add spurious
            # beats, so a noisy sinus window reads as irregular as real AF.
            rr_cv = float(np.std(rr) / np.mean(rr))
            if noisy:
                rr_cv *= rng.uniform(2.5, 5.0)
            release_id = f"{patient}_{w:04d}"
            ids.append(release_id)
            sigs.append(sig)
            rows.append({"release_id": release_id, "patient": patient,
                         "label": int(is_af), "rr_cv_measured": rr_cv})
        signals_by_patient[patient] = (np.array(ids), np.stack(sigs))

    X = given_columns(rng, rows)
    df = pd.DataFrame(X, columns=[f"f{j:03d}" for j in range(N_GIVEN)])
    meta = pd.DataFrame(rows).drop(columns=["rr_cv_measured"])
    df = pd.concat([meta.reset_index(drop=True), df], axis=1)

    patients = np.array(sorted(signals_by_patient))
    rng.shuffle(patients)
    n_tr = int(round(0.60 * len(patients)))
    n_va = int(round(0.20 * len(patients)))
    splits = {"train": patients[:n_tr],
              "val": patients[n_tr:n_tr + n_va],
              "test": patients[n_tr + n_va:]}

    for name, members in splits.items():
        part = df[df["patient"].isin(members)].reset_index(drop=True)
        if name == "test":
            part[["release_id", "label"]].to_csv(
                out_dir / "test_labels.csv", index=False)
            part = part.drop(columns=["label"])
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, {len(members)} patients")

    for patient, (ids, sigs) in signals_by_patient.items():
        np.savez_compressed(out_dir / "raw_signals" / f"{patient}.npz",
                            release_id=ids, signal=sigs)

    prevalence = df["label"].mean()
    print(f"total {len(df)} windows, AF prevalence {prevalence:.3f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
