"""
COL774 Assignment 1 - Part 2(c), VERSION 1: given-columns baseline.

    python3 part_c.py dataset_dir/ model.pkl final_features.csv

The simplest version that still respects the metric. phi(x) is just the 392
supplied feature columns, cleaned; no raw-signal processing, no feature
selection, no basis expansion. Fast (a few seconds), nothing to go wrong, and
it is the honest reference point the other versions have to beat.

Even here the pipeline is built around M = TP/P - FP/(N*eps), eps = 1/100, i.e.
M = TPR - 100*FPR. One false positive costs as much as 100/N of recall, so:

  * every feature is winsorised to train-set quantiles, because one wild
    outlier at test time is enough to produce a confident false positive;
  * cross-validation is patient-grouped, since rows from one patient are
    highly correlated and ungrouped folds would flatter the model;
  * the threshold is taken from pooled out-of-fold predictions rather than
    left at 0.5, which is nowhere near optimal for this metric.

Version 2 adds feature selection, version 3 adds raw-signal ECG features,
version 4 adds a spline basis, version 5 picks among them automatically.

Disclosure (for report.pdf): written with assistance from Claude (Anthropic).
No third-party source code is copied in; only NumPy, pandas and scikit-learn
are used.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 774
EPSILON = 1.0 / 100.0
ALPHA = 1.0 / EPSILON               # M = TPR - ALPHA * FPR
MIN_TPR_ELIGIBLE = 0.10             # below this the submission is not graded
MIN_TPR_TARGET = 0.15               # our own safety margin above that
N_FOLDS = 5
WINSOR_Q = 0.001                    # clip features to train [0.1%, 99.9%]
THRESHOLD_TOLERANCE = 0.01          # give up this much M to gain FP headroom
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLASS_WEIGHT_GRID = (None, "balanced")

NON_FEATURE_COLUMNS = {"release_id", "patient", "label"}

# Patients the course flagged as genuinely hard to classify (announcement:
# validation performance looks unexpectedly poor, "particularly for patient 051
# in val.csv and 045 in train.csv"). Dropped from training and from threshold
# selection only - never from test.csv, where every row must still be scored.
# Set to () to keep them.
EXCLUDE_PATIENTS = ("045", "051")


def load_split(dataset_dir, name):
    path = Path(dataset_dir) / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def _patient_forms(value):
    """Normalised forms of a patient id, so '045', 45 and 'P045' all match."""
    s = str(value).strip()
    forms = {s, s.lower()}
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        forms.add(digits)
        forms.add(str(int(digits)))
    return forms


def drop_excluded_patients(df, name):
    """Remove flagged patients from a labelled split. Never call this on
    test.csv: every test row has to appear in final_features.csv."""
    if df is None or "patient" not in df.columns or not EXCLUDE_PATIENTS:
        return df
    wanted = set()
    for pid in EXCLUDE_PATIENTS:
        wanted |= _patient_forms(pid)
    mask = np.array([bool(_patient_forms(p) & wanted)
                     for p in df["patient"].to_numpy()], dtype=bool)
    if mask.any():
        names = sorted({str(p) for p in df["patient"].to_numpy()[mask]})
        print(f"{name}: dropped {int(mask.sum())} rows from flagged "
              f"patient(s) {names}", file=sys.stderr)
    return df.loc[~mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Preprocessing - every statistic is fit on train only
# --------------------------------------------------------------------------
def fit_preprocessor(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        median = np.nanmedian(X, axis=0)
        lo = np.nanquantile(X, WINSOR_Q, axis=0)
        hi = np.nanquantile(X, 1.0 - WINSOR_Q, axis=0)
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.where(np.isfinite(lo), lo, median)
    hi = np.where(np.isfinite(hi), hi, median)
    swap = lo > hi
    lo[swap], hi[swap] = hi[swap], lo[swap]

    filled = np.where(np.isfinite(X), X, median)
    clipped = np.clip(filled, lo, hi)
    std = clipped.std(axis=0, ddof=0)
    return {"median": median, "lo": lo, "hi": hi, "mean": clipped.mean(axis=0),
            "std": np.where(std > 0, std, 1.0)}


def apply_preprocessor(X, state):
    filled = np.where(np.isfinite(X), X, state["median"])
    clipped = np.clip(filled, state["lo"], state["hi"])
    out = (clipped - state["mean"]) / state["std"]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Metric, threshold choice
# --------------------------------------------------------------------------
def m_metric(y_true, prob, threshold, alpha=ALPHA):
    pred = prob >= threshold
    p = int(np.count_nonzero(y_true == 1))
    n = int(np.count_nonzero(y_true == 0))
    if p == 0 or n == 0:
        return -np.inf, 0, 0, 0.0, 0.0
    tp = int(np.count_nonzero(pred & (y_true == 1)))
    fp = int(np.count_nonzero(pred & (y_true == 0)))
    tpr, fpr = tp / p, fp / n
    return tpr - alpha * fpr, tp, fp, tpr, fpr


def sweep_thresholds(y_true, prob, alpha=ALPHA):
    """M, TPR and FPR at every distinct cut of `prob`, descending by
    threshold. Exact and O(n log n) - a fixed grid can miss the optimum."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    n_pos = int(np.count_nonzero(y == 1))
    n_neg = int(np.count_nonzero(y == 0))
    if n_pos == 0 or n_neg == 0 or p.size == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    last = np.r_[np.nonzero(np.diff(ps))[0], ps.size - 1]
    tpr = tp[last] / n_pos
    fpr = fp[last] / n_neg
    return ps[last], tpr - alpha * fpr, tpr, fpr


def select_threshold(y_true, prob, alpha=ALPHA, min_tpr=MIN_TPR_TARGET,
                     tolerance=THRESHOLD_TOLERANCE):
    """Highest threshold whose M is within `tolerance` of the best.

    The arg-max of M on a finite sample sits exactly where a couple of
    negatives happen to fall below the cut; a slightly stricter threshold costs
    almost no recall and protects against the test set placing one more
    negative just above it.
    """
    thr, m, tpr, _ = sweep_thresholds(y_true, prob, alpha)
    ok = tpr >= min_tpr
    if not np.any(ok):
        return 0.5, -np.inf
    best_m = float(m[ok].max())
    good = ok & (m >= best_m - tolerance)
    return float(thr[np.argmax(good)]), best_m


def theoretical_threshold(y_true):
    """Threshold maximising expected M for calibrated probabilities: adding a
    window with P(AF)=p changes M by p/P - alpha*(1-p)/N, positive exactly when
    p > alpha*P/(N + alpha*P). Printed as a sanity anchor."""
    p = int(np.count_nonzero(y_true == 1))
    n = int(np.count_nonzero(y_true == 0))
    if p == 0 or n == 0:
        return np.nan
    return ALPHA * p / (n + ALPHA * p)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(c, class_weight):
    return LogisticRegression(C=c, class_weight=class_weight, max_iter=5000,
                              solver="lbfgs", random_state=RANDOM_SEED)


def grouped_oof_probs(phi, y, groups, c, class_weight, n_folds=N_FOLDS):
    n_folds = max(2, min(n_folds, len(np.unique(groups))))
    oof = np.full(y.shape, np.nan, dtype=np.float64)
    for tr, te in GroupKFold(n_splits=n_folds).split(phi, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model(c, class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(phi[tr], y[tr])
        oof[te] = model.predict_proba(phi[te])[:, 1]
    return np.nan_to_num(oof, nan=0.0)


def tune_hyperparameters(phi, y, groups):
    """Pick (C, class_weight) by patient-grouped out-of-fold M."""
    best = (C_GRID[len(C_GRID) // 2], "balanced", -np.inf)
    for class_weight in CLASS_WEIGHT_GRID:
        for c in C_GRID:
            oof = grouped_oof_probs(phi, y, groups, c, class_weight)
            _, m = select_threshold(y, oof)
            if m > best[2]:
                best = (c, class_weight, m)
    return best


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_c.py dataset_dir/ model.pkl "
              "final_features.csv", file=sys.stderr)
        sys.exit(1)

    dataset_dir, model_path, final_features_path = sys.argv[1:4]
    np.random.seed(RANDOM_SEED)

    train_df = load_split(dataset_dir, "train")
    val_df = load_split(dataset_dir, "val")
    test_df = load_split(dataset_dir, "test")
    if train_df is None or test_df is None:
        raise FileNotFoundError("dataset_dir must contain train.csv and test.csv")

    # test_df is deliberately untouched - every test row must be scored.
    train_df = drop_excluded_patients(train_df, "train.csv")
    val_df = drop_excluded_patients(val_df, "val.csv")

    feature_cols = get_feature_columns(train_df)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    groups = (train_df["patient"].to_numpy() if "patient" in train_df.columns
              else np.arange(len(train_df)))

    prep = fit_preprocessor(train_df[feature_cols].to_numpy(dtype=np.float64))
    phi_train = apply_preprocessor(
        train_df[feature_cols].to_numpy(dtype=np.float64), prep)
    phi_test = apply_preprocessor(
        test_df[feature_cols].to_numpy(dtype=np.float64), prep)
    if phi_train.shape[1] >= 500:
        raise RuntimeError(f"phi has {phi_train.shape[1]} dims, spec needs < 500")

    c, class_weight, cv_m = tune_hyperparameters(phi_train, y_train, groups)
    print(f"[v1 baseline] d={phi_train.shape[1]} C={c} "
          f"class_weight={class_weight} grouped-CV M={cv_m:.4f}", file=sys.stderr)

    model = make_model(c, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(phi_train, y_train)

    pooled_prob = grouped_oof_probs(phi_train, y_train, groups, c, class_weight)
    pooled_y = y_train
    if val_df is not None and "label" in val_df.columns:
        phi_val = apply_preprocessor(
            val_df[feature_cols].to_numpy(dtype=np.float64), prep)
        pooled_prob = np.concatenate(
            [pooled_prob, model.predict_proba(phi_val)[:, 1]])
        pooled_y = np.concatenate(
            [pooled_y, val_df["label"].to_numpy(dtype=np.int64)])

    threshold, pooled_m = select_threshold(pooled_y, pooled_prob)
    _, tp, fp, tpr, fpr = m_metric(pooled_y, pooled_prob, threshold)
    print(f"threshold={threshold:.4f} (calibrated-optimal anchor "
          f"{theoretical_threshold(pooled_y):.4f}) pooled M={pooled_m:.4f} "
          f"TP={tp} FP={fp} TPR={tpr:.4f} FPR={fpr:.4f}", file=sys.stderr)
    if tpr < MIN_TPR_ELIGIBLE:
        print("WARNING: pooled TPR below the 0.1 eligibility floor",
              file=sys.stderr)

    with open(model_path, "wb") as f:
        pickle.dump({"weights": model.coef_[0].astype(np.float64),
                     "bias": float(model.intercept_[0]),
                     "threshold": float(threshold),
                     "method": "logreg[v1-baseline]"}, f)

    out = pd.DataFrame(phi_test,
                       columns=[f"feat_{i}" for i in range(phi_test.shape[1])])
    out.insert(0, "release_id", test_df["release_id"].to_numpy())
    out.to_csv(final_features_path, index=False)
    print(f"wrote {model_path} and {final_features_path} "
          f"({len(out)} rows, {phi_test.shape[1]} features)", file=sys.stderr)


if __name__ == "__main__":
    main()
