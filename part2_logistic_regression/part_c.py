"""
COL774 Assignment 1.2, Part (c) - AF detection.

    python3 part_c.py dataset_dir/ model.pkl final_features.csv

dataset_dir holds train.csv, val.csv and test.csv. We fit a feature map on
train only, train a binary logistic regression on it, pick a decision
threshold, and write

    model.pkl           {'weights', 'bias', 'threshold'}
    final_features.csv  feat_0..feat_{d-1} + release_id, one row per test row

which is scored as sigmoid(phi(x) . weights + bias) >= threshold.

phi(x) is the 392 supplied feature columns, cleaned. The metric is
M = TPR - 100 * FPR, so one false positive costs about as much as 100/N of
recall, and that drives most of what follows: clipping features to training
quantiles so an outlier cannot cause a confident false positive, grouping every
fold by patient, and choosing a threshold that leans strict rather than sitting
on the arg-max.
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 774
ALPHA = 100.0                   # M = TPR - ALPHA * FPR, i.e. epsilon = 1/100
MIN_TPR_ELIGIBLE = 0.10         # below this the submission is not graded
MIN_TPR = 0.15                  # our own margin above that
FOLDS = 5
CLIP_Q = 0.001                  # winsorise to train [0.1%, 99.9%]
SLACK = 0.01                    # M we will give up for a stricter threshold
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
WEIGHT_GRID = (None, "balanced")

META_COLS = {"release_id", "patient", "label"}

# The course flagged these as unusually hard patients. Dropped from training
# and from threshold selection, never from the scored set.
SKIP_PATIENTS = ("045", "051")


def load_split(dataset_dir, name):
    path = Path(dataset_dir) / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def get_feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def _patient_forms(value):
    """'045', 45 and 'P045' should all match the same patient."""
    s = str(value).strip()
    forms = {s, s.lower()}
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        forms |= {digits, str(int(digits))}
    return forms


def drop_excluded_patients(df, name):
    if df is None or "patient" not in df.columns or not SKIP_PATIENTS:
        return df
    wanted = set()
    for pid in SKIP_PATIENTS:
        wanted |= _patient_forms(pid)
    hit = np.array([bool(_patient_forms(p) & wanted)
                    for p in df["patient"].to_numpy()], dtype=bool)
    if hit.any():
        who = sorted({str(p) for p in df["patient"].to_numpy()[hit]})
        print(f"{name}: dropped {int(hit.sum())} rows from patient(s) {who}",
              file=sys.stderr)
    return df.loc[~hit].reset_index(drop=True)


def fit_preprocessor(X):
    """Median, clip limits, mean and sd - all from the training split only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(X, axis=0)
        lo = np.nanquantile(X, CLIP_Q, axis=0)
        hi = np.nanquantile(X, 1.0 - CLIP_Q, axis=0)

    med = np.nan_to_num(med, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.where(np.isfinite(lo), lo, med)
    hi = np.where(np.isfinite(hi), hi, med)
    flip = lo > hi
    lo[flip], hi[flip] = hi[flip], lo[flip]

    Z = np.clip(np.where(np.isfinite(X), X, med), lo, hi)
    sd = Z.std(axis=0, ddof=0)
    return {"median": med, "lo": lo, "hi": hi, "mean": Z.mean(axis=0),
            "std": np.where(sd > 0, sd, 1.0)}


def apply_preprocessor(X, st):
    Z = np.clip(np.where(np.isfinite(X), X, st["median"]), st["lo"], st["hi"])
    Z = (Z - st["mean"]) / st["std"]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def m_metric(y, prob, thr, alpha=ALPHA):
    pred = prob >= thr
    P = int(np.count_nonzero(y == 1))
    N = int(np.count_nonzero(y == 0))
    if P == 0 or N == 0:
        return -np.inf, 0, 0, 0.0, 0.0
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    return tp / P - alpha * fp / N, tp, fp, tp / P, fp / N


def sweep_thresholds(y, prob, alpha=ALPHA):
    """M, TPR, FPR at every distinct cut, ordered from strictest downwards.

    Sorting once and taking cumulative counts is exact, and unlike a fixed grid
    it cannot step over the optimum.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    P = int(np.count_nonzero(y == 1))
    N = int(np.count_nonzero(y == 0))
    if P == 0 or N == 0 or p.size == 0:
        empty = np.empty(0)
        return empty, empty, empty, empty

    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    edge = np.r_[np.nonzero(np.diff(ps))[0], ps.size - 1]
    tpr = tp[edge] / P
    fpr = fp[edge] / N
    return ps[edge], tpr - alpha * fpr, tpr, fpr


def select_threshold(y, prob, alpha=ALPHA, min_tpr=MIN_TPR, slack=SLACK):
    """Strictest threshold whose M is still within `slack` of the best.

    The arg-max sits exactly where a couple of negatives happen to land below
    the cut. Nudging the threshold up costs very little recall and protects
    against the test set putting one more negative above it.
    """
    thr, m, tpr, _ = sweep_thresholds(y, prob, alpha)
    ok = tpr >= min_tpr
    if not np.any(ok):
        return 0.5, -np.inf
    best = float(m[ok].max())
    good = ok & (m >= best - slack)
    return float(thr[np.argmax(good)]), best     # thr descends, so take first


def theoretical_threshold(y):
    """For calibrated probabilities, admitting a window with P(AF)=p changes M
    by p/P - alpha(1-p)/N, so the break-even point is alpha*P/(N + alpha*P)."""
    P = int(np.count_nonzero(y == 1))
    N = int(np.count_nonzero(y == 0))
    return np.nan if P == 0 or N == 0 else ALPHA * P / (N + ALPHA * P)


def make_model(c, class_weight):
    return LogisticRegression(C=c, class_weight=class_weight, max_iter=5000,
                              solver="lbfgs", random_state=SEED)


def grouped_oof_probs(Z, y, groups, c, class_weight, folds=FOLDS):
    folds = max(2, min(folds, len(np.unique(groups))))
    oof = np.full(y.shape, np.nan)
    for tr, te in GroupKFold(n_splits=folds).split(Z, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model(c, class_weight)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z[tr], y[tr])
        oof[te] = model.predict_proba(Z[te])[:, 1]
    return np.nan_to_num(oof, nan=0.0)


def tune_hyperparameters(Z, y, groups):
    """Pick C and the class weighting by patient-grouped out-of-fold M."""
    best = (C_GRID[len(C_GRID) // 2], "balanced", -np.inf)
    for class_weight in WEIGHT_GRID:
        for c in C_GRID:
            oof = grouped_oof_probs(Z, y, groups, c, class_weight)
            _, m = select_threshold(y, oof)
            if m > best[2]:
                best = (c, class_weight, m)
    return best


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: part_c.py dataset_dir/ model.pkl final_features.csv")

    dataset_dir, model_path, features_path = sys.argv[1:4]
    np.random.seed(SEED)

    train_df = load_split(dataset_dir, "train")
    val_df = load_split(dataset_dir, "val")
    test_df = load_split(dataset_dir, "test")
    if train_df is None or test_df is None:
        raise FileNotFoundError("need train.csv and test.csv in dataset_dir")

    # test_df is left alone - every test row still has to be scored
    train_df = drop_excluded_patients(train_df, "train.csv")
    val_df = drop_excluded_patients(val_df, "val.csv")

    feats = get_feature_columns(train_df)
    y = train_df["label"].to_numpy(dtype=np.int64)
    groups = (train_df["patient"].to_numpy() if "patient" in train_df.columns
              else np.arange(len(train_df)))

    prep = fit_preprocessor(train_df[feats].to_numpy(dtype=np.float64))
    Ztr = apply_preprocessor(train_df[feats].to_numpy(dtype=np.float64), prep)
    Zte = apply_preprocessor(test_df[feats].to_numpy(dtype=np.float64), prep)
    if Ztr.shape[1] >= 500:
        raise RuntimeError(f"phi has {Ztr.shape[1]} dims, the spec allows < 500")

    c, class_weight, cv_m = tune_hyperparameters(Ztr, y, groups)
    print(f"d={Ztr.shape[1]} C={c} class_weight={class_weight} "
          f"grouped-CV M={cv_m:.4f}", file=sys.stderr)

    model = make_model(c, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Ztr, y)

    # threshold from out-of-fold predictions on train plus the untouched val
    pooled_p = grouped_oof_probs(Ztr, y, groups, c, class_weight)
    pooled_y = y
    if val_df is not None and "label" in val_df.columns:
        Zva = apply_preprocessor(val_df[feats].to_numpy(dtype=np.float64), prep)
        pooled_p = np.concatenate([pooled_p, model.predict_proba(Zva)[:, 1]])
        pooled_y = np.concatenate(
            [pooled_y, val_df["label"].to_numpy(dtype=np.int64)])

    thr, pooled_m = select_threshold(pooled_y, pooled_p)
    _, tp, fp, tpr, fpr = m_metric(pooled_y, pooled_p, thr)
    print(f"threshold={thr:.4f} (calibrated anchor "
          f"{theoretical_threshold(pooled_y):.4f}) pooled M={pooled_m:.4f} "
          f"TP={tp} FP={fp} TPR={tpr:.4f} FPR={fpr:.4f}", file=sys.stderr)
    if tpr < MIN_TPR_ELIGIBLE:
        print("warning: pooled TPR is under the 0.1 eligibility floor",
              file=sys.stderr)

    with open(model_path, "wb") as f:
        pickle.dump({"weights": model.coef_[0].astype(np.float64),
                     "bias": float(model.intercept_[0]),
                     "threshold": float(thr),
                     "method": "logreg"}, f)

    out = pd.DataFrame(Zte, columns=[f"feat_{i}" for i in range(Zte.shape[1])])
    out.insert(0, "release_id", test_df["release_id"].to_numpy())
    out.to_csv(features_path, index=False)
    print(f"wrote {model_path} and {features_path} "
          f"({len(out)} rows, {Zte.shape[1]} features)", file=sys.stderr)


if __name__ == "__main__":
    main()
