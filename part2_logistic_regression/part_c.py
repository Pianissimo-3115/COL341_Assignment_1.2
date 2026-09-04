"""
COL774 Assignment 1 - Part 2(c): Feature engineering for AF detection
(competitively graded).

This is a STARTER / BASELINE pipeline, not a finished submission:
  - It uses the 392 given feature columns (78 baseline + 314 Goodfellow-style)
    as-is, with median imputation + standardisation fit on train.csv only.
  - It trains a plain L2-regularised logistic regression (permitted: the
    assignment allows any gradient-based optimiser for training, only the
    final scoring function has to be sigmoid(phi(x) . w + b)).
  - It picks a decision threshold by sweeping a grid and maximising the
    assignment's M metric using patient-grouped out-of-fold predictions on
    train.csv pooled with direct predictions on val.csv.

TODO for you (this is exactly where the competitive marks come from):
  - Engineer features from raw_signals/{patient}.npz (xraw, 7500 samples).
  - Do feature selection over the 392 given columns (many are likely
    redundant/noisy for this task).
  - Investigate noisy patients / label-quality issues via OOF errors.
  - Tune the threshold-selection strategy (see `select_threshold`).

Usage:
    python3 part_c.py dataset_dir/ model.pkl final_features.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

RANDOM_SEED = 774
EPSILON = 1.0 / 100.0  # M = TP/P - FP/(N*epsilon)
MIN_TPR_FOR_GRADING = 0.1
NON_FEATURE_COLUMNS = {"release_id", "patient", "label"}


def load_split(dataset_dir, name):
    path = Path(dataset_dir) / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def fit_preprocessor(X):
    """Median-impute then standardise, fit on training data only."""
    median = np.nanmedian(X, axis=0)
    X_imputed = np.where(np.isnan(X), median, X)
    mean = X_imputed.mean(axis=0)
    std = X_imputed.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    return {"median": median, "mean": mean, "std": std_safe}


def apply_preprocessor(X, state):
    X_imputed = np.where(np.isnan(X), state["median"], X)
    return (X_imputed - state["mean"]) / state["std"]


def m_metric(y_true, prob, threshold):
    pred = prob >= threshold
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        return -np.inf, 0, 0, 0.0
    TP = np.sum((pred == 1) & (y_true == 1))
    FP = np.sum((pred == 1) & (y_true == 0))
    tpr = TP / P
    fpr = FP / N
    M = tpr - fpr / EPSILON
    return M, TP, FP, tpr


def select_threshold(y_true, prob):
    best_t, best_M = 0.5, -np.inf
    for t in np.linspace(0.0, 1.0, 2001):
        M, _, _, tpr = m_metric(y_true, prob, t)
        if tpr < MIN_TPR_FOR_GRADING:
            continue
        if M > best_M:
            best_M, best_t = M, t
    return best_t, best_M


def get_oof_probs(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=np.float64)
    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return oof


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 part_c.py dataset_dir/ model.pkl final_features.csv", file=sys.stderr)
        sys.exit(1)

    dataset_dir, model_path, final_features_path = sys.argv[1:4]

    train_df = load_split(dataset_dir, "train")
    val_df = load_split(dataset_dir, "val")
    test_df = load_split(dataset_dir, "test")
    if train_df is None or test_df is None:
        raise FileNotFoundError("dataset_dir must contain train.csv and test.csv")

    feature_cols = get_feature_columns(train_df)

    X_train_raw = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    patient_train = train_df["patient"].to_numpy()

    prep_state = fit_preprocessor(X_train_raw)
    X_train = apply_preprocessor(X_train_raw, prep_state)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0,
                              random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    # Pool patient-grouped out-of-fold predictions on train with direct
    # predictions on val (if available) to choose a decision threshold.
    oof_train = get_oof_probs(X_train, y_train, patient_train)
    pooled_prob = oof_train
    pooled_y = y_train

    if val_df is not None and "label" in val_df.columns:
        X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float64)
        y_val = val_df["label"].to_numpy(dtype=np.int64)
        X_val = apply_preprocessor(X_val_raw, prep_state)
        prob_val = clf.predict_proba(X_val)[:, 1]
        pooled_prob = np.concatenate([pooled_prob, prob_val])
        pooled_y = np.concatenate([pooled_y, y_val])

    threshold, best_M = select_threshold(pooled_y, pooled_prob)
    print(f"Selected threshold={threshold:.4f} (pooled M={best_M:.4f})", file=sys.stderr)

    weights = clf.coef_[0].astype(np.float64)
    bias = float(clf.intercept_[0])

    with open(model_path, "wb") as f:
        pickle.dump({"weights": weights, "bias": bias, "threshold": threshold}, f)

    X_test_raw = test_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = apply_preprocessor(X_test_raw, prep_state)

    feat_df = pd.DataFrame(X_test, columns=[f"feat_{i}" for i in range(X_test.shape[1])])
    feat_df.insert(0, "release_id", test_df["release_id"].to_numpy())
    feat_df.to_csv(final_features_path, index=False)


if __name__ == "__main__":
    main()
