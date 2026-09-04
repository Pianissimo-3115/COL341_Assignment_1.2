"""
COL774 Assignment 1 - Part 2(b): Class imbalance handling for multinomial
logistic regression. All four methods use the mini-batch AdaGrad protocol
from Part (a) (T=200, lr=0.3, batch=32, eps=1e-8, seed=774).

Usage (exactly as specified by the assignment):

    python3 part_b.py part_ab_train.csv part_ab_test_public.csv baseline \
        predictions_baseline.txt weights_baseline.txt

    python3 part_b.py part_ab_train.csv part_ab_test_public.csv classweight \
        predictions_classweight.txt weights_classweight.txt

    python3 part_b.py part_ab_train.csv part_ab_test_public.csv classweight2 \
        predictions_classweight2.txt weights_classweight2.txt

    python3 part_b.py part_ab_train.csv part_ab_test_public.csv focal \
        predictions_focal.txt weights_focal.txt
"""
import sys

import numpy as np
import pandas as pd

NUM_CLASSES = 3
NON_FEATURE_COLUMNS = {"release_id", "label"}

# Fixed mini-batch AdaGrad protocol shared by every method in this part.
EPOCHS = 200
LR = 0.3
BATCH_SIZE = 32
ADAGRAD_EPS = 1e-8
SHUFFLE_SEED = 774

CLASSWEIGHT2_POWER = 0.3
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_POWER = 0.5
PROB_CLIP = (1e-12, 1 - 1e-12)

METHODS = {"baseline", "classweight", "classweight2", "focal"}


def load_dataset(path, feature_names=None):
    df = pd.read_csv(path)
    if feature_names is None:
        feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None
    return feature_names, X, y


def fit_standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std == 0, 1.0, std)
    return mean, std_safe


def apply_standardizer(X, mean, std):
    return (X - mean) / std


def one_hot(y, num_classes=NUM_CLASSES):
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def stable_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    shifted = np.clip(shifted, -60.0, 0.0)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def class_alpha(y, num_classes=NUM_CLASSES):
    """alpha_k = n / (3 * n_k) for each class k, indexed 0..num_classes-1."""
    n = y.shape[0]
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    return n / (num_classes * counts)


def train(X, y, Y_onehot, method, snapshot_epochs=()):
    n, m = X.shape
    k = Y_onehot.shape[1]

    W = np.zeros((m, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    alpha_c = class_alpha(y)  # per-class alpha, shape (k,)

    if method == "baseline":
        example_weight = np.ones(n, dtype=np.float64)
    elif method == "classweight":
        example_weight = alpha_c[y]
    elif method == "classweight2":
        example_weight = alpha_c[y] ** CLASSWEIGHT2_POWER
    elif method == "focal":
        alpha_focal_c = alpha_c ** FOCAL_ALPHA_POWER
        example_alpha_focal = alpha_focal_c[y]
    else:
        raise ValueError(f"Unknown method '{method}'")

    rng = np.random.default_rng(SHUFFLE_SEED)
    train_losses = []
    snapshot_epochs = set(snapshot_epochs)
    snapshots = {}

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)

        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            Xb = X[idx]
            Yb = Y_onehot[idx]
            yb = y[idx]

            logits = Xb @ W + b
            P = stable_softmax(logits)
            diff = P - Yb  # (bs, k)

            if method == "focal":
                p_t = np.clip(P[np.arange(idx.shape[0]), yb], *PROB_CLIP)
                a_t = example_alpha_focal[idx]
                factor = a_t * (1 - p_t) ** (FOCAL_GAMMA - 1) * (
                    (1 - p_t) - FOCAL_GAMMA * p_t * np.log(p_t)
                )
                weighted_diff = factor[:, None] * diff
                gW = Xb.T @ weighted_diff / idx.shape[0]
                gb = weighted_diff.sum(axis=0) / idx.shape[0]
            else:
                wb = example_weight[idx]
                norm = wb.sum()
                weighted_diff = diff * (wb / norm)[:, None]
                gW = Xb.T @ weighted_diff
                gb = weighted_diff.sum(axis=0)

            GW += gW * gW
            Gb += gb * gb
            W -= LR * gW / (np.sqrt(GW) + ADAGRAD_EPS)
            b -= LR * gb / (np.sqrt(Gb) + ADAGRAD_EPS)

        train_losses.append(compute_reported_loss(X, y, Y_onehot, W, b, method, alpha_c))
        if epoch in snapshot_epochs:
            snapshots[epoch] = (W.copy(), b.copy())

    return W, b, train_losses, snapshots


def compute_reported_loss(X, y, Y_onehot, W, b, method, alpha_c):
    """Full-training-set loss under the method's own objective (for plotting)."""
    logits = X @ W + b
    P = stable_softmax(logits)
    p_true = np.clip(P[np.arange(X.shape[0]), y], *PROB_CLIP)

    if method == "baseline":
        return -np.mean(np.log(p_true))
    if method == "classweight":
        w = alpha_c[y]
        return np.sum(w * -np.log(p_true)) / np.sum(w)
    if method == "classweight2":
        w = alpha_c[y] ** CLASSWEIGHT2_POWER
        return np.sum(w * -np.log(p_true)) / np.sum(w)
    if method == "focal":
        a_t = (alpha_c ** FOCAL_ALPHA_POWER)[y]
        return -np.mean(a_t * (1 - p_true) ** FOCAL_GAMMA * np.log(p_true))
    raise ValueError(method)


def write_weights(path, W, b):
    with open(path, "w") as f:
        f.write(",".join(f"{v:.17g}" for v in b) + "\n")
        for row in W:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


def write_predictions(path, P):
    with open(path, "w") as f:
        for row in P:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python3 part_b.py train.csv test.csv "
            "{baseline,classweight,classweight2,focal} predictions.txt weights.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path, test_path, method, pred_path, weights_path = sys.argv[1:6]
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")

    feature_names, X_train, y_train = load_dataset(train_path)
    _, X_test, _ = load_dataset(test_path, feature_names=feature_names)

    mean, std = fit_standardizer(X_train)
    X_train_std = apply_standardizer(X_train, mean, std)
    X_test_std = apply_standardizer(X_test, mean, std)

    Y_train = one_hot(y_train)

    W, b, _train_losses, _snapshots = train(X_train_std, y_train, Y_train, method)

    logits_test = X_test_std @ W + b
    P_test = stable_softmax(logits_test)

    write_predictions(pred_path, P_test)
    write_weights(weights_path, W, b)


if __name__ == "__main__":
    main()
