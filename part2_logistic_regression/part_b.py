"""
COL774 Assignment 1.2, Part (b) - class imbalance.

    python3 part_b.py part_ab_train.csv part_ab_test_public.csv baseline \
        predictions_baseline.txt weights_baseline.txt

methods: baseline | classweight | classweight2 | focal

All four use the mini-batch AdaGrad settings from Part (a), so the only thing
that differs between them is how the loss weights each example.
"""
import sys

import numpy as np
import pandas as pd

K = 3
META_COLS = {"release_id", "label"}

# fixed AdaGrad protocol from part (a)
EPOCHS = 200
LR = 0.3
BATCH = 32
EPS = 1e-8
SEED = 774

POWER2 = 0.3            # classweight2 uses alpha ** 0.3
GAMMA = 2.0             # focal
ALPHA_POW = 0.5         # focal uses alpha ** 0.5
PCLIP = (1e-12, 1 - 1e-12)

METHODS = ("baseline", "classweight", "classweight2", "focal")


def load_dataset(path, feature_names=None):
    df = pd.read_csv(path)
    if feature_names is None:
        feature_names = [c for c in df.columns if c not in META_COLS]
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64) if "label" in df.columns else None
    return feature_names, X, y


def fit_standardizer(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    return mu, np.where(sd == 0, 1.0, sd)


def apply_standardizer(X, mu, sd):
    return (X - mu) / sd


def one_hot(y, k=K):
    Y = np.zeros((y.shape[0], k), dtype=np.float64)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    Z = np.clip(Z, -60.0, 0.0)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


stable_softmax = softmax        # the report scripts import this name


def class_alpha(y, k=K):
    """alpha_c = n / (3 * n_c)."""
    counts = np.bincount(y, minlength=k).astype(np.float64)
    return y.shape[0] / (k * counts)


def train(X, y, Y, method, snapshot_epochs=()):
    n, m = X.shape
    W = np.zeros((m, Y.shape[1]))
    b = np.zeros(Y.shape[1])
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    alpha = class_alpha(y)
    weight = None
    focal_alpha = None
    if method == "baseline":
        weight = np.ones(n)
    elif method == "classweight":
        weight = alpha[y]
    elif method == "classweight2":
        weight = alpha[y] ** POWER2
    elif method == "focal":
        focal_alpha = (alpha ** ALPHA_POW)[y]
    else:
        raise ValueError(f"unknown method {method!r}")

    rng = np.random.default_rng(SEED)
    losses = []
    snapshot_epochs = set(snapshot_epochs)
    snapshots = {}

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)

        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            Xb, Yb = X[idx], Y[idx]

            P = softmax(Xb @ W + b)
            R = P - Yb

            if method == "focal":
                pt = np.clip(P[np.arange(idx.shape[0]), y[idx]], *PCLIP)
                at = focal_alpha[idx]
                scale = at * (1 - pt) ** (GAMMA - 1) * (
                    (1 - pt) - GAMMA * pt * np.log(pt))
                Rw = scale[:, None] * R
                gW = Xb.T @ Rw / idx.shape[0]
                gb = Rw.sum(axis=0) / idx.shape[0]
            else:
                # weights inside the batch sum to 1
                wb = weight[idx]
                Rw = R * (wb / wb.sum())[:, None]
                gW = Xb.T @ Rw
                gb = Rw.sum(axis=0)

            GW += gW * gW
            Gb += gb * gb
            W -= LR * gW / (np.sqrt(GW) + EPS)
            b -= LR * gb / (np.sqrt(Gb) + EPS)

        losses.append(reported_loss(X, y, W, b, method, alpha))
        if epoch in snapshot_epochs:
            snapshots[epoch] = (W.copy(), b.copy())

    return W, b, losses, snapshots


def reported_loss(X, y, W, b, method, alpha):
    """Whole-training-set loss under the method's own objective, for plotting."""
    P = softmax(X @ W + b)
    pt = np.clip(P[np.arange(X.shape[0]), y], *PCLIP)

    if method == "baseline":
        return -np.mean(np.log(pt))
    if method == "focal":
        at = (alpha ** ALPHA_POW)[y]
        return -np.mean(at * (1 - pt) ** GAMMA * np.log(pt))
    w = alpha[y] if method == "classweight" else alpha[y] ** POWER2
    return np.sum(w * -np.log(pt)) / np.sum(w)


# kept for anything importing the old name
compute_reported_loss = reported_loss


def write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


def main():
    if len(sys.argv) != 6:
        sys.exit("usage: part_b.py train.csv test.csv "
                 "{baseline,classweight,classweight2,focal} predictions.txt "
                 "weights.txt")

    train_csv, test_csv, method, pred_path, weight_path = sys.argv[1:6]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    feats, X_train, y_train = load_dataset(train_csv)
    _, X_test, _ = load_dataset(test_csv, feature_names=feats)

    mu, sd = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mu, sd)
    Xte = apply_standardizer(X_test, mu, sd)

    W, b, _, _ = train(Xtr, y_train, one_hot(y_train), method)

    write_rows(pred_path, softmax(Xte @ W + b))
    write_rows(weight_path, np.vstack([b, W]))


if __name__ == "__main__":
    main()
