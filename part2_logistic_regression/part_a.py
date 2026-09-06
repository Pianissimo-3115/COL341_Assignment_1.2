"""
COL774 Assignment 1.2, Part (a) - multinomial logistic regression trained with
four gradient descent variants.

    python3 part_a.py part_ab_train.csv part_ab_test_public.csv full_batch \
        predictions_full_batch.txt weights_full_batch.txt

methods: full_batch | mini_batch | sgd | adagrad

Two extra optional args (val.csv, loss_curve.csv) dump the loss curve used for
the report plots. The grader never passes them.
"""
import sys
import time

import numpy as np
import pandas as pd

K = 3                                   # classes: N, A, O
META_COLS = {"release_id", "label"}
SEED = 774

SETTINGS = {
    "full_batch": dict(epochs=500, lr=0.3, batch=None, shuffle=False, ada=False),
    "mini_batch": dict(epochs=200, lr=0.03, batch=32, shuffle=True, ada=False),
    "sgd":        dict(epochs=30, lr=0.001, batch=1, shuffle=True, ada=False),
    "adagrad":    dict(epochs=200, lr=0.3, batch=32, shuffle=True, ada=True,
                       eps=1e-8),
}


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
    # subtract the row max, then clip, before exponentiating
    Z = Z - Z.max(axis=1, keepdims=True)
    Z = np.clip(Z, -60.0, 0.0)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# kept under the old name too - the report scripts import it
stable_softmax = softmax


def cross_entropy_loss(X, Y, W, b):
    P = np.clip(softmax(X @ W + b), 1e-15, 1.0)
    return -np.mean(np.sum(Y * np.log(P), axis=1))


def train(X, Y, method, X_val=None, Y_val=None, snapshot_epochs=()):
    """Fit W, b with the chosen optimiser.

    Returns W, b, train losses, val losses, weight snapshots and the cumulative
    training seconds per epoch (the report plots loss against time, and one
    epoch is not the same amount of work for every method).
    """
    cfg = SETTINGS[method]
    n, m = X.shape
    W = np.zeros((m, Y.shape[1]))
    b = np.zeros(Y.shape[1])
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    bs = cfg["batch"] or n
    lr = cfg["lr"]
    eps = cfg.get("eps", 1e-8)
    rng = np.random.default_rng(SEED) if cfg["shuffle"] else None

    train_losses, val_losses, epoch_times = [], [], []
    snapshot_epochs = set(snapshot_epochs)
    snapshots = {}
    elapsed = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        order = rng.permutation(n) if cfg["shuffle"] else np.arange(n)

        for start in range(0, n, bs):
            idx = order[start:start + bs]
            Xb, Yb = X[idx], Y[idx]

            P = softmax(Xb @ W + b)
            R = P - Yb                          # residual
            gW = Xb.T @ R / idx.shape[0]        # divide by this batch's size
            gb = R.sum(axis=0) / idx.shape[0]

            if cfg["ada"]:
                GW += gW * gW
                Gb += gb * gb
                W -= lr * gW / (np.sqrt(GW) + eps)
                b -= lr * gb / (np.sqrt(Gb) + eps)
            else:
                W -= lr * gW
                b -= lr * gb

        elapsed += time.perf_counter() - t0

        train_losses.append(cross_entropy_loss(X, Y, W, b))
        if X_val is not None:
            val_losses.append(cross_entropy_loss(X_val, Y_val, W, b))
        epoch_times.append(elapsed)
        if epoch in snapshot_epochs:
            snapshots[epoch] = (W.copy(), b.copy())

    return W, b, train_losses, val_losses, snapshots, epoch_times


def write_rows(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


def main():
    if len(sys.argv) < 6:
        sys.exit("usage: part_a.py train.csv test.csv "
                 "{full_batch,mini_batch,sgd,adagrad} predictions.txt "
                 "weights.txt [val.csv] [loss_curve.csv]")

    train_csv, test_csv, method, pred_path, weight_path = sys.argv[1:6]
    val_csv = sys.argv[6] if len(sys.argv) > 6 else None
    curve_path = sys.argv[7] if len(sys.argv) > 7 else None

    if method not in SETTINGS:
        raise ValueError(f"unknown method {method!r}, expected one of "
                         f"{list(SETTINGS)}")

    feats, X_train, y_train = load_dataset(train_csv)
    _, X_test, _ = load_dataset(test_csv, feature_names=feats)

    mu, sd = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mu, sd)
    Xte = apply_standardizer(X_test, mu, sd)
    Ytr = one_hot(y_train)

    Xva = Yva = None
    if val_csv:
        _, X_val, y_val = load_dataset(val_csv, feature_names=feats)
        Xva = apply_standardizer(X_val, mu, sd)
        Yva = one_hot(y_val)

    W, b, tr_loss, va_loss, _, secs = train(Xtr, Ytr, method, Xva, Yva)

    write_rows(pred_path, softmax(Xte @ W + b))
    write_rows(weight_path, np.vstack([b, W]))   # bias first, then the 78 rows

    if curve_path:
        with open(curve_path, "w") as f:
            f.write("epoch,seconds,train_loss,val_loss\n")
            for i, loss in enumerate(tr_loss, start=1):
                v = va_loss[i - 1] if va_loss else ""
                f.write(f"{i},{secs[i - 1]},{loss},{v}\n")


if __name__ == "__main__":
    main()
