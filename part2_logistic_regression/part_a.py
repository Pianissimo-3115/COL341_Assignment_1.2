"""
COL774 Assignment 1 - Part 2(a): Multinomial (softmax) logistic regression
trained with four from-scratch gradient-descent variants.

Usage (exactly as specified by the assignment):

    python3 part_a.py part_ab_train.csv part_ab_test_public.csv full_batch \
        predictions_full_batch.txt weights_full_batch.txt

    python3 part_a.py part_ab_train.csv part_ab_test_public.csv mini_batch \
        predictions_mini_batch.txt weights_mini_batch.txt

    python3 part_a.py part_ab_train.csv part_ab_test_public.csv sgd \
        predictions_sgd.txt weights_sgd.txt

    python3 part_a.py part_ab_train.csv part_ab_test_public.csv adagrad \
        predictions_adagrad.txt weights_adagrad.txt

Optional extra arguments (NOT used by the autograder, only for producing the
training/validation loss-curve plots required in the report):

    python3 part_a.py train.csv test.csv <method> pred.txt weights.txt \
        part_ab_val.csv loss_curve.csv
"""
import sys

import numpy as np
import pandas as pd

NUM_CLASSES = 3
NON_FEATURE_COLUMNS = {"release_id", "label"}

HYPERPARAMS = {
    "full_batch": dict(epochs=500, lr=0.3, batch_size=None, shuffle=False, adagrad=False),
    "mini_batch": dict(epochs=200, lr=0.03, batch_size=32, shuffle=True, adagrad=False),
    "sgd": dict(epochs=30, lr=0.001, batch_size=1, shuffle=True, adagrad=False),
    "adagrad": dict(epochs=200, lr=0.3, batch_size=32, shuffle=True, adagrad=True, eps=1e-8),
}
SHUFFLE_SEED = 774


def load_dataset(path, feature_names=None):
    """Load a part_ab csv. Returns (feature_names, X, y_or_None)."""
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


def cross_entropy_loss(X, Y_onehot, W, b):
    logits = X @ W + b
    P = stable_softmax(logits)
    P_safe = np.clip(P, 1e-15, 1.0)
    return -np.mean(np.sum(Y_onehot * np.log(P_safe), axis=1))


def train(X, Y_onehot, method, X_val=None, Y_onehot_val=None):
    """Generic trainer covering all four required optimisers."""
    hp = HYPERPARAMS[method]
    n, m = X.shape
    k = Y_onehot.shape[1]

    W = np.zeros((m, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    batch_size = hp["batch_size"] or n
    rng = np.random.default_rng(SHUFFLE_SEED) if hp["shuffle"] else None
    eps = hp.get("eps", 1e-8)

    train_losses = []
    val_losses = []

    for _ in range(hp["epochs"]):
        order = rng.permutation(n) if hp["shuffle"] else np.arange(n)

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            Xb = X[idx]
            Yb = Y_onehot[idx]
            bs = idx.shape[0]

            logits = Xb @ W + b
            P = stable_softmax(logits)
            diff = P - Yb  # (bs, k)

            gW = Xb.T @ diff / bs
            gb = diff.sum(axis=0) / bs

            if hp["adagrad"]:
                GW += gW * gW
                Gb += gb * gb
                W -= hp["lr"] * gW / (np.sqrt(GW) + eps)
                b -= hp["lr"] * gb / (np.sqrt(Gb) + eps)
            else:
                W -= hp["lr"] * gW
                b -= hp["lr"] * gb

        # Record loss once per full epoch, over the FULL training set.
        train_losses.append(cross_entropy_loss(X, Y_onehot, W, b))
        if X_val is not None:
            val_losses.append(cross_entropy_loss(X_val, Y_onehot_val, W, b))

    return W, b, train_losses, val_losses


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
    if len(sys.argv) < 6:
        print(
            "Usage: python3 part_a.py train.csv test.csv "
            "{full_batch,mini_batch,sgd,adagrad} predictions.txt weights.txt "
            "[val.csv] [loss_curve.csv]",
            file=sys.stderr,
        )
        sys.exit(1)

    train_path, test_path, method, pred_path, weights_path = sys.argv[1:6]
    val_path = sys.argv[6] if len(sys.argv) > 6 else None
    loss_curve_path = sys.argv[7] if len(sys.argv) > 7 else None

    if method not in HYPERPARAMS:
        raise ValueError(f"Unknown method '{method}', expected one of {list(HYPERPARAMS)}")

    feature_names, X_train, y_train = load_dataset(train_path)
    _, X_test, _ = load_dataset(test_path, feature_names=feature_names)

    mean, std = fit_standardizer(X_train)
    X_train_std = apply_standardizer(X_train, mean, std)
    X_test_std = apply_standardizer(X_test, mean, std)

    Y_train = one_hot(y_train)

    X_val_std, Y_val = None, None
    if val_path is not None:
        _, X_val, y_val = load_dataset(val_path, feature_names=feature_names)
        X_val_std = apply_standardizer(X_val, mean, std)
        Y_val = one_hot(y_val)

    W, b, train_losses, val_losses = train(X_train_std, Y_train, method, X_val_std, Y_val)

    logits_test = X_test_std @ W + b
    P_test = stable_softmax(logits_test)

    write_predictions(pred_path, P_test)
    write_weights(weights_path, W, b)

    if loss_curve_path is not None:
        with open(loss_curve_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, tl in enumerate(train_losses, start=1):
                vl = val_losses[i - 1] if val_losses else ""
                f.write(f"{i},{tl},{vl}\n")


if __name__ == "__main__":
    main()
