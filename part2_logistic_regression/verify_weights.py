"""
Self-verification harness (NOT part of the graded submission).

Runs every gradient-descent variant from part_a.py and every class-imbalance
method from part_b.py on the real training data, snapshots the weights after
each of the first 5 epochs, and diffs them against the reference weight
traces released on the course Google Drive
(drive_data/weight_traces/part_a/<method>_epoch<N>.txt and .../part_b/...).

Also cross-checks the recorded training loss against drive_data/weight_traces/
loss_by_epoch.csv.

Usage:
    python verify_weights.py [data_dir] [weight_traces_dir]

Defaults: data_dir=drive_data, weight_traces_dir=drive_data/weight_traces
(both relative to this script's directory).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part_a  # noqa: E402
import part_b  # noqa: E402

SNAPSHOT_EPOCHS = (1, 2, 3, 4, 5)
ATOL = 1e-4
RTOL = 1e-4

PART_A_METHODS = ["full_batch", "mini_batch", "sgd", "adagrad"]
PART_B_METHODS = ["baseline", "classweight", "classweight2", "focal"]


def load_reference_weights(path):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    b_ref = np.array([float(v) for v in lines[0].split(",")])
    W_ref = np.array([[float(v) for v in line.split(",")] for line in lines[1:]])
    return W_ref, b_ref


def compare(W, b, W_ref, b_ref):
    diff_W = np.abs(W - W_ref)
    diff_b = np.abs(b - b_ref)
    max_diff = max(diff_W.max(), diff_b.max())
    ok = np.allclose(W, W_ref, atol=ATOL, rtol=RTOL) and np.allclose(b, b_ref, atol=ATOL, rtol=RTOL)
    return ok, max_diff


def main():
    here = Path(__file__).resolve().parent
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "drive_data"
    trace_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_dir / "weight_traces"

    train_path = data_dir / "part_ab_train.csv"
    if not train_path.exists():
        print(f"Could not find {train_path}", file=sys.stderr)
        sys.exit(1)

    ref_loss = None
    loss_csv = trace_dir / "loss_by_epoch.csv"
    if loss_csv.exists():
        ref_loss = pd.read_csv(loss_csv)

    results = []

    # ---- Part (a) ----
    feature_names, X_train, y_train = part_a.load_dataset(str(train_path))
    mean, std = part_a.fit_standardizer(X_train)
    X_train_std = part_a.apply_standardizer(X_train, mean, std)
    Y_train = part_a.one_hot(y_train)

    loss_rows = []

    for method in PART_A_METHODS:
        _, _, train_losses, _, snapshots, _times = part_a.train(
            X_train_std, Y_train, method, snapshot_epochs=SNAPSHOT_EPOCHS
        )
        for epoch in SNAPSHOT_EPOCHS:
            ref_path = trace_dir / "part_a" / f"{method}_epoch{epoch}.txt"
            if not ref_path.exists() or epoch not in snapshots:
                continue
            W_ref, b_ref = load_reference_weights(ref_path)
            W, b = snapshots[epoch]
            ok, max_diff = compare(W, b, W_ref, b_ref)
            results.append(("part_a", method, epoch, ok, max_diff))
            if epoch <= len(train_losses):
                loss_rows.append(("part_a", method, epoch, train_losses[epoch - 1]))

    # ---- Part (b) ----
    feature_names_b, X_train_b, y_train_b = part_b.load_dataset(str(train_path))
    mean_b, std_b = part_b.fit_standardizer(X_train_b)
    X_train_b_std = part_b.apply_standardizer(X_train_b, mean_b, std_b)
    Y_train_b = part_b.one_hot(y_train_b)

    for method in PART_B_METHODS:
        _, _, train_losses, snapshots = part_b.train(
            X_train_b_std, y_train_b, Y_train_b, method, snapshot_epochs=SNAPSHOT_EPOCHS
        )
        for epoch in SNAPSHOT_EPOCHS:
            ref_path = trace_dir / "part_b" / f"{method}_epoch{epoch}.txt"
            if not ref_path.exists() or epoch not in snapshots:
                continue
            W_ref, b_ref = load_reference_weights(ref_path)
            W, b = snapshots[epoch]
            ok, max_diff = compare(W, b, W_ref, b_ref)
            results.append(("part_b", method, epoch, ok, max_diff))
            if epoch <= len(train_losses):
                loss_rows.append(("part_b", method, epoch, train_losses[epoch - 1]))

    # ---- Report ----
    header = f"{'part':<7} {'method':<14} {'epoch':>5} {'max|diff|':>12} {'result':>8}"
    print(header)
    print("-" * len(header))
    n_fail = 0
    for part, method, epoch, ok, max_diff in results:
        status = "PASS" if ok else "FAIL"
        n_fail += 0 if ok else 1
        print(f"{part:<7} {method:<14} {epoch:>5} {max_diff:>12.6g} {status:>8}")

    print()
    print(f"{len(results) - n_fail}/{len(results)} snapshots matched within "
          f"atol={ATOL}, rtol={RTOL}")

    if ref_loss is not None and loss_rows:
        print()
        header2 = f"{'part':<7} {'method':<14} {'epoch':>5} {'our_loss':>14} {'ref_loss':>14} {'abs_diff':>10}"
        print(header2)
        print("-" * len(header2))
        loss_fail = 0
        for part, method, epoch, our_loss in loss_rows:
            match = ref_loss[(ref_loss["part"] == part) & (ref_loss["method"] == method)
                              & (ref_loss["epoch"] == epoch)]
            if match.empty:
                continue
            ref_val = float(match["train_loss"].iloc[0])
            diff = abs(our_loss - ref_val)
            if diff > 1e-4:
                loss_fail += 1
            print(f"{part:<7} {method:<14} {epoch:>5} {our_loss:>14.6g} {ref_val:>14.6g} {diff:>10.3g}")
        if loss_fail:
            n_fail += loss_fail

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
