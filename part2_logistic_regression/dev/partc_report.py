"""
Dev tool (NOT part of the submission).

Produces everything report.pdf needs for Part (c), by re-running part_c.py's
own pipeline so the numbers quoted in the report are the numbers the submitted
script actually produces.

    python partc_report.py dataset_dir/ out_dir/ [variant]

Writes into out_dir/:
    m_vs_threshold.png   M vs decision threshold on val.csv, chosen value marked
                         -> report item (iii), first plot
    epsilon_sweep.png    best achievable M and its threshold vs 1/epsilon
                         swept over [10, 1000]  -> report item (iii), 2nd plot
    report_tables.md     top-10 features and how they were ranked   -> item (i)
                         class-imbalance comparison                 -> item (ii)
                         final metrics at chosen threshold vs 0.5   -> item (iv)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from sklearn.metrics import average_precision_score               # noqa: E402

import part_c                                                      # noqa: E402

EPS_INVERSE_GRID = np.linspace(10.0, 1000.0, 200)


def average_precision(y, prob):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, prob))


def metrics_row(y, prob, threshold, alpha=part_c.ALPHA):
    m, tp, fp, tpr, fpr = part_c.m_metric(y, prob, threshold, alpha)
    return {"threshold": threshold, "M": m, "TP": tp, "FP": fp,
            "recall": tpr, "fpr": fpr, "AP": average_precision(y, prob)}


def fmt(row, name):
    return (f"| {name} | {row['threshold']:.4f} | {row['M']:.4f} | {row['TP']} "
            f"| {row['FP']} | {row['recall']:.4f} | {row['fpr']:.4f} "
            f"| {row['AP']:.4f} |")


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    dataset_dir, out_dir = sys.argv[1], Path(sys.argv[2])
    variant = sys.argv[3] if len(sys.argv) == 4 else part_c.DEFAULT_VARIANT
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = part_c.load_split(dataset_dir, "train")
    val_df = part_c.load_split(dataset_dir, "val")
    if train_df is None or val_df is None or "label" not in val_df.columns:
        raise SystemExit("need train.csv and a labelled val.csv")

    feature_cols = part_c.get_feature_columns(train_df)
    n_given = len(feature_cols)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    groups = train_df["patient"].to_numpy()

    use_raw = variant in ("raw", "gam")
    raw_names = part_c._raw_feature_names() if use_raw else []
    all_names = feature_cols + raw_names

    X_train = part_c.build_matrix(train_df, feature_cols, dataset_dir,
                                  raw_names, use_raw)
    X_val = part_c.build_matrix(val_df, feature_cols, dataset_dir,
                                raw_names, use_raw)

    state = part_c.fit_feature_map(X_train, y_train, groups, variant, n_given)
    phi_train = part_c.apply_feature_map(X_train, state)
    phi_val = part_c.apply_feature_map(X_val, state)

    c, class_weight, cv_m = part_c.tune_hyperparameters(phi_train, y_train, groups)
    model = part_c.make_model(c, class_weight)
    model.fit(phi_train, y_train)
    prob_val = model.predict_proba(phi_val)[:, 1]

    # Threshold exactly as part_c.py picks it: pooled grouped-OOF on train plus
    # the final model's predictions on val.
    oof = part_c.grouped_oof_probs(phi_train, y_train, groups, c, class_weight)
    pooled_prob = np.concatenate([oof, prob_val])
    pooled_y = np.concatenate([y_train, y_val])
    threshold, pooled_m = part_c.select_threshold(pooled_y, pooled_prob)

    # ---- plot 1: M vs threshold on val ----
    thr, m_curve, tpr_curve, _ = part_c.sweep_thresholds(y_val, prob_val)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thr, m_curve, lw=1.6, color="#1f4e79", label="M on val.csv")
    ax.axvline(threshold, color="#c0392b", ls="--", lw=1.4,
               label=f"chosen threshold = {threshold:.3f}")
    ax.axvline(0.5, color="#7f8c8d", ls=":", lw=1.2, label="default 0.5")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("decision threshold on P(AF)")
    ax.set_ylabel(r"$M = TPR - FPR/\epsilon$   ($\epsilon = 1/100$)")
    ax.set_title("Part (c): M vs decision threshold on val.csv")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "m_vs_threshold.png", dpi=160)
    plt.close(fig)

    # ---- plot 2: sweep 1/epsilon over [10, 1000] ----
    best_m, best_t = [], []
    for alpha in EPS_INVERSE_GRID:
        t_a, m_a, tpr_a, _ = part_c.sweep_thresholds(y_val, prob_val, alpha)
        ok = tpr_a >= part_c.MIN_TPR_ELIGIBLE
        if not np.any(ok):
            best_m.append(np.nan)
            best_t.append(np.nan)
            continue
        i = np.argmax(np.where(ok, m_a, -np.inf))
        best_m.append(m_a[i])
        best_t.append(t_a[i])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(EPS_INVERSE_GRID, best_m, lw=1.6, color="#1f4e79")
    axes[0].set_xlabel(r"$1/\epsilon$ (false-positive penalty)")
    axes[0].set_ylabel("best achievable M on val.csv")
    axes[0].set_title(r"(a) best M vs $1/\epsilon$")
    axes[0].grid(alpha=0.3)

    axes[1].plot(EPS_INVERSE_GRID, best_t, lw=1.6, color="#c0392b")
    axes[1].set_xlabel(r"$1/\epsilon$ (false-positive penalty)")
    axes[1].set_ylabel("threshold achieving the best M")
    axes[1].set_title(r"(b) arg-max threshold vs $1/\epsilon$")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "epsilon_sweep.png", dpi=160)
    plt.close(fig)

    # ---- tables ----
    lines = ["# Part (c) report tables", "",
             f"variant = `{variant}`, d = {phi_train.shape[1]}, "
             f"C = {c}, class_weight = {class_weight}", "",
             f"patient-grouped CV M = {cv_m:.4f}, "
             f"pooled threshold-selection M = {pooled_m:.4f}", ""]

    lines += ["## (i) Top 10 features", "",
              "Ranked by mean out-of-fold |AUC - 0.5| across patient-grouped "
              "folds, so a column that only separates a few patients cannot "
              "outrank one that generalises.", "",
              "| rank | feature | mean OOF \\|AUC-0.5\\| |", "|---|---|---|"]
    ranking = state.get("ranking")
    if ranking is not None:
        names = [all_names[i] for i in state["base_cols"]]
        for rank, j in enumerate(np.argsort(-ranking)[:10], start=1):
            lines.append(f"| {rank} | `{names[j]}` | {ranking[j]:.4f} |")
    else:
        lines.append("| - | (variant does not rank features) | - |")

    lines += ["", "## (ii) Class imbalance", "",
              "Same features and threshold rule, changing only the training "
              "class weights.", "",
              "| setting | threshold | M | TP | FP | recall | FPR | AP |",
              "|---|---|---|---|---|---|---|---|"]
    for setting in (None, "balanced"):
        alt = part_c.make_model(c, setting)
        alt.fit(phi_train, y_train)
        alt_prob = alt.predict_proba(phi_val)[:, 1]
        alt_oof = part_c.grouped_oof_probs(phi_train, y_train, groups, c, setting)
        alt_t, _ = part_c.select_threshold(
            np.concatenate([y_train, y_val]),
            np.concatenate([alt_oof, alt_prob]))
        lines.append(fmt(metrics_row(y_val, alt_prob, alt_t),
                         f"class_weight={setting}"))

    lines += ["", "## (iv) Final metrics on val.csv", "",
              "| setting | threshold | M | TP | FP | recall | FPR | AP |",
              "|---|---|---|---|---|---|---|---|",
              fmt(metrics_row(y_val, prob_val, threshold), "chosen threshold"),
              fmt(metrics_row(y_val, prob_val, 0.5), "default 0.5")]

    # Baseline: the given 392 columns only, i.e. no raw-signal engineering.
    base_state = part_c.fit_feature_map(X_train, y_train, groups, "given", n_given)
    base_phi = part_c.apply_feature_map(X_train, base_state)
    base_c, base_cw, _ = part_c.tune_hyperparameters(base_phi, y_train, groups)
    base_model = part_c.make_model(base_c, base_cw)
    base_model.fit(base_phi, y_train)
    base_prob = base_model.predict_proba(
        part_c.apply_feature_map(X_val, base_state))[:, 1]
    base_oof = part_c.grouped_oof_probs(base_phi, y_train, groups, base_c, base_cw)
    base_t, _ = part_c.select_threshold(
        np.concatenate([y_train, y_val]), np.concatenate([base_oof, base_prob]))
    lines.append(fmt(metrics_row(y_val, base_prob, base_t),
                     "baseline (392 given cols)"))

    anchor = part_c.theoretical_threshold(pooled_y)
    lines += ["", "## (iii) Threshold selection", "",
              "The threshold is chosen on pooled patient-grouped out-of-fold "
              "predictions on train.csv plus the final model's predictions on "
              "val.csv, taking the *largest* threshold whose M is within "
              f"{part_c.THRESHOLD_TOLERANCE} of the best rather than the raw "
              "arg-max, because the arg-max sits exactly where a couple of "
              "negatives happen to fall below the cut.", "",
              f"- chosen threshold: **{threshold:.4f}**",
              f"- calibrated-optimal anchor "
              f"alpha*P/(N + alpha*P) = **{anchor:.4f}**",
              "", "Trend to explain in the report: as 1/epsilon grows the "
              "false-positive penalty dominates, so the optimal threshold "
              "rises towards 1 and the best achievable M falls, since buying "
              "recall stops being worth even one extra false positive.", ""]

    (out_dir / "report_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir/'m_vs_threshold.png'}")
    print(f"wrote {out_dir/'epsilon_sweep.png'}")
    print(f"wrote {out_dir/'report_tables.md'}")


if __name__ == "__main__":
    main()
