"""
Dev tool (NOT part of the submission).

One script that produces every number and figure report.pdf needs, so the whole
report can be generated from a single run on Kaggle.

    python run_report_pipeline.py <part_ab_dir> <partc_dataset_dir> <out_dir>

Pass "skip" for either input directory to skip that half.

  <part_ab_dir>          holds part_ab_train.csv and part_ab_val.csv
  <partc_dataset_dir>    holds train.csv, val.csv, test.csv, raw_signals/
  <out_dir>              created if missing

Outputs into <out_dir>:

  REPORT_DATA.md                 <- everything in one file; send this back
  parta_loss_curves.csv          raw per-epoch loss/time for all four methods
  parta_train_loss_vs_time.png   report Part (a) plot 1  (required)
  parta_val_loss_vs_time.png     report Part (a) plot 2  (required)
  partc_m_vs_threshold.png       report Part (c) plot 1  (required)
  partc_epsilon_sweep.png        report Part (c) plot 2  (required)

Part (a) plots put loss against wall-clock training time, as the assignment
asks, not against epochs: one epoch means something different for each method
(full-batch takes a single step, SGD takes n), so epochs are not comparable.

Part (c) evaluates all five versions on val.csv, which is the comparison that
decides which one to submit. Everything is patient-grouped, and no version ever
sees val during training.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from sklearn.metrics import average_precision_score                # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PART_A_METHODS = ["full_batch", "mini_batch", "sgd", "adagrad"]
VERSIONS = [
    ("v1 baseline", "part_c_v1_baseline.py"),
    ("v2 selected", "part_c_v2_selected.py"),
    ("v3 raw", "part_c_v3_raw.py"),
    ("v4 gam", "part_c_v4_gam.py"),
    ("v5 auto", "part_c_v5_auto.py"),
]
COLOURS = {"full_batch": "#1f4e79", "mini_batch": "#c0392b",
           "sgd": "#27ae60", "adagrad": "#8e44ad"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Part (a)
# --------------------------------------------------------------------------
def run_part_a(part_ab_dir, out_dir, lines):
    import part_a

    train_csv = Path(part_ab_dir) / "part_ab_train.csv"
    val_csv = Path(part_ab_dir) / "part_ab_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        lines.append("## Part (a)\n\n*Skipped: part_ab_train.csv / part_ab_val.csv "
                     f"not found under `{part_ab_dir}`.*\n")
        return

    names, X_train, y_train = part_a.load_dataset(str(train_csv))
    _, X_val, y_val = part_a.load_dataset(str(val_csv), feature_names=names)
    mean, std = part_a.fit_standardizer(X_train)
    Xtr = part_a.apply_standardizer(X_train, mean, std)
    Xva = part_a.apply_standardizer(X_val, mean, std)
    Ytr, Yva = part_a.one_hot(y_train), part_a.one_hot(y_val)

    rows, curves = [], {}
    for method in PART_A_METHODS:
        t0 = time.perf_counter()
        _, _, tr, va, _, secs = part_a.train(Xtr, Ytr, method, Xva, Yva)
        wall = time.perf_counter() - t0
        curves[method] = (secs, tr, va)
        best_epoch = int(np.argmin(va)) + 1
        rows.append({
            "method": method,
            "epochs": len(tr),
            "train_seconds": secs[-1],
            "final_train_loss": tr[-1],
            "final_val_loss": va[-1],
            "min_val_loss": min(va),
            "min_val_epoch": best_epoch,
            "min_val_seconds": secs[best_epoch - 1],
        })
        print(f"[part a] {method}: {len(tr)} epochs in {secs[-1]:.2f}s "
              f"(wall {wall:.2f}s), val min {min(va):.4f} @ epoch {best_epoch}",
              file=sys.stderr)

    flat = []
    for method, (secs, tr, va) in curves.items():
        for i, (s, a, b) in enumerate(zip(secs, tr, va), start=1):
            flat.append({"method": method, "epoch": i, "seconds": s,
                         "train_loss": a, "val_loss": b})
    pd.DataFrame(flat).to_csv(out_dir / "parta_loss_curves.csv", index=False)

    for which, idx, fname, title in [
        ("training", 1, "parta_train_loss_vs_time.png",
         "Part (a): training loss vs wall-clock training time"),
        ("validation", 2, "parta_val_loss_vs_time.png",
         "Part (a): validation loss vs wall-clock training time"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for method in PART_A_METHODS:
            secs, tr, va = curves[method]
            ax.plot(secs, tr if idx == 1 else va, lw=1.6,
                    color=COLOURS[method], label=method)
        ax.set_xlabel("wall-clock training time (s)")
        ax.set_ylabel(f"{which} cross-entropy loss")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)

    df = pd.DataFrame(rows)
    fastest = df.loc[df["train_seconds"].idxmin(), "method"]
    best_loss = df.loc[df["final_train_loss"].idxmin(), "method"]

    lines.append("## Part (a) — loss vs time\n")
    lines.append("Plots: `parta_train_loss_vs_time.png`, "
                 "`parta_val_loss_vs_time.png` (log-scaled time axis).\n")
    lines.append("| method | epochs | train secs | final train loss | final val loss "
                 "| min val loss | at epoch | at secs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['epochs']} | {r['train_seconds']:.2f} "
            f"| {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} "
            f"| {r['min_val_loss']:.4f} | {r['min_val_epoch']} "
            f"| {r['min_val_seconds']:.2f} |")
    lines.append("")
    lines.append(f"- Cheapest in wall-clock training time: **{fastest}**.")
    lines.append(f"- Lowest final training loss: **{best_loss}**.")
    lines.append("- Overfitting onset per method = the epoch of minimum validation "
                 "loss in the table above; validation loss rising after that point "
                 "while training loss keeps falling is the signature.")
    for r in rows:
        tail = "no overfitting within the prescribed epochs" \
            if r["min_val_epoch"] == r["epochs"] else \
            f"val loss bottoms at epoch {r['min_val_epoch']} then rises"
        lines.append(f"  - `{r['method']}`: {tail}.")
    lines.append("")


# --------------------------------------------------------------------------
# Part (c)
# --------------------------------------------------------------------------
def fit_version(mod, name, X_train, y_train, groups, n_given):
    """Returns (phi_fn, extra) where phi_fn(X_raw) -> phi, handling the three
    API shapes across the five versions."""
    if not hasattr(mod, "fit_feature_map"):
        prep = mod.fit_preprocessor(X_train)
        if hasattr(mod, "rank_features"):                       # v2
            Xs = mod.apply_preprocessor(X_train, prep)
            ranking = mod.rank_features(Xs, y_train, groups)
            keep = np.sort(np.argsort(-ranking)[:min(mod.SELECT_DIMS, Xs.shape[1])])
            return (lambda X: mod.apply_preprocessor(X, prep)[:, keep],
                    {"ranking": ranking, "cols": np.arange(X_train.shape[1])})
        return lambda X: mod.apply_preprocessor(X, prep), {}     # v1

    import inspect
    n_params = len(inspect.signature(mod.fit_feature_map).parameters)
    if n_params >= 5:                                            # v5 auto
        use_raw = X_train.shape[1] > n_given
        cands = [c for c in mod.CANDIDATES if use_raw or c not in ("raw", "gam")]
        best, best_m = cands[0], -np.inf
        for cand in cands:
            st = mod.fit_feature_map(X_train, y_train, groups, cand, n_given)
            _, _, m = mod.tune_hyperparameters(
                mod.apply_feature_map(X_train, st), y_train, groups)
            if m > best_m:
                best, best_m = cand, m
        state = mod.fit_feature_map(X_train, y_train, groups, best, n_given)
        return (lambda X: mod.apply_feature_map(X, state),
                {"ranking": state.get("ranking"),
                 "cols": state.get("base_cols"), "chose": best})
    state = mod.fit_feature_map(X_train, y_train, groups)         # v3 / v4
    return (lambda X: mod.apply_feature_map(X, state),
            {"ranking": state.get("ranking"),
             "cols": np.arange(X_train.shape[1])})


def run_part_c(dataset_dir, out_dir, lines):
    train_df = pd.read_csv(Path(dataset_dir) / "train.csv")
    val_path = Path(dataset_dir) / "val.csv"
    if not val_path.exists():
        lines.append("## Part (c)\n\n*Skipped: val.csv not found, so no version "
                     "comparison is possible.*\n")
        return
    val_df = pd.read_csv(val_path)

    results, best_name, best_m, best_pack = [], None, -np.inf, None
    for label, filename in VERSIONS:
        mod = load_module(ROOT / "variants" / filename, filename[:-3])
        tr = mod.drop_excluded_patients(train_df.copy(), "train.csv")
        va = mod.drop_excluded_patients(val_df.copy(), "val.csv")

        feature_cols = mod.get_feature_columns(tr)
        n_given = len(feature_cols)
        y_tr = tr["label"].to_numpy(dtype=np.int64)
        y_va = va["label"].to_numpy(dtype=np.int64)
        groups = tr["patient"].to_numpy()

        use_raw = hasattr(mod, "_raw_feature_names") and \
            (Path(dataset_dir) / "raw_signals").is_dir()
        raw_names = mod._raw_feature_names() if use_raw else []
        Xtr = mod.build_matrix(tr, feature_cols, dataset_dir, raw_names, use_raw) \
            if hasattr(mod, "build_matrix") else tr[feature_cols].to_numpy(float)
        Xva = mod.build_matrix(va, feature_cols, dataset_dir, raw_names, use_raw) \
            if hasattr(mod, "build_matrix") else va[feature_cols].to_numpy(float)

        phi_fn, extra = fit_version(mod, label, Xtr, y_tr, groups, n_given)
        phi_tr, phi_va = phi_fn(Xtr), phi_fn(Xva)

        c, cw, cv_m = mod.tune_hyperparameters(phi_tr, y_tr, groups)
        model = mod.make_model(c, cw)
        model.fit(phi_tr, y_tr)
        prob_va = model.predict_proba(phi_va)[:, 1]

        oof = mod.grouped_oof_probs(phi_tr, y_tr, groups, c, cw)
        thr, _ = mod.select_threshold(np.concatenate([y_tr, y_va]),
                                      np.concatenate([oof, prob_va]))
        m_at, tp, fp, tpr, fpr = mod.m_metric(y_va, prob_va, thr)
        m_half, tp5, fp5, tpr5, fpr5 = mod.m_metric(y_va, prob_va, 0.5)
        ap = float(average_precision_score(y_va, prob_va)) \
            if len(np.unique(y_va)) > 1 else float("nan")

        results.append(dict(label=label, d=phi_tr.shape[1], C=c, cw=str(cw),
                            cv_m=cv_m, thr=thr, M=m_at, TP=tp, FP=fp, recall=tpr,
                            fpr=fpr, AP=ap, M_half=m_half, TP_half=tp5,
                            FP_half=fp5, recall_half=tpr5, fpr_half=fpr5,
                            chose=extra.get("chose", "")))
        print(f"[part c] {label}: d={phi_tr.shape[1]} val M={m_at:.4f} "
              f"(TP={tp} FP={fp})", file=sys.stderr)

        if m_at > best_m:
            best_m, best_name = m_at, label
            names = feature_cols + raw_names
            best_pack = (mod, y_va, prob_va, extra, names, y_tr, phi_tr,
                         phi_va, groups, c, cw)

    lines.append("## Part (c) — all five versions on val.csv\n")
    lines.append("Chosen threshold from pooled patient-grouped OOF on train plus "
                 "val; `M = TPR - 100*FPR`.\n")
    lines.append("| version | d | C | class_weight | grouped-CV M | threshold "
                 "| **val M** | TP | FP | recall | FPR | AP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['label']}{(' -> ' + r['chose']) if r['chose'] else ''} "
            f"| {r['d']} | {r['C']} | {r['cw']} | {r['cv_m']:.4f} "
            f"| {r['thr']:.4f} | **{r['M']:.4f}** | {r['TP']} | {r['FP']} "
            f"| {r['recall']:.4f} | {r['fpr']:.4f} | {r['AP']:.4f} |")
    lines.append("")
    lines.append(f"**Best on val: {best_name} (M = {best_m:.4f}).**\n")

    lines.append("### (iv) Final metrics — chosen threshold vs 0.5\n")
    lines.append("| version | setting | M | TP | FP | recall | FPR |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['label']} | chosen {r['thr']:.4f} | {r['M']:.4f} "
                     f"| {r['TP']} | {r['FP']} | {r['recall']:.4f} | {r['fpr']:.4f} |")
        lines.append(f"| {r['label']} | default 0.5 | {r['M_half']:.4f} "
                     f"| {r['TP_half']} | {r['FP_half']} | {r['recall_half']:.4f} "
                     f"| {r['fpr_half']:.4f} |")
    lines.append("")

    (mod, y_va, prob_va, extra, names, y_tr, phi_tr, phi_va,
     groups, c, cw) = best_pack

    ranking = extra.get("ranking")
    lines.append("### (i) Top 10 features of the best version\n")
    if ranking is not None:
        cols = extra.get("cols")
        cols = np.arange(len(ranking)) if cols is None else cols
        lines.append("Ranked by mean out-of-fold |AUC - 0.5| over patient-grouped "
                     "folds.\n")
        lines.append("| rank | feature | mean OOF \\|AUC-0.5\\| |")
        lines.append("|---|---|---|")
        for rank, j in enumerate(np.argsort(-ranking)[:10], start=1):
            nm = names[cols[j]] if cols[j] < len(names) else f"col_{cols[j]}"
            lines.append(f"| {rank} | `{nm}` | {ranking[j]:.4f} |")
    else:
        lines.append("*This version uses every column and does not rank them; "
                     "run v2 or v3 for a ranking.*")
    lines.append("")

    lines.append("### (ii) Class imbalance\n")
    lines.append("| class_weight | threshold | M | TP | FP | recall | FPR | AP |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for setting in (None, "balanced"):
        alt = mod.make_model(c, setting)
        alt.fit(phi_tr, y_tr)
        alt_prob = alt.predict_proba(phi_va)[:, 1]
        alt_oof = mod.grouped_oof_probs(phi_tr, y_tr, groups, c, setting)
        t, _ = mod.select_threshold(np.concatenate([y_tr, y_va]),
                                    np.concatenate([alt_oof, alt_prob]))
        m, tp, fp, tpr, fpr = mod.m_metric(y_va, alt_prob, t)
        ap = float(average_precision_score(y_va, alt_prob))
        lines.append(f"| {setting} | {t:.4f} | {m:.4f} | {tp} | {fp} "
                     f"| {tpr:.4f} | {fpr:.4f} | {ap:.4f} |")
    lines.append("")

    # ---- required plots, on the best version ----
    thr_best = [r for r in results if r["label"] == best_name][0]["thr"]
    t_curve, m_curve, _, _ = mod.sweep_thresholds(y_va, prob_va)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t_curve, m_curve, lw=1.6, color="#1f4e79", label="M on val.csv")
    ax.axvline(thr_best, color="#c0392b", ls="--", lw=1.4,
               label=f"chosen = {thr_best:.3f}")
    ax.axvline(0.5, color="#7f8c8d", ls=":", lw=1.2, label="default 0.5")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("decision threshold on P(AF)")
    ax.set_ylabel(r"$M = TPR - FPR/\epsilon$  ($\epsilon = 1/100$)")
    ax.set_title(f"Part (c): M vs threshold on val.csv — {best_name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "partc_m_vs_threshold.png", dpi=160)
    plt.close(fig)

    alphas = np.linspace(10.0, 1000.0, 200)
    best_ms, best_ts = [], []
    for a in alphas:
        t_a, m_a, tpr_a, _ = mod.sweep_thresholds(y_va, prob_va, a)
        ok = tpr_a >= mod.MIN_TPR_ELIGIBLE
        if not np.any(ok):
            best_ms.append(np.nan), best_ts.append(np.nan)
            continue
        i = int(np.argmax(np.where(ok, m_a, -np.inf)))
        best_ms.append(m_a[i]), best_ts.append(t_a[i])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(alphas, best_ms, lw=1.6, color="#1f4e79")
    axes[0].set_xlabel(r"$1/\epsilon$"), axes[0].set_ylabel("best achievable M")
    axes[0].set_title(r"(a) best M vs $1/\epsilon$"), axes[0].grid(alpha=0.3)
    axes[1].plot(alphas, best_ts, lw=1.6, color="#c0392b")
    axes[1].set_xlabel(r"$1/\epsilon$"), axes[1].set_ylabel("threshold at best M")
    axes[1].set_title(r"(b) arg-max threshold vs $1/\epsilon$"), axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "partc_epsilon_sweep.png", dpi=160)
    plt.close(fig)

    lines.append("### (iii) Threshold selection\n")
    anchor = mod.theoretical_threshold(np.concatenate([y_tr, y_va]))
    lines.append(f"- chosen threshold: **{thr_best:.4f}**")
    lines.append(f"- calibrated-optimal anchor `alpha*P/(N+alpha*P)` = "
                 f"**{anchor:.4f}**")
    lines.append("- Plots: `partc_m_vs_threshold.png`, "
                 "`partc_epsilon_sweep.png`.\n")


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    part_ab_dir, partc_dir, out = sys.argv[1:4]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Report data", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    if part_ab_dir.lower() != "skip":
        run_part_a(part_ab_dir, out_dir, lines)
    if partc_dir.lower() != "skip":
        run_part_c(partc_dir, out_dir, lines)

    (out_dir / "REPORT_DATA.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_dir/'REPORT_DATA.md'} and the figures beside it",
          file=sys.stderr)


if __name__ == "__main__":
    main()
