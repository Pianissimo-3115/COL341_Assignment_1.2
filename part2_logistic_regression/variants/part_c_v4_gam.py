"""
COL774 Assignment 1 - Part 2(c), VERSION 4: raw features + spline basis (GAM).

    python3 part_c.py dataset_dir/ model.pkl final_features.csv

`dataset_dir` holds train.csv / val.csv / test.csv (release_id, patient, label,
392 feature columns) and raw_signals/{patient}.npz (release_id, signal). The
script fits a feature map phi(x), trains a binary logistic regression on it,
picks a decision threshold, then writes

    model.pkl            {'weights': (d,), 'bias': float, 'threshold': float}
    final_features.csv   feat_0..feat_{d-1} + release_id, one row per test row

and is scored as sigmoid(phi(x) . weights + bias) >= threshold.

Why the pipeline looks the way it does
--------------------------------------
The metric is M = TP/P - FP/(N*eps) with eps = 1/100, i.e. M = TPR - 100*FPR.
One false positive costs as much as 100/N recall, so on a val set with N ~ 1000
non-AF windows a single FP burns 0.1 of M while a single extra true positive
buys only 1/P ~ 0.01. Everything below is built around "be almost certain
before calling AF":

  * winsorising every feature to train-set quantiles, so a single wild outlier
    at test time cannot produce a huge logit and a confident false positive;
  * signal-quality features, because noise causes spurious QRS detections,
    which look exactly like the irregular RR series that means AF - quality
    features let the model discount those windows instead of firing on them;
  * a conservative threshold rule (take the *largest* threshold whose M is
    within a tolerance of the best) rather than the raw arg-max, which
    overfits the validation sample;
  * patient-grouped CV everywhere, since rows from one patient are highly
    correlated and ungrouped folds would badly over-estimate performance.

What this version adds over versions 1 and 2
--------------------------------------------
Versions 1 and 2 only reshuffle the 392 supplied columns. This one goes back to
raw_signals/ and computes its own features, which is where the marks are: the
supplied columns come from the published Goodfellow/ecg-features library and
are dominated by RR/HRV statistics, so they largely miss the *other* defining
sign of AF - the loss of organised atrial activity - and they cannot tell a
genuinely irregular rhythm from a noisy recording.

The added features re-implement published, openly described methods:

  * Pan-Tompkins QRS detection, then RR-irregularity measures: COSEn (Lake &
    Moorman), symbolic-dynamics Shannon entropy (Zhou et al.), Lorenz-plot dRR
    occupancy (Sarkar et al.), Poincare SD1/SD2, turning-point ratio.
  * Atrial activity via zero-padding QRST cancellation: subtract an average
    beat at every R peak, then measure P-wave amplitude and beat-to-beat
    reproducibility, and the 4-9 Hz fibrillatory-wave band power in the TQ
    intervals where atrial activity is unobscured.
  * Signal-quality descriptors, which exist specifically to prevent false
    positives: noise causes spurious QRS detections, which look exactly like
    the irregular RR series that means AF. At 100x cost per false positive,
    letting the model see that a recording is untrustworthy matters more than
    any classifier tweak.

The selected columns are then ranked down to SELECT_DIMS as in version 2.

What this version adds over version 3
-------------------------------------
The classifier has to stay a linear logistic regression, but nothing stops
phi(x) from being a richer basis. Several of the strongest AF features are
informative in a non-monotone way - heart rate is the clearest case, where both
unusually fast and unusually slow rhythms are suspicious while normal rates are
not, so no single linear coefficient can express it. This version therefore
spline-expands the GAM_TOP_K strongest columns (cubic B-splines, GAM_N_KNOTS
knots) and leaves the rest linear, turning the model into a generalised
additive model that is still exactly sigmoid(phi(x) . w + b) as the spec
requires. The kept-column count is trimmed if needed so that d stays under 500.

If raw_signals/ is absent or unreadable the raw features are skipped and this
degrades to version 2's behaviour rather than failing.

Disclosure (for report.pdf)
---------------------------
Written with assistance from Claude (Anthropic). No third-party source code is
copied into this file; only NumPy, SciPy and scikit-learn are imported. The
raw-signal features re-implement published, openly described methods, cited at
their definitions below: Pan-Tompkins QRS detection; Lake & Moorman's
coefficient of sample entropy (COSEn); Zhou et al.'s symbolic-dynamics Shannon
entropy; Sarkar et al.'s Lorenz-plot dRR analysis; and zero-padding QRST
cancellation for fibrillatory-wave analysis.
"""
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats as spstats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_SEED = 774
EPSILON = 1.0 / 100.0
ALPHA = 1.0 / EPSILON               # M = TPR - ALPHA * FPR
MIN_TPR_ELIGIBLE = 0.10             # below this the submission is not graded
MIN_TPR_TARGET = 0.15               # our own safety margin above that
MAX_FEATURE_DIM = 480               # spec requires d < 500

N_FOLDS = 5
WINSOR_Q = 0.001                    # clip features to train [0.1%, 99.9%]
SELECT_DIMS = 220                   # columns kept by the AUC ranking
GAM_TOP_K = 45                      # strongest columns given a spline basis
GAM_N_KNOTS = 5
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CLASS_WEIGHT_GRID = (None, "balanced")
THRESHOLD_TOLERANCE = 0.01          # give up this much M to gain FP headroom

# Raw-signal extraction is capped so the whole run stays inside Kaggle's
# 40-minute budget; rows not reached keep NaN and are imputed like any other
# missing value.
RAW_TIME_BUDGET_S = 900.0
WINDOW_SECONDS = 30.0

NON_FEATURE_COLUMNS = {"release_id", "patient", "label"}

# Patients the course flagged as genuinely hard to classify (announcement:
# validation performance looks unexpectedly poor, "particularly for patient 051
# in val.csv and 045 in train.csv"). Dropped from training and from threshold
# selection only - never from test.csv, where every row must still be scored.
# Set to () to keep them.
EXCLUDE_PATIENTS = ("045", "051")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
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
# Raw-signal feature extraction
# --------------------------------------------------------------------------
def _bandpass(x, fs, lo, hi, order=2):
    nyq = 0.5 * fs
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    if lo_n <= 0 or lo_n >= hi_n:
        return x
    b, a = sps.butter(order, [lo_n, hi_n], btype="band")
    return sps.filtfilt(b, a, x)


def detect_rpeaks(x, fs):
    """Pan-Tompkins style QRS detection (bandpass, derivative, square,
    moving-window integrate, adaptive peak picking). Returns sample indices."""
    xf = _bandpass(x, fs, 5.0, 15.0)
    if not np.all(np.isfinite(xf)):
        return np.array([], dtype=int), x
    squared = np.gradient(xf) ** 2
    win = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    peak_scale = np.percentile(integrated, 98)
    if not np.isfinite(peak_scale) or peak_scale <= 0:
        return np.array([], dtype=int), xf

    min_gap = max(1, int(round(0.25 * fs)))          # 240 bpm ceiling
    peaks, props = sps.find_peaks(integrated, height=0.30 * peak_scale,
                                  distance=min_gap)
    if peaks.size == 0:
        return np.array([], dtype=int), xf

    # Second pass: re-threshold against the detected peaks themselves, which
    # adapts to recordings whose overall amplitude is unusual.
    heights = props["peak_heights"]
    peaks = peaks[heights >= 0.30 * np.median(heights)]

    # Snap each detection onto the nearby |bandpassed| maximum.
    half = max(1, int(round(0.060 * fs)))
    refined = []
    for p in peaks:
        lo, hi = max(0, p - half), min(xf.size, p + half + 1)
        refined.append(lo + int(np.argmax(np.abs(xf[lo:hi]))))
    return np.unique(np.asarray(refined, dtype=int)), xf


def _match_count(series, m, r):
    """Number of Chebyshev-distance matches among length-m embeddings."""
    n = series.size - m + 1
    if n < 2:
        return 0
    emb = np.lib.stride_tricks.sliding_window_view(series, m)
    dist = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
    np.fill_diagonal(dist, np.inf)
    return int(np.count_nonzero(dist <= r) // 2)


def sample_entropy(series, m=2, r_ratio=0.2):
    series = np.asarray(series, dtype=np.float64)
    if series.size < m + 2:
        return np.nan
    r = r_ratio * np.std(series)
    if r <= 0:
        return np.nan
    b = _match_count(series, m, r)
    a = _match_count(series, m + 1, r)
    if a <= 0 or b <= 0:
        return np.nan
    return -np.log(a / b)


def cosen(rr, m=1, min_matches=5):
    """Coefficient of sample entropy (Lake & Moorman), the standard entropy
    detector for AF in short RR series: the tolerance r is widened until the
    template match count is usable, then the density is corrected for scale.

        COSEn = SampEn(m, r) + ln(2r) - ln(mean RR)
    """
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < m + 3:
        return np.nan
    for r in np.arange(0.010, 0.305, 0.005):
        b = _match_count(rr, m, r)
        a = _match_count(rr, m + 1, r)
        if b >= min_matches and a >= min_matches:
            return -np.log(a / b) + np.log(2.0 * r) - np.log(np.mean(rr))
    return np.nan


def symbolic_shannon_entropy(rr, thresh=0.02, word_len=3):
    """Symbolic-dynamics Shannon entropy (Zhou et al.): quantise successive RR
    differences into {fall, flat, rise}, read words of `word_len` symbols and
    take the normalised Shannon entropy of the word histogram. AF spreads mass
    over many word types; sinus rhythm concentrates it on the 'flat' word."""
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < word_len + 2:
        return np.nan, np.nan
    d = np.diff(rr)
    symbols = np.where(d < -thresh, 0, np.where(d > thresh, 2, 1))
    if symbols.size < word_len:
        return np.nan, np.nan
    words = np.lib.stride_tricks.sliding_window_view(symbols, word_len)
    codes = words @ (3 ** np.arange(word_len))
    counts = np.bincount(codes, minlength=3 ** word_len).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(3 ** word_len))
    return entropy, float(np.count_nonzero(counts) / (3 ** word_len))


def lorenz_features(rr):
    """Occupancy statistics of the dRR Lorenz plot (Sarkar et al.). AF scatters
    points over a wide cloud; sinus rhythm piles them at the origin; isolated
    ectopy makes a sparse, structured pattern instead of a diffuse one."""
    out = dict.fromkeys(
        ["lz_occupied", "lz_origin_frac", "lz_entropy", "lz_radius_mean",
         "lz_radius_std", "lz_pac_frac", "lz_irregularity"], np.nan)
    rr = np.asarray(rr, dtype=np.float64)
    if rr.size < 4:
        return out
    d = np.diff(rr) * 1000.0                       # ms
    if d.size < 2:
        return out
    x, y = d[:-1], d[1:]
    lim, bin_ms = 600.0, 25.0
    xc = np.clip(x, -lim, lim - 1e-9)
    yc = np.clip(y, -lim, lim - 1e-9)
    nbin = int(2 * lim / bin_ms)
    ix = ((xc + lim) / bin_ms).astype(int)
    iy = ((yc + lim) / bin_ms).astype(int)
    hist = np.bincount(ix * nbin + iy, minlength=nbin * nbin).astype(np.float64)
    occupied = np.count_nonzero(hist)
    p = hist[hist > 0] / hist.sum()

    radius = np.hypot(x, y)
    origin = np.count_nonzero(radius <= 50.0) / radius.size
    # A premature beat gives a short RR immediately followed by a long one,
    # i.e. points in the second/fourth quadrant far from the diagonal.
    pac = np.count_nonzero((np.sign(x) != np.sign(y)) & (radius > 100.0))

    out.update(
        lz_occupied=float(occupied),
        lz_origin_frac=float(origin),
        lz_entropy=float(-(p * np.log(p)).sum()),
        lz_radius_mean=float(np.mean(radius)),
        lz_radius_std=float(np.std(radius)),
        lz_pac_frac=float(pac / radius.size),
        lz_irregularity=float(occupied / radius.size),
    )
    return out


def rr_interval_features(rpeaks, fs):
    """Rhythm-irregularity features. AF is 'irregularly irregular', so these
    carry most of the discriminative signal."""
    names = [
        "rr_n", "rr_valid_frac", "rr_mean", "rr_std", "rr_cv", "rr_median",
        "rr_iqr", "rr_min", "rr_max", "rr_range", "rr_rmssd", "rr_nrmssd",
        "rr_pnn20", "rr_pnn50", "rr_pnn70", "rr_mad_drr", "rr_skew",
        "rr_kurtosis", "rr_drr_std", "rr_drr_skew", "rr_sd1", "rr_sd2",
        "rr_sd_ratio", "rr_ellipse_area", "rr_acf1", "rr_acf2", "rr_tpr",
        "rr_hjorth_mobility", "rr_hjorth_complexity", "rr_hist_entropy",
        "rr_sampen", "rr_cosen", "rr_sym_entropy", "rr_sym_coverage",
        "rr_hr_bpm", "rr_hr_std_bpm", "rr_frac_short", "rr_frac_long",
    ]
    out = dict.fromkeys(names, np.nan)
    if rpeaks.size < 4:
        out["rr_n"] = float(max(rpeaks.size - 1, 0))
        return out

    all_rr = np.diff(rpeaks) / float(fs)
    valid = all_rr[(all_rr >= 0.30) & (all_rr <= 2.50)]
    out["rr_n"] = float(valid.size)
    out["rr_valid_frac"] = float(valid.size / all_rr.size)
    if valid.size < 4:
        return out

    d = np.diff(valid)
    mean_rr = float(np.mean(valid))
    out.update(
        rr_mean=mean_rr,
        rr_std=float(np.std(valid)),
        rr_cv=float(np.std(valid) / mean_rr),
        rr_median=float(np.median(valid)),
        rr_iqr=float(np.subtract(*np.percentile(valid, [75, 25]))),
        rr_min=float(np.min(valid)),
        rr_max=float(np.max(valid)),
        rr_range=float(np.ptp(valid)),
        rr_rmssd=float(np.sqrt(np.mean(d ** 2))),
        rr_nrmssd=float(np.sqrt(np.mean(d ** 2)) / mean_rr),
        rr_pnn20=float(np.mean(np.abs(d) > 0.020)),
        rr_pnn50=float(np.mean(np.abs(d) > 0.050)),
        rr_pnn70=float(np.mean(np.abs(d) > 0.070)),
        rr_mad_drr=float(1.4826 * np.median(np.abs(d - np.median(d)))),
        rr_skew=float(spstats.skew(valid)),
        rr_kurtosis=float(spstats.kurtosis(valid)),
        rr_drr_std=float(np.std(d)),
        rr_drr_skew=float(spstats.skew(d)) if d.size > 2 else np.nan,
        rr_hr_bpm=float(60.0 / mean_rr),
        rr_hr_std_bpm=float(np.std(60.0 / valid)),
        rr_frac_short=float(np.mean(valid < 0.6 * np.median(valid))),
        rr_frac_long=float(np.mean(valid > 1.4 * np.median(valid))),
    )

    sd1 = np.sqrt(0.5) * np.std(d)
    sd2 = np.sqrt(max(0.0, 2.0 * np.var(valid) - 0.5 * np.var(d)))
    out.update(
        rr_sd1=float(sd1),
        rr_sd2=float(sd2),
        rr_sd_ratio=float(sd1 / sd2) if sd2 > 0 else np.nan,
        rr_ellipse_area=float(np.pi * sd1 * sd2),
    )

    centred = valid - mean_rr
    denom = float(np.sum(centred ** 2))
    if denom > 0:
        out["rr_acf1"] = float(np.sum(centred[:-1] * centred[1:]) / denom)
        if valid.size > 3:
            out["rr_acf2"] = float(np.sum(centred[:-2] * centred[2:]) / denom)

    if valid.size >= 3:
        interior = valid[1:-1]
        turning = np.count_nonzero(
            ((interior > valid[:-2]) & (interior > valid[2:]))
            | ((interior < valid[:-2]) & (interior < valid[2:])))
        out["rr_tpr"] = float(turning / interior.size)

    var_rr = np.var(valid)
    if var_rr > 0 and d.size > 1:
        mobility = np.sqrt(np.var(d) / var_rr)
        out["rr_hjorth_mobility"] = float(mobility)
        d2 = np.diff(d)
        if np.var(d) > 0 and d2.size > 0 and mobility > 0:
            out["rr_hjorth_complexity"] = float(
                np.sqrt(np.var(d2) / np.var(d)) / mobility)

    hist, _ = np.histogram(valid, bins=16, range=(0.30, 2.50))
    p = hist[hist > 0] / max(hist.sum(), 1)
    if p.size:
        out["rr_hist_entropy"] = float(-(p * np.log(p)).sum() / np.log(16))

    out["rr_sampen"] = sample_entropy(valid, m=2, r_ratio=0.2)
    out["rr_cosen"] = cosen(valid)
    sym_entropy, sym_coverage = symbolic_shannon_entropy(valid)
    out["rr_sym_entropy"] = sym_entropy
    out["rr_sym_coverage"] = sym_coverage
    out.update(lorenz_features(valid))
    return out


def atrial_activity_features(x, xf, fs, rpeaks):
    """P-wave / fibrillatory-wave evidence.

    AF's other defining sign is the loss of organised atrial depolarisation:
    no P wave, replaced by 4-9 Hz fibrillatory waves. We build an average beat,
    subtract it at every R peak (zero-padding QRST cancellation) and analyse
    what is left in the TQ intervals, where atrial activity is unobscured.
    """
    names = [
        "aa_p_amp", "aa_p_energy", "aa_p_corr_mean", "aa_p_corr_std",
        "aa_p_energy_cv", "aa_pt_ratio", "aa_beat_corr_mean",
        "aa_beat_corr_std", "aa_beat_corr_p10", "aa_beat_good_frac",
        "aa_amp_cv", "aa_resid_ratio", "aa_tq_rms", "aa_tq_dom_freq",
        "aa_tq_band_ratio", "aa_tq_spec_entropy", "aa_tq_peak_ratio",
        "aa_tq_frac",
    ]
    out = dict.fromkeys(names, np.nan)
    if rpeaks.size < 4:
        return out

    rr = np.diff(rpeaks)
    med_rr = int(np.median(rr))
    if med_rr < int(0.3 * fs):
        return out

    pre = int(0.35 * med_rr)
    post = int(0.45 * med_rr)
    keep = rpeaks[(rpeaks - pre >= 0) & (rpeaks + post < xf.size)]
    if keep.size < 3:
        return out

    beats = np.stack([xf[p - pre: p + post] for p in keep])
    template = np.median(beats, axis=0)
    t_norm = float(template @ template)
    if t_norm <= 0:
        return out

    # Least-squares amplitude fit of each beat onto the template, so a beat
    # that only differs in gain is not counted as morphologically different.
    scales = (beats @ template) / t_norm
    residuals = beats - scales[:, None] * template

    beat_sd = beats.std(axis=1)
    ok = beat_sd > 0
    corr = np.full(beats.shape[0], np.nan)
    if template.std() > 0 and np.any(ok):
        corr[ok] = ((beats[ok] - beats[ok].mean(axis=1, keepdims=True))
                    @ (template - template.mean())) / (
            beats.shape[1] * beat_sd[ok] * template.std())

    finite = corr[np.isfinite(corr)]
    if finite.size:
        out.update(
            aa_beat_corr_mean=float(np.mean(finite)),
            aa_beat_corr_std=float(np.std(finite)),
            aa_beat_corr_p10=float(np.percentile(finite, 10)),
            aa_beat_good_frac=float(np.mean(finite > 0.90)),
        )
    if np.mean(np.abs(scales)) > 0:
        out["aa_amp_cv"] = float(np.std(scales) / np.mean(np.abs(scales)))
    beat_energy = float(np.mean(beats ** 2))
    if beat_energy > 0:
        out["aa_resid_ratio"] = float(np.mean(residuals ** 2) / beat_energy)

    # P-wave window: just before the QRS, where the P wave lives in sinus
    # rhythm. In AF it is absent, so both its amplitude and - more tellingly -
    # its beat-to-beat reproducibility collapse.
    p_lo = max(0, pre - int(0.30 * med_rr))
    p_hi = max(p_lo + 2, pre - int(0.08 * med_rr))
    p_band = beats[:, p_lo:p_hi]
    if p_band.shape[1] >= 3:
        p_template = p_band.mean(axis=0)
        out["aa_p_amp"] = float(np.mean(np.abs(p_band)))
        p_energy = np.mean(p_band ** 2, axis=1)
        out["aa_p_energy"] = float(np.mean(p_energy))
        if np.mean(p_energy) > 0:
            out["aa_p_energy_cv"] = float(np.std(p_energy) / np.mean(p_energy))
        if p_template.std() > 0:
            sd = p_band.std(axis=1)
            good = sd > 0
            if np.any(good):
                pc = ((p_band[good] - p_band[good].mean(axis=1, keepdims=True))
                      @ (p_template - p_template.mean())) / (
                    p_band.shape[1] * sd[good] * p_template.std())
                pc = pc[np.isfinite(pc)]
                if pc.size:
                    out["aa_p_corr_mean"] = float(np.mean(pc))
                    out["aa_p_corr_std"] = float(np.std(pc))

        t_lo = min(beats.shape[1] - 2, pre + int(0.15 * med_rr))
        t_hi = min(beats.shape[1], pre + int(0.40 * med_rr))
        if t_hi - t_lo >= 3:
            t_energy = float(np.mean(beats[:, t_lo:t_hi] ** 2))
            out["aa_pt_ratio"] = float(out["aa_p_energy"] / (t_energy + 1e-12))

    # QRST cancellation: remove the scaled template at every beat, then keep
    # only TQ samples (zero elsewhere) before taking the spectrum.
    cancelled = xf.copy()
    for scale, p in zip(scales, keep):
        cancelled[p - pre: p + post] -= scale * template

    mask = np.zeros(xf.size, dtype=bool)
    for i, p in enumerate(keep[:-1]):
        step = keep[i + 1] - p
        lo = p + int(0.40 * step)
        hi = keep[i + 1] - int(0.10 * step)
        if hi > lo:
            mask[lo:hi] = True
    out["aa_tq_frac"] = float(mask.mean())
    if mask.sum() < int(0.5 * fs):
        return out

    tq = cancelled[mask]
    out["aa_tq_rms"] = float(np.sqrt(np.mean(tq ** 2)))

    gated = np.where(mask, cancelled, 0.0)
    nper = min(gated.size, int(4 * fs))
    freqs, psd = sps.welch(gated, fs=fs, nperseg=nper)
    band = (freqs >= 1.0) & (freqs <= 20.0)
    if np.any(band) and psd[band].sum() > 0:
        fb, pb = freqs[band], psd[band]
        pn = pb / pb.sum()
        out["aa_tq_spec_entropy"] = float(
            -(pn[pn > 0] * np.log(pn[pn > 0])).sum() / np.log(pn.size))
        fib = (fb >= 4.0) & (fb <= 9.0)
        if np.any(fib):
            out["aa_tq_band_ratio"] = float(pb[fib].sum() / pb.sum())
            out["aa_tq_dom_freq"] = float(fb[fib][np.argmax(pb[fib])])
            out["aa_tq_peak_ratio"] = float(pb[fib].max() / np.median(pb))
    return out


def signal_quality_features(x, xf, fs, rpeaks):
    """Noise descriptors. These exist to *prevent* false positives: a noisy
    strip produces spurious QRS detections and therefore fake RR irregularity,
    which is indistinguishable from AF unless the model can see that the
    recording is untrustworthy."""
    out = dict.fromkeys(
        ["q_std", "q_kurtosis", "q_skew", "q_flat_frac", "q_sat_frac",
         "q_baseline_ratio", "q_hf_ratio", "q_qrs_snr", "q_zero_cross",
         "q_amp_iqr", "q_range", "q_peak_rate"], np.nan)
    if x.size == 0:
        return out

    std = float(np.std(x))
    out.update(
        q_std=std,
        q_kurtosis=float(spstats.kurtosis(x)),
        q_skew=float(spstats.skew(x)),
        q_range=float(np.ptp(x)),
        q_amp_iqr=float(np.subtract(*np.percentile(x, [75, 25]))),
        q_zero_cross=float(np.mean(np.diff(np.signbit(x - np.median(x))) != 0)),
        q_peak_rate=float(rpeaks.size / (x.size / fs)),
    )
    if std > 0:
        out["q_flat_frac"] = float(np.mean(np.abs(np.diff(x)) < 1e-6))
        out["q_sat_frac"] = float(np.mean(np.abs(x - np.median(x)) > 6 * std))

    nper = min(x.size, int(4 * fs))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nper)
    total = psd.sum()
    if total > 0:
        out["q_baseline_ratio"] = float(psd[freqs < 0.7].sum() / total)
        out["q_hf_ratio"] = float(psd[freqs > 40.0].sum() / total)

    # Energy inside QRS windows versus everywhere else: a clean ECG is mostly
    # quiet between beats, a noisy one is not.
    if rpeaks.size >= 3:
        qrs = np.zeros(xf.size, dtype=bool)
        half = max(1, int(0.06 * fs))
        for p in rpeaks:
            qrs[max(0, p - half): min(xf.size, p + half)] = True
        if qrs.any() and (~qrs).any():
            off = float(np.mean(xf[~qrs] ** 2))
            out["q_qrs_snr"] = float(np.mean(xf[qrs] ** 2) / (off + 1e-12))
    return out


def raw_window_features(x, fs):
    """All raw-signal features for one 30-second window."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.median(np.abs(x - np.median(x)))
    if scale > 0:
        x = (x - np.median(x)) / (1.4826 * scale)

    rpeaks, xf = detect_rpeaks(x, fs)
    feats = {}
    feats.update(rr_interval_features(rpeaks, fs))
    feats.update(atrial_activity_features(x, xf, fs, rpeaks))
    feats.update(signal_quality_features(x, xf, fs, rpeaks))
    return feats


def _raw_feature_names():
    dummy = raw_window_features(np.zeros(16), 250.0)
    return sorted(dummy.keys())


def extract_raw_features(df, dataset_dir, feature_names, time_budget=RAW_TIME_BUDGET_S):
    """Raw features for every row of `df`, aligned to df's row order.

    Rows whose signal cannot be found - missing raw_signals/, missing patient
    file, unknown release_id, or the time budget running out - stay NaN and are
    imputed downstream, so this never fails the run.
    """
    out = np.full((len(df), len(feature_names)), np.nan, dtype=np.float64)
    raw_dir = Path(dataset_dir) / "raw_signals"
    if not raw_dir.is_dir():
        print("raw_signals/ not found - continuing without raw features",
              file=sys.stderr)
        return out

    npz_paths = sorted(raw_dir.glob("*.npz"))
    if not npz_paths:
        print("raw_signals/ has no .npz files", file=sys.stderr)
        return out

    # Map release_id -> file, reading only the (small) id arrays first so we
    # never hold more than one patient's signals in memory.
    location = {}
    for path in npz_paths:
        try:
            with np.load(path, allow_pickle=False) as z:
                ids = z["release_id"]
        except Exception as exc:                       # unreadable file
            print(f"skipping {path.name}: {exc}", file=sys.stderr)
            continue
        for row, rid in enumerate(ids):
            location[rid] = (path, row)

    wanted = {}
    for i, rid in enumerate(df["release_id"].to_numpy()):
        hit = location.get(rid)
        if hit is None:
            continue
        wanted.setdefault(hit[0], []).append((i, hit[1]))

    if not wanted:
        print("no release_id matched raw_signals/ - continuing without raw "
              "features", file=sys.stderr)
        return out

    index = {name: j for j, name in enumerate(feature_names)}
    started = time.time()
    done = 0
    for path, rows in wanted.items():
        if time.time() - started > time_budget:
            print(f"raw-feature time budget hit after {done} windows",
                  file=sys.stderr)
            break
        try:
            with np.load(path, allow_pickle=False) as z:
                signals = z["signal"]
        except Exception as exc:
            print(f"skipping {path.name}: {exc}", file=sys.stderr)
            continue
        fs = max(1.0, round(signals.shape[1] / WINDOW_SECONDS))
        for df_row, sig_row in rows:
            if time.time() - started > time_budget:
                break
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    feats = raw_window_features(signals[sig_row], fs)
                except Exception:
                    continue                            # leave the row NaN
            for name, value in feats.items():
                j = index.get(name)
                if j is not None and np.isfinite(value):
                    out[df_row, j] = value
            done += 1
        del signals
    print(f"raw features: {done}/{len(df)} windows in "
          f"{time.time() - started:.1f}s", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# Tabular preprocessing - every statistic is fit on train only
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
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0, ddof=0)
    return {"median": median, "lo": lo, "hi": hi, "mean": mean,
            "std": np.where(std > 0, std, 1.0)}


def apply_preprocessor(X, state):
    filled = np.where(np.isfinite(X), X, state["median"])
    clipped = np.clip(filled, state["lo"], state["hi"])
    return (clipped - state["mean"]) / state["std"]


# --------------------------------------------------------------------------
# Scoring, feature ranking, threshold choice
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


def auc_score(y_true, score):
    """Rank-based AUC (ties averaged); avoids a sklearn.metrics import."""
    y_true = np.asarray(y_true)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    finite = np.isfinite(score)
    if not finite.all():
        score = np.where(finite, score, np.nanmedian(score[finite])
                         if finite.any() else 0.0)
    ranks = spstats.rankdata(score)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def rank_features(X, y, groups, n_folds=N_FOLDS):
    """Score every column by out-of-fold |AUC - 0.5|, averaged over
    patient-grouped folds so a column that only works on a few patients does
    not outrank one that generalises."""
    n_folds = max(2, min(n_folds, len(np.unique(groups))))
    scores = np.zeros((n_folds, X.shape[1]))
    for f, (_, idx) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        if len(np.unique(y[idx])) < 2:
            continue
        for j in range(X.shape[1]):
            scores[f, j] = abs(auc_score(y[idx], X[idx, j]) - 0.5)
    return scores.mean(axis=0)


def sweep_thresholds(y_true, prob, alpha=ALPHA):
    """M, TPR and FPR at every distinct cut of `prob`, descending by threshold.

    Exact and O(n log n): sorting once and taking cumulative counts beats
    evaluating a fixed grid, and it never misses the optimum between grid
    points.
    """
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

    The arg-max of M on a finite sample sits right where a couple of negatives
    happen to fall below the cut; a slightly stricter threshold usually costs
    almost nothing in recall and buys real protection against the private test
    set placing one more negative just above it.
    """
    thr, m, tpr, _ = sweep_thresholds(y_true, prob, alpha)
    ok = tpr >= min_tpr
    if not np.any(ok):
        return 0.5, -np.inf
    best_m = float(m[ok].max())
    # `thr` is descending, so the first acceptable index is the largest
    # threshold that stays within tolerance of the best M.
    good = ok & (m >= best_m - tolerance)
    return float(thr[np.argmax(good)]), best_m


def theoretical_threshold(y_true):
    """Threshold that maximises expected M for calibrated probabilities.

    Including a window with P(AF)=p changes M by p/P - alpha*(1-p)/N, which is
    positive exactly when p > alpha*P/(N + alpha*P). Reported as a sanity
    anchor for the empirically chosen value.
    """
    p = int(np.count_nonzero(y_true == 1))
    n = int(np.count_nonzero(y_true == 0))
    if p == 0 or n == 0:
        return np.nan
    return ALPHA * p / (n + ALPHA * p)


# --------------------------------------------------------------------------
# Feature map: fit on train, then applied unchanged everywhere
# --------------------------------------------------------------------------
def fit_feature_map(X_raw, y, groups):
    """Returns the fitted phi. `X_raw` is the concatenation of the given
    columns and (when available) the raw-signal columns."""
    state = {"prep": fit_preprocessor(X_raw)}
    X = apply_preprocessor(X_raw, state["prep"])
    ranking = rank_features(X, y, groups)
    order = np.argsort(-ranking)
    keep = np.sort(order[:min(SELECT_DIMS, X.shape[1])])
    state["ranking"] = ranking

    # Spline-expand the strongest columns, keeping the rest linear. The model
    # stays linear in phi(x) - it just gets a richer basis.
    top = np.sort(order[:min(GAM_TOP_K, X.shape[1])])
    spline = SplineTransformer(n_knots=GAM_N_KNOTS, degree=3,
                               include_bias=False)
    spline.fit(X[:, top])
    width = spline.transform(X[:1, top]).shape[1]
    if keep.size + width > MAX_FEATURE_DIM:
        keep = np.sort(order[:max(0, MAX_FEATURE_DIM - width)])
    state["keep"] = keep
    state["spline"] = spline
    state["spline_cols"] = top
    return state


def apply_feature_map(X_raw, state):
    X = apply_preprocessor(X_raw, state["prep"])
    phi = np.hstack([X[:, state["keep"]],
                     state["spline"].transform(X[:, state["spline_cols"]])])
    return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------
# Model fitting
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


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_matrix(df, feature_cols, dataset_dir, raw_names, use_raw):
    X = df[feature_cols].to_numpy(dtype=np.float64)
    if not use_raw:
        return X
    raw = extract_raw_features(df, dataset_dir, raw_names)
    return np.hstack([X, raw])


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

    use_raw = (Path(dataset_dir) / "raw_signals").is_dir()
    if not use_raw:
        print("no raw_signals/ - this version degrades to given-columns-only",
              file=sys.stderr)
    raw_names = _raw_feature_names() if use_raw else []

    X_train = build_matrix(train_df, feature_cols, dataset_dir, raw_names, use_raw)
    X_test = build_matrix(test_df, feature_cols, dataset_dir, raw_names, use_raw)

    X_val, y_val = None, None
    if val_df is not None and "label" in val_df.columns:
        X_val = build_matrix(val_df, feature_cols, dataset_dir, raw_names, use_raw)
        y_val = val_df["label"].to_numpy(dtype=np.int64)

    state = fit_feature_map(X_train, y_train, groups)
    phi_train = apply_feature_map(X_train, state)
    if phi_train.shape[1] >= 500:
        raise RuntimeError(f"phi has {phi_train.shape[1]} dims, spec needs < 500")

    all_names = feature_cols + raw_names
    top = [all_names[j] for j in np.argsort(-state["ranking"])[:5]]
    c, class_weight, cv_m = tune_hyperparameters(phi_train, y_train, groups)
    n_linear = state["keep"].size
    print(f"[v4 gam] d={phi_train.shape[1]} ({n_linear} linear + "
          f"{phi_train.shape[1] - n_linear} spline, from {X_train.shape[1]} "
          f"columns) C={c} class_weight={class_weight} "
          f"grouped-CV M={cv_m:.4f}", file=sys.stderr)
    print(f"top features: {', '.join(top)}", file=sys.stderr)

    model = make_model(c, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(phi_train, y_train)

    # Threshold from pooled patient-grouped OOF predictions on train plus the
    # final model's predictions on the untouched val split.
    pooled_prob = grouped_oof_probs(phi_train, y_train, groups, c, class_weight)
    pooled_y = y_train
    if X_val is not None:
        prob_val = model.predict_proba(apply_feature_map(X_val, state))[:, 1]
        pooled_prob = np.concatenate([pooled_prob, prob_val])
        pooled_y = np.concatenate([pooled_y, y_val])

    threshold, pooled_m = select_threshold(pooled_y, pooled_prob)
    anchor = theoretical_threshold(pooled_y)
    _, tp, fp, tpr, fpr = m_metric(pooled_y, pooled_prob, threshold)
    print(f"threshold={threshold:.4f} (calibrated-optimal anchor "
          f"{anchor:.4f}) pooled M={pooled_m:.4f} TP={tp} FP={fp} "
          f"TPR={tpr:.4f} FPR={fpr:.4f}", file=sys.stderr)
    if tpr < MIN_TPR_ELIGIBLE:
        print("WARNING: pooled TPR below the 0.1 eligibility floor",
              file=sys.stderr)

    with open(model_path, "wb") as f:
        pickle.dump({"weights": model.coef_[0].astype(np.float64),
                     "bias": float(model.intercept_[0]),
                     "threshold": float(threshold),
                     "method": "logreg[v4-gam]"}, f)

    phi_test = apply_feature_map(X_test, state)
    out = pd.DataFrame(phi_test,
                       columns=[f"feat_{i}" for i in range(phi_test.shape[1])])
    out.insert(0, "release_id", test_df["release_id"].to_numpy())
    out.to_csv(final_features_path, index=False)
    print(f"wrote {model_path} and {final_features_path} "
          f"({len(out)} rows, {phi_test.shape[1]} features)", file=sys.stderr)


if __name__ == "__main__":
    main()
