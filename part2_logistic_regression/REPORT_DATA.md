# Report data

Generated 2026-09-06 18:22:39

## Part (a) — loss vs time

Plots: `parta_train_loss_vs_time.png`, `parta_val_loss_vs_time.png` (log-scaled time axis).

| method | epochs | train secs | final train loss | final val loss | min val loss | at epoch | at secs |
|---|---|---|---|---|---|---|---|
| full_batch | 500 | 0.70 | 0.4793 | 0.5339 | 0.5339 | 500 | 0.70 |
| mini_batch | 200 | 0.60 | 0.4652 | 0.5301 | 0.5281 | 72 | 0.22 |
| sgd | 30 | 2.09 | 0.4857 | 0.5350 | 0.5350 | 30 | 2.09 |
| adagrad | 200 | 0.71 | 0.4626 | 0.5286 | 0.5265 | 193 | 0.68 |

- Cheapest in wall-clock training time: **mini_batch**.
- Lowest final training loss: **adagrad**.
- Overfitting onset per method = the epoch of minimum validation loss in the table above; validation loss rising after that point while training loss keeps falling is the signature.
  - `full_batch`: no overfitting within the prescribed epochs.
  - `mini_batch`: val loss bottoms at epoch 72 then rises.
  - `sgd`: no overfitting within the prescribed epochs.
  - `adagrad`: val loss bottoms at epoch 193 then rises.

## Part (b) — class imbalance

Training class counts: N=2723, A(AF)=366, O=1221. Metrics on `part_ab_val.csv`.

| method | Macro-AP | balanced acc | plain acc | recall N | recall A (AF) | recall O | precision A |
|---|---|---|---|---|---|---|---|
| baseline | 0.7765 | 0.7186 | 0.7998 | 0.9366 | **0.6923** | 0.5267 | 0.8060 |
| classweight | 0.7748 | 0.7555 | 0.7738 | 0.8305 | **0.7949** | 0.6412 | 0.6200 |
| classweight2 | 0.7777 | 0.7322 | 0.7965 | 0.9161 | **0.7308** | 0.5496 | 0.7600 |
| focal | 0.7659 | 0.7403 | 0.7846 | 0.8767 | **0.7564** | 0.5878 | 0.6782 |

- `classweight` vs baseline: AF recall 0.6923 -> 0.7949, balanced acc 0.7186 -> 0.7555, plain acc 0.7998 -> 0.7738, Macro-AP 0.7765 -> 0.7748.
- `classweight2` vs baseline: AF recall 0.6923 -> 0.7308, balanced acc 0.7186 -> 0.7322, plain acc 0.7998 -> 0.7965, Macro-AP 0.7765 -> 0.7777.
- `focal` vs baseline: AF recall 0.6923 -> 0.7564, balanced acc 0.7186 -> 0.7403, plain acc 0.7998 -> 0.7846, Macro-AP 0.7765 -> 0.7659.
