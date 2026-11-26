# CPU reproduction

## Proposed README sentence

> The recipe is seed-sensitive on CPU; the shipped checkpoint lies outside the CPU seed range on precipitation RMSE, heavy-rain CSI, and day-8 precipitation variance ratio.

The shipped checkpoint falls outside the two-seed CPU interval in **16 of 28** comparisons.
By metric, it falls outside on 6/12 RMSE, 7/12 CSI, and 3/4 variance ratios.
Day-8 RMSE lies inside the range on all four folds. The largest distances
outside the range were 0.00705 mm/day for RMSE, 0.00239 for CSI, and
0.00992 for variance ratio.
A two-seed comparison measures observed variation. Same-seed repeatability and
reproducibility across other machines were outside this experiment.

## Environment and training

| Component | Version |
| --- | --- |
| python | `3.14.7 (main, Aug 25 2026, 13:50:14) [Clang 22.1.3 ]` |
| torch | `2.15.0` |
| numpy | `2.5.5` |
| xarray | `2026.9.0` |
| netCDF4 | `1.7.5.1.post1` |
| platform | `macOS-26.5.1-arm64-arm-64bit-Mach-O` |
| main commit | `b44d6b58ba0afa19501bf5eb2cc5f21d3fdaf635` |
| Gaussian augmentation stage | `0b87d2034e0bcbd2136c2ac7029addc48b5dbbe9` |
| Organizer training archive SHA-256 | `f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b` |

Both runs used the full organizer archive, width 16, batch 48, four CPU
threads, and the documented two-stage recipe. Each fine-tune started
from the same seed's completed Gaussian-augmentation checkpoint.
The seed runs overlapped on this machine. Stage wall time includes
Python startup, data loading, training, and checkpoint writes.
A supervisor enforced a three-hour limit per stage.

| CPU seed | Base wall (s) | Fine-tune wall (s) | Total per seed (s) | Final checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| 42 | 871.04 | 242.79 | 1113.82 | `76f2b3930bac1070b653f1361abb8dd889fc3bd31fecb409e4d914fb3dcd4c53` |
| 43 | 871.04 | 242.79 | 1113.82 | `fa7e9c0727af1218f7716ef2117dcce7b8335bdb6b027c3352640101ca761f08` |

The concurrent training runs took 1113.82 seconds end to end.

```sh
REPO="$(pwd)"
DATA_NC="$REPO/tests/_windows/india_climate_train.nc"
PYTHON_BIN="$REPO/.venv/bin/python"
RUN="$REPO/tests/_windows/cpu-repro"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$RUN/source"
git archive 0b87d20 engine model train | tar -x -C "$RUN/source"
for seed in 42 43; do
  (
    cd "$RUN/source" || exit
    "$PYTHON_BIN" -u -m train.train "$DATA_NC" --split all --device cpu --threads 4 \
      --seed "$seed" --width 16 --batch-size 48 --curriculum 1,1,1,2,2,3,4,4,6,8,8 \
      --learning-rate 0.002 --output "$RUN/seed$seed/base/airavat.pt"
  ) || exit
  "$PYTHON_BIN" -u -m train.train "$DATA_NC" --split all --device cpu --threads 4 \
    --seed "$seed" --width 16 --batch-size 48 --curriculum 1,2,3,4 --learning-rate 0.00035 \
    --far-field-weight 18 --event-rain-weight 0 --resume "$RUN/seed$seed/base/airavat.pt" \
    --output "$RUN/seed$seed/final/airavat.pt" || exit
done
```

## Fold comparisons

These are **in-sample** model comparisons. The two CPU checkpoints and the
shipped Apple MPS checkpoint all used the full training archive, so all three
saw the scored years. Every checkpoint was evaluated through the CPU engine
on identical daily windows contained in each protocol fold using weather
through 2019. The existing fold-trained checkpoints were produced on MPS
with a different training split and were left out of the CPU seed reference.

All triplets are day 1 / 4 / 8. The existing evaluator supplies latitude-weighted
precipitation RMSE in mm/day, CSI for rain strictly above 10 mm/day, and the
mean of per-window day-8 spatial variance ratios. The CPU seed spread is
the absolute difference between the two seeds in each cell.

| Fold | Use | Windows | Checkpoint training device | Rain RMSE (mm/day) | Heavy-rain CSI | Day-8 rain variance ratio |
| --- | --- | ---: | --- | --- | --- | ---: |
| 2008–2009 | selection | 721 | CPU seed 42 | 4.2666 / 5.8846 / 6.4526 | 0.57057 / 0.39867 / 0.35307 | 0.7375 |
| 2008–2009 | selection | 721 | CPU seed 43 | 4.2506 / 5.8422 / 6.3457 | 0.57084 / 0.39684 / 0.36043 | 0.8125 |
| 2008–2009 | selection | 721 | Shipped Apple MPS | 4.2737 / 5.8843 / 6.4456 | 0.57029 / 0.39901 / 0.35330 | 0.7459 |
| 2008–2009 | selection | — | **Absolute CPU seed spread** | 0.0160 / 0.0425 / 0.1069 | 0.00027 / 0.00183 / 0.00737 | 0.0749 |
| 2014–2015 | selection | 720 | CPU seed 42 | 4.3101 / 5.9065 / 6.5441 | 0.53223 / 0.38198 / 0.33551 | 0.7371 |
| 2014–2015 | selection | 720 | CPU seed 43 | 4.3093 / 5.8670 / 6.3403 | 0.53340 / 0.37832 / 0.34913 | 0.7527 |
| 2014–2015 | selection | 720 | Shipped Apple MPS | 4.3085 / 5.9131 / 6.5381 | 0.53265 / 0.38293 / 0.33707 | 0.7272 |
| 2014–2015 | selection | — | **Absolute CPU seed spread** | 0.0008 / 0.0395 / 0.2038 | 0.00117 / 0.00365 / 0.01362 | 0.0156 |
| 2011–2012 | acceptance | 721 | CPU seed 42 | 4.3039 / 5.7657 / 6.3099 | 0.56040 / 0.40386 / 0.35666 | 0.7808 |
| 2011–2012 | acceptance | 721 | CPU seed 43 | 4.2953 / 5.7726 / 6.2795 | 0.56029 / 0.40240 / 0.36163 | 0.9344 |
| 2011–2012 | acceptance | 721 | Shipped Apple MPS | 4.3044 / 5.7655 / 6.3096 | 0.56105 / 0.40385 / 0.35638 | 0.7740 |
| 2011–2012 | acceptance | — | **Absolute CPU seed spread** | 0.0087 / 0.0069 / 0.0304 | 0.00010 / 0.00146 / 0.00497 | 0.1536 |
| 2017–2018 | acceptance | 720 | CPU seed 42 | 4.3804 / 5.9222 / 6.4453 | 0.54553 / 0.39033 / 0.35563 | 0.7771 |
| 2017–2018 | acceptance | 720 | CPU seed 43 | 4.3315 / 5.8614 / 6.3053 | 0.54492 / 0.38907 / 0.35321 | 0.8404 |
| 2017–2018 | acceptance | 720 | Shipped Apple MPS | 4.3839 / 5.9079 / 6.4210 | 0.54523 / 0.39272 / 0.35801 | 0.7763 |
| 2017–2018 | acceptance | — | **Absolute CPU seed spread** | 0.0489 / 0.0608 / 0.1400 | 0.00062 / 0.00126 / 0.00243 | 0.0632 |

The table below locates the shipped metric relative to the observed CPU
seed range. The distance above or below the interval is in the same unit
as the metric. Range decisions use unrounded values.

| Fold | Rain RMSE, days 1 / 4 / 8 | Heavy-rain CSI, days 1 / 4 / 8 | Day-8 rain variance ratio |
| --- | --- | --- | --- |
| 2008–2009 | above 0.00705 / inside / inside | below 0.00028 / above 0.00035 / inside | inside |
| 2014–2015 | below 0.00080 / above 0.00656 / inside | inside / above 0.00096 / inside | below 0.00992 |
| 2011–2012 | above 0.00042 / below 0.00018 / inside | above 0.00065 / inside / below 0.00027 | below 0.00671 |
| 2017–2018 | above 0.00350 / inside / inside | inside / above 0.00239 / above 0.00237 | below 0.00081 |

## Runtime checks

Ordinary CPU inference loaded both 15-epoch checkpoints and produced valid
10-step rollouts of shape `(10, 6, 32, 36)` with finite values and nonnegative
rain. For the HTTP checks, an isolated copy of the unmodified server loaded
each scratch checkpoint in turn. The checks used synthetic inputs.

| Checkpoint | Direct 10-step call | Schema validator | Contract | Shocks | Impact |
| --- | --- | --- | --- | --- | --- |
| CPU seed 42 | PASS (8.29 ms) | PASS | PASS | PASS | PASS |
| CPU seed 43 | PASS (5.73 ms) | PASS | PASS | PASS | PASS |

The shipped checkpoint SHA-256 is `30143677fb684a5e3daa92efcc7042de174d63c2ad07d7d509da9b258899796d`.
