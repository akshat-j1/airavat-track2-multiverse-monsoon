# Local training and evaluation

Install the local training dependencies with `python -m pip install -r requirements-train.txt` from the repository root. The official training archive uses NetCDF-4. Xarray uses the compiled `netCDF4` backend to read it during training and data inspection. The CPU Engine loads its model and static fields from `weights/airavat.pt`. The organizer's offline serving environment needs the runtime dependencies only.

```sh
python -m train.inspect /path/to/india_climate_train.nc
python -m train.eval_baselines /path/to/india_climate_train.nc --val-years 2
python -m train.train /path/to/india_climate_train.nc --split train --val-years 2 --output /tmp/airavat-validation/airavat.pt
python -m tests.eval_skill /path/to/india_climate_train.nc --val-years 2 --weights /tmp/airavat-validation
python -m train.train /path/to/india_climate_train.nc --split all --output weights/airavat.pt
```

`--val-years 2` reserves the last two calendar years represented in the dataset. This includes a partial final year if one is present. Every forecast window stays within its partition. `--split all` fits on the entire time axis for deployment. To run the optional shock diagnostics in `tests.eval_skill`, install the official kit wheel and supply its `stats.json`.

Use `--clim-years START END` to set the inclusive calendar-year range for
climatology and anomaly scale. The network still trains on every eligible
training year. With a validation split, both fits exclude validation and
buffer days. With multiple years in the window pool, `--sample-weight-recent`
keeps the epoch size fixed and samples with replacement. Per-window weights
rise linearly from 1 in the earliest year to 2 in the latest, using the year of
the last input day.

Pass `--folds START END` in place of `--val-years` and use a separate
checkpoint path for each run. The three audit folds are:

```sh
python -m train.train "$DATA_NC" --folds 2008 2009 --output /tmp/fold-a/airavat.pt
python -m train.train "$DATA_NC" --folds 2014 2015 --output /tmp/fold-b/airavat.pt
python -m train.train "$DATA_NC" --folds 2020 2021 --output /tmp/fold-c/airavat.pt
```

Each fold holds out the full interval and removes 14 days of training data on
either side where available. `--max-epochs` caps the supplied curriculum.
Validation uses a fixed rollout horizon so epoch losses can be compared. With
a holdout, the trainer saves the best checkpoint and stops after three epochs
without improvement. With `--split all`, it saves the latest epoch and records
null validation loss. The adjacent `airavat.loss.json` records the losses and
the selected epoch. Use the same `--seed`, backend, thread count, and software
versions for reproducible runs. MPS reports warnings for operations where
PyTorch cannot guarantee determinism.

### Screen rollout and rain-loss settings

Use `--curriculum` to reach 8 or 10 rollout steps. With
`--scheduled-sampling`, unpaired training windows start with a 50% chance
of receiving the observed state between steps. That chance falls linearly
to zero by the final epoch. Both members of each paired-shock window always
use their model predictions. The paired-shock and far-field penalties use
the existing objective.

`--event-rain-weight` enables the existing event-weighted rain loss.
`--rain-csi-weight` adds a differentiable overlap loss at the 10 mm/day
threshold alongside BCE. `--rain-energy-weight` matches log spatial
gradient energy in physical rain units. All three default to zero.

For tuning, use the same `--validation-steps` and
`--exclude-proxy-windows` across candidates. This removes validation
windows that share any inputs or targets with the five recorded proxies
and uses the base training objective as the comparison loss. Validation
always uses model predictions without augmentation. Run the official
benchmark with the frozen candidates after selection.

### Ensemble and shrinkage checkpoints

For an ensemble, put the first member at `airavat.pt` and the remaining
members at `airavat-member-*.pt` in the same directory. Members must have the
same climatology, anomaly scale, and static fields. The engine runs each
member's rollout independently and averages the anomalies before decoding.
Pass a file to the benchmark to evaluate one member or a directory to
evaluate the bundle.

```sh
python -m tests.benchmark --weights /tmp/e4 --label E4
python -m train.fit_shrinkage "$DATA_NC" --weights /tmp/e4 --output /tmp/e5
python -m tests.benchmark --weights /tmp/e5 --label E5
```

The fitter requires checkpoint metadata showing that training and the
climatology fit ended before the holdout. It excludes windows that overlap
any of the five proxies and fits a constrained least-squares multiplier in
anomaly space for each of the 14 leads and six channels. Only the primary
checkpoint in the new bundle stores the multiplier array. The rain
multiplier stays at 1 for every lead, preserving the unshrunk ensemble's
rain peaks. The benchmark reports model-to-observed peak rain ratios for
the cyclone and heavy-rain proxies. Larger leads use the array's own fitted
row.

## Not used for the submission

The shipped model was trained only on the organizer's supplied 2005–2021
archive. The earlier WeatherBench2 rows and the commands below belong to an
optional extension experiment, not part of submission. The supplied archive
remains the benchmark input. Its upstream source attribution is recorded in
`validation.json`.

```sh
python -m pip install -r requirements-train.txt
python -m pip install 'zarr>=3' gcsfs
ORGANIZER_NC=/path/to/india_climate_train.nc
EXTENDED_NC=data/r1/india_climate_1979_2021.nc
BUILD_DIR="$(mktemp -d)"
mkdir -p "$(dirname "$EXTENDED_NC")"

python -m monsoon_r1.build_dataset train \
  --start 1979-01-01 --end 2004-12-31 --out "$BUILD_DIR/history.npz"
python - "$ORGANIZER_NC" "$BUILD_DIR/organizer.npz" <<'PY'
from pathlib import Path
import sys
import numpy as np
from model.unet import LAT, LON
from train.common import load_weather

source = load_weather(Path(sys.argv[1]))
dates = np.char.replace(source.dates.astype(str), "-", "").astype(int)
np.savez_compressed(sys.argv[2], data=source.physical, dates=dates, lat=LAT, lon=LON)
PY
python -m monsoon_r1.build_dataset merge \
  --parts "$BUILD_DIR/history.npz" "$BUILD_DIR/organizer.npz" --out "$BUILD_DIR/extended.npz"
python - "$ORGANIZER_NC" "$BUILD_DIR/extended.npz" "$EXTENDED_NC" <<'PY'
from pathlib import Path
import sys
import numpy as np
from monsoon_r1.build_dataset import export_nc
from train.common import load_weather

source = load_weather(Path(sys.argv[1]))
output = Path(sys.argv[3])
export_nc(Path(sys.argv[2]), output, static=source.static)
extended = load_weather(output)
np.testing.assert_array_equal(extended.static, source.static)
np.testing.assert_array_equal(extended.dates[-len(source.dates):], source.dates)
np.testing.assert_array_equal(extended.physical[-len(source.dates):], source.physical)
print(output, extended.dates[0], extended.dates[-1], extended.physical.shape)
PY
```

## Reproduce the shipped checkpoint

The retained 15-epoch checkpoint was trained on the organizer archive with SHA-256
`f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b`.
The original run used Apple MPS, Python 3.14.7, PyTorch 2.15.0, NumPy 2.5.5,
xarray 2026.9.0, and netCDF4 1.7.5.1.post1. Expect CPU or other backends to
produce different weights and rerun the validation metrics on the result.

Stage 1 uses the original Gaussian shock augmentation at commit `0b87d20`.
Stage 2 uses compact raised-cosine shocks and the far-field penalty in the
current trainer. Run from the repository root with an activated training venv.

```sh
DATANC=/absolute/path/to/india_climate_train.nc
PYTHON_BIN="$(command -v python)"
RUN=$(mktemp -d)
mkdir -p "$RUN/source"
git archive 0b87d20 engine model train | tar -x -C "$RUN/source"
(
  cd "$RUN/source"
  PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON_BIN" -m train.train "$DATANC" \
    --split all --device mps --seed 42 --width 16 --batch-size 48 \
    --curriculum 1,1,1,2,2,3,4,4,6,8,8 --learning-rate 0.002 \
    --output "$RUN/base/airavat.pt"
)
PYTORCH_ENABLE_MPS_FALLBACK=1 "$PYTHON_BIN" -m train.train "$DATANC" \
  --split all --device mps --seed 42 --width 16 --batch-size 48 \
  --curriculum 1,2,3,4 --learning-rate 0.00035 --far-field-weight 18 \
  --event-rain-weight 0 --resume "$RUN/base/airavat.pt" \
  --output "$RUN/final/airavat.pt"
```

For held-out evaluation, use `--split train` in both stages and
`--val-years 2` in stage 2. Pass the event archive's externally chosen first
verification days as `--event LABEL=DATE` and `--shock-target DATE` arguments
to `python -m tests.eval_skill`. The dated cases and per-event results are in
[`tests/fixtures/proxy_events.json`](../tests/fixtures/proxy_events.json), and
`validation.json` retains the aggregate metrics. The deployment checkpoint's
shock probes use archive observations included in its training period.

The experimental `--event-rain-weight` loss adds intensity, local peak and
rainfall gradient terms. The tested settings increased some storm peaks and
reduced the aggregate event score, so the integrated candidate preserves the
shipped weights.
