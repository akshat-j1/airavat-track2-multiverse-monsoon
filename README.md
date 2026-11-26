# Multiverse of Monsoon / Airavat

A CPU weather model with a static counterfactual dashboard. The API uses physical
units and the channel order `t2m, tp, u10, v10, msl, tcwv`. Read [CONTRACT.md](CONTRACT.md)
for the interface and [docs/INTEGRATION_REPORT.md](docs/INTEGRATION_REPORT.md) for
the integration gate, measured results, assumptions, and outstanding ML issues.

## Run on Linux

From a fresh checkout, install the CPU runtime into a virtual environment and start the server.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 python serve.py --port 8000
```

Open the [local dashboard](http://127.0.0.1:8000/dashboard/) once `/health` reports
`airavat-anomaly-unet`. The checkpoint is stored directly in Git at `weights/airavat.pt`.
An absent, invalid, or pointer-only checkpoint fails startup. A running engine that
fails a forecast returns HTTP 500. Set `MONSOON_DEV_STUB=1` only when developing
without a usable checkpoint. In that mode the server identifies a startup fallback as a stub.

Install the official kit into a local test environment, then run the checks in another terminal.

```sh
. .venv/bin/activate
python -m pip install /path/to/monsoon_r1_kit-1.0.0-py3-none-any.whl
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
python tests/check_contract.py --url http://127.0.0.1:8000 --require-real
python tests/check_shocks.py --url http://127.0.0.1:8000 --require-real
node tests/check_impact.js
```

Install the training dependencies locally and point the training tools at the external archive.

```sh
python -m pip install -r requirements-train.txt
DATA_NC=/path/to/india_climate_train.nc
python -m train.inspect "$DATA_NC"
python -m train.eval_baselines "$DATA_NC" --val-years 2
python -m train.train "$DATA_NC" --split train --val-years 2 --output /tmp/airavat-validation/airavat.pt
python -m tests.eval_skill "$DATA_NC" --val-years 2 --weights /tmp/airavat-validation
```

## Run on Windows PowerShell

Use the existing virtual environment, or create one with `python -m venv .venv`.

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONDONTWRITEBYTECODE = "1"
python serve.py --port 8000
```

Open `http://127.0.0.1:8000/dashboard/`. The server binds to `0.0.0.0` by default.
The trained model is required by default. `/health` must report `airavat-anomaly-unet`.
The checkpoint is `weights/airavat.pt`, loaded from the repository's weights
directory. Serving uses only the runtime dependencies in `requirements.txt`.
`requirements-train.txt` contains dependencies for training and dataset inspection
only, including xarray and netCDF4.

The frontend uses only local files in `dashboard/`; it has no external fonts,
scripts, CDN assets, or build step. It calls the same origin under `/dashboard/`,
otherwise `http://localhost:8000`. Synthetic browser inputs are demonstrations,
not observations. Uploaded windows must be JSON arrays shaped `[3,6,32,36]` or
objects with `init_states` and optional `doy_sincos`. Select m/day before uploading
if the file uses metres for precipitation. The import converts it to mm/day.

Use **Spin up start state** to forecast 14 days from the synthetic seed and reuse
the last three returned days as the next input. The input label identifies the
model-generated fields. Choose a shock, run both timelines, and move the slider
to compare their live forecasts. The day-of-year phase follows the API's 365-day
cycle, so day 366 has the same phase as day 1.

All land impact summaries are indicative scenario metrics, not agronomic
forecasts. They use returned rollouts and a static land mask. Prime/Branch cards
show unweighted means per land cell. Editable thresholds use the loaded forecasts.
Live precipitation is always in mm/day. **Run sensitivity ensemble** reuses Prime
and makes eight additional live forecasts with small local temperature or wind
perturbations. Charts show the min–max band and median of daily land means.
Exceedance probabilities average member frequencies across land cells for the
selected day. The ensemble illustrates sensitivity to these disturbances and
its frequencies are uncalibrated.

## Static grid

The dashboard ships only latitude/longitude and the checkpoint's two static
fields, not archived weather. Regenerate them after changing model weights:

```powershell
python tools/export_grid.py --checkpoint weights/airavat.pt
```

Alternatively, export an externally stored training NetCDF. Both separate static
variables and the official `static[static_channel,lat,lon]` layout are supported:

```powershell
python tools/export_grid.py --input C:\external-data\india_climate_train.nc
```

## Verify

Run these in a second activated PowerShell terminal while the trained server runs:

```powershell
python -m monsoon_r1.schema_validator http://localhost:8000
python tests/check_contract.py --url http://127.0.0.1:8000 --require-real
python tests/check_shocks.py --url http://127.0.0.1:8000 --require-real
node tests/check_impact.js
```

To check shocks on real observations, install the dataset dependencies and
export a local test window from the official archive:

```powershell
python tools/export_window.py --input C:\external-data\india_climate_train.nc --last-index 195 --output tests/_windows/real.npz
python tests/check_shocks.py --url http://127.0.0.1:8000 --require-real --window tests/_windows/real.npz
```

The first command requires the official `monsoon_r1_kit-1.0.0-py3-none-any.whl`
from the [organizers' v1.0.0 release](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0)
and its local client dependencies. These are testing tools, not server runtime
dependencies. The Python app checks use stdlib urllib and NumPy only; `--window`
accepts `.npy`, `.npz`, and JSON windows. Without it, checks use a synthetic
constant physical window on the contract grid, not a held-out skill evaluation.
Scratch test inputs, logs, and downloaded kit files belong in ignored
`tests/_windows/`. On this Windows machine `localhost` first attempts IPv6 and
adds about two seconds before falling back to IPv4; the explicit IPv4 URL avoids
that client overhead for latency measurements.

## Train and assess skill

Follow the training, dataset inspection, and held-out evaluation instructions in
[train/README.md](train/README.md), including the commands to build the extended
archive and reproduce the shipped checkpoint. Keep the training archive outside Git.

## Build the submission brief

With Contributor A's `train/brief_page1.md` present, use an authoring environment
with ReportLab to combine it with `brief/page2.md` and its dashboard capture.

```sh
python -m pip install reportlab
python brief/build.py
pdfinfo brief/brief.pdf
```

The PDF should contain exactly two pages.

## Ownership and integration

Work branches are `fix/ml` and `fix/app`. Shared releases use `main`. Follow
`CLAUDE.md` for file ownership and rebase work onto shared `main` at handoff.
