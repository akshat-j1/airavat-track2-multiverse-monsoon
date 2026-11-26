# Multiverse of Monsoon / Airavat

A CPU weather model with a static counterfactual dashboard. The API uses physical
units and the channel order `t2m, tp, u10, v10, msl, tcwv`. Read [CONTRACT.md](CONTRACT.md)
for the interface and [docs/INTEGRATION_REPORT.md](docs/INTEGRATION_REPORT.md) for
the integration gate, measured results, assumptions, and outstanding ML issues.

## Run on Windows PowerShell

Use the existing virtual environment, or create one with `python -m venv .venv`.

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
git lfs install
git lfs pull
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:MONSOON_STRICT = "1"
python serve.py --port 8000
```

Open `http://127.0.0.1:8000/dashboard/`. The server binds to `0.0.0.0` by default.
Strict mode prevents startup/request failures from silently falling back to the
development stub. `/health` must report `airavat-anomaly-unet`, without `stub`.
The checkpoint is `weights/airavat.pt`, loaded from the repository's weights
directory. Serving uses only the runtime dependencies in `requirements.txt`.
`requirements-train.txt` contains dependencies for training and dataset inspection
only, including xarray and netCDF4.

The frontend uses only local files in `dashboard/`; it has no external fonts,
scripts, CDN assets, or build step. It calls the same origin under `/dashboard/`,
otherwise `http://localhost:8000`. Synthetic browser inputs are demonstrations,
not observations. Uploaded windows must be JSON arrays shaped `[3,6,32,36]` or
objects with `init_states` and optional `doy_sincos`.

All land impact summaries are indicative scenario metrics, not agronomic
forecasts. They use returned rollouts and a static land mask. Prime/Branch cards
show unweighted means per land cell; sea cells are excluded. Editable thresholds
do not rerun the model. Contract rainfall is mm/day. The explicit units selector
supports inspecting other inputs, but must remain mm for this engine.

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

Run these in a second activated PowerShell terminal while the strict server runs:

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
[train/README.md](train/README.md). Keep the training archive outside Git.

## Ownership and integration

Work branches are `ml` and `app`; shared releases use `main`. Follow `CLAUDE.md`:
rebase on shared `main` at handoff, preserve ownership, and never force-push.
