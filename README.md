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
directory. Inference needs no training archive, official kit, external service,
network, GPU, or generated cache in the repository.

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

Part A owns `engine/engine.py`, `model/`, `train/`, `weights/`, `assets/`, and
`tests/eval_skill.py`. Its training archive must remain outside Git. Training
expects the official dataset schema, including `data`, `static`, the named
channel coordinates, and consecutive dates.

```powershell
python -m train.inspect C:\external-data\india_climate_train.nc
python -m train.train C:\external-data\india_climate_train.nc --output C:\external-models\heldout\airavat.pt --split train --device cpu --threads 4
python -m tests.eval_skill C:\external-data\india_climate_train.nc --weights C:\external-models\heldout
python -m train.train C:\external-data\india_climate_train.nc --output C:\external-models\airavat.pt --split all --device cpu --threads 4
```

The evaluation weights directory must contain the held-out checkpoint named
`airavat.pt`; the final all-years checkpoint is not a held-out skill test.
Use `python -m train.train --help` for curriculum, width, resume, and batch options.
The organizer sandbox supplies the packages in root `requirements.txt`.
**Outstanding Part A packaging issue:** `train/requirements.txt` additionally
lists `netCDF4`, which is not on the supplied sandbox list and is not vendored.
NetCDF3 input works through the supplied SciPy backend; reading
NetCDF4 input requires Part A to resolve that backend dependency before claiming
training reproducibility in the supplied sandbox. See the integration report.

## Ownership and integration

Work branches are `ml` and `app`; shared releases use `main`. Follow `CLAUDE.md`:
rebase on shared `main` at handoff, preserve ownership, and never force-push.
Do not push an integration with unresolved Part A verification failures.
