# Airavat experiments

## Step 1 — Verify the baseline

Added this log. Used the existing organizer scoring wrappers in
`tests/eval_skill.py`. The proxy and shock dates came from
`train/validation.json`. The training partition came from the archive time axis.
The local archive ended before the project cutoff. This step evaluated existing
checkpoints and performed no training or tuning.

The recorded 28.69/45 and −2.0% came from an earlier train-only fit. The shipped
file was fit on the full public archive, including the five proxy windows, so
its replay is in-sample. The recreated CPU train-only checkpoint provides a
paired reference when testing a future change to that model. It is an ignored
scratch artifact from the earlier calibration experiment.

### Official forecast and shock scoring

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
@'
import hashlib
import json
from pathlib import Path

import torch

from engine.engine import Engine
from tests.eval_skill import evaluate_events, evaluate_shocks
from train.common import load_weather, validation_start

path = Path("weights/airavat.pt")
audit = json.loads(Path("train/validation.json").read_text(encoding="utf-8"))
release = audit["release_candidate_audit"]
events = release["event_forecast"]
weather = load_weather(Path("tests/_windows/india_climate_train.nc"))
cut = validation_start(weather.dates, 2)
cases = list(dict.fromkeys(f'{e["event"]}={e["first_target"]}' for e in events["events"]))
for name, weights in (("shipped", path.parent), ("recreated_train_only", Path("tests/_windows/calibration/reference-final"))):
    checkpoint = weights / "airavat.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    engine = Engine(weights)
    score = evaluate_events(engine, weather, cut, cases)
    shocks = evaluate_shocks(engine, weather, Path("assets/stats.json"), 0, cut, release["shock_first_target_dates"])
    checks = [shock for case in shocks for shock in case["shocks"].values()]
    winter = [case["heavy_cells_model"] for case in score["events"] if case["event"] == "winter_disturbance"]
    print(json.dumps({"checkpoint": name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          "format_version": metadata["format_version"], "split": metadata["split"], "fit_period": metadata["fit_period"],
          "archive_last_date": str(weather.dates[-1]), "forecast_score": score["points"],
          "error_points": score["rmse_pts"], "rain_points": score["csi_pts"],
          "day8_skill": score["mean_skill_vs_best_by_lead"][-1], "day8_variance_ratio": score["day8_variance_ratio"],
          "winter_predictions": winter, "winter_observed": [case["heavy_cells_observed"] for case in score["events"] if case["event"] == "winter_disturbance"],
          "propagation": sum(s["propagates"] for s in checks), "locality": sum(s["localized"] for s in checks),
          "stability": sum(s["stable"] for s in checks), "exact_zero": sum(s["zero_shock_exact"] for s in shocks)}))
print(json.dumps({"recorded_scope": release["forecast_scope"], "recorded_forecast_score": events["points"],
                  "recorded_day8_skill": events["mean_skill_vs_best_by_lead"][-1]}))
'@ | & .venv/Scripts/python.exe -B -
```

Output, formatted from the three JSON lines. Variance ratios follow the
contract channel order. Winter counts are predictions and observations for
days 1, 4, and 8.

| Output field | Shipped checkpoint | Recreated train-only checkpoint |
|---|---:|---:|
| Split, format version | all, 1 | train, 1 |
| Fit period | 2005-01-01 through 2021-12-31 | 2005-01-01 through 2019-12-31 |
| Archive last date | 2021-12-31 | 2021-12-31 |
| Forecast points /45 | 37.630563 | 31.286780 |
| RMSE points /30 | 22.630563 | 19.607013 |
| Rain points /15 | 15.000000 | 11.679767 |
| Day-8 best-reference skill | −2.012472% | −1.364859% |
| Day-8 variance ratios | 0.875964 / 0.603719 / 1.022665 / 0.864425 / 0.896895 / 0.872377 | 0.872475 / 0.557414 / 1.067280 / 0.882122 / 0.909492 / 0.882297 |
| Winter cells, predictions | 54 / 0 / 0 | 32 / 0 / 0 |
| Winter cells, observations | 62 / 41 / 13 | 62 / 41 / 13 |
| Propagation /20 | 19 | 19 |
| Locality /20 | 20 | 20 |
| Stability /20 | 20 | 20 |
| Exact zero-shock /10 | 10 | 10 |

The shipped SHA-256 was
`30143677fb684a5e3daa92efcc7042de174d63c2ad07d7d509da9b258899796d`.
The recreated reference SHA-256 was
`1e78821512364935cbd19412a20b639b3dbf5f42d59eef475eb1aa31e4ea0`.
The same command read the archived train-only score of `28.685252775391433`
and day-8 skill of `-0.02005392629766817` from `train/validation.json`.

### Cold server on four physical cores

Timed the first 10-step HTTP request after readiness, including decode of the
response. The server performed its built-in startup warmup. The host process
was pinned to one logical CPU on each of four physical cores. The first
launcher attempt reported a `ctypes.ArgumentError` when converting the
pseudo-handle. The command below uses a typed handle.

```powershell
@'
import json, os, socket, subprocess, time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from monsoon_r1 import scoring

server_code = """
import ctypes as c, runpy, sys
from ctypes import wintypes as w
class Cpu(c.Structure):
    _fields_ = [('mask', c.c_size_t), ('relationship', c.c_int),
                ('details', c.c_uint64 * 2)]
win = c.WinDLL('kernel32', use_last_error=True)
process = w.HANDLE(-1)
allowed, system, size = c.c_size_t(), c.c_size_t(), w.DWORD()
assert win.GetProcessAffinityMask(process, c.byref(allowed), c.byref(system))
win.GetLogicalProcessorInformation(None, c.byref(size))
cpus = (Cpu * (size.value // c.sizeof(Cpu)))()
assert win.GetLogicalProcessorInformation(cpus, c.byref(size))
cores = sorted(cpu.mask & allowed.value for cpu in cpus if cpu.relationship == 0 and cpu.mask & allowed.value)
assert len(cores) >= 4
mask = sum(core & -core for core in cores[:4])
assert win.SetProcessAffinityMask(process, c.c_size_t(mask))
print(f'affinity={mask:#x} physical_cores=4', flush=True)
sys.argv[0] = 'serve.py'
runpy.run_path('serve.py', run_name='__main__')
"""
with np.load("tests/_windows/real.npz", allow_pickle=False) as fixture:
    payload = json.dumps({"init_states": fixture["init_states"].tolist(),
                          "doy_sincos": fixture["doy_sincos"].tolist(), "steps": 10}).encode()
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"
runtime = Path("tests/_windows/runtime/Scripts/python.exe").resolve()
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
env.pop("MONSOON_DEV_STUB", None)
start = time.perf_counter()
server = subprocess.Popen([str(runtime), "-B", "-c", server_code, "--port", str(port), "--host", "127.0.0.1"],
                          env=env, stdout=subprocess.PIPE)
try:
    for _ in range(150):
        if server.poll() is not None:
            raise RuntimeError("Server exited before /health")
        try:
            with urlopen(url + "/health", timeout=1) as handle:
                health = json.load(handle)
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise TimeoutError("Server did not start")
    startup = time.perf_counter() - start
    assert health == {"status": "ready", "model_name": "airavat-anomaly-unet"}
    req = Request(url + "/simulate", data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(req, timeout=15) as handle:
        result = json.load(handle)
    elapsed = time.perf_counter() - start
    print(server.stdout.readline().decode().strip())
    print(json.dumps({"startup_s": startup, "first_10_step_http_s": elapsed,
                      "inference_ms": result["inference_ms"], "health": health,
                      "official_validity_errors": scoring.validity(np.asarray(result["rollout"]), 10)}))
finally:
    if server.poll() is None:
        server.terminate()
        server.wait(timeout=10)
'@ | & .venv/Scripts/python.exe -B -
```

```text
[serve] ready: model_name=airavat-anomaly-unet
affinity=0x55 physical_cores=4
{"startup_s": 6.155032800044864, "first_10_step_http_s": 0.20369759993627667, "inference_ms": 34.62, "health": {"status": "ready", "model_name": "airavat-anomaly-unet"}, "official_validity_errors": []}
```

`git check-attr filter -- weights/airavat.pt` returned
`weights/airavat.pt: filter: unspecified`. No candidate weights were promoted.

## Step 2 — Preflight the extended archive

The supplied reference archive loaded successfully. The extended archive was
absent from its requested path. The format, static-field equality, seam
statistics, period statistics, and land warming trend for the extended archive
are pending. This step changed only the experiment log.

```powershell
@'
from pathlib import Path
import numpy as np
from model.unet import CHANNELS, RAIN
from train.common import load_weather

reference = load_weather(Path("tests/_windows/india_climate_train.nc"))
print("Reference load_weather PASS", reference.physical.shape, reference.static.shape,
      reference.dates[0], reference.dates[-1], "min_tp", reference.physical[:, RAIN].min())
for i, name in enumerate(CHANNELS):
    a = reference.physical[:,i]
    print(name, "mean", f"{a.mean(dtype=np.float64):.6f}", "std", f"{a.std(dtype=np.float64):.6f}")
target = Path("data/r1/india_climate_1979_2021.nc")
try:
    load_weather(target)
except FileNotFoundError:
    print(f"Extended load_weather BLOCKED (FileNotFoundError): {target}")
    raise SystemExit(2)
'@ | & .venv/Scripts/python.exe -B -
```

```text
Reference load_weather PASS (6209, 6, 32, 36) (2, 32, 36) 2005-01-01 2021-12-31 min_tp 0.0
t2m mean 292.356904 std 13.463642
tp mean 3.618069 std 7.553341
u10 mean 0.887652 std 2.898980
v10 mean 0.191332 std 2.426953
msl mean 101122.321925 std 619.019240
tcwv mean 30.033538 std 20.494364
Extended load_weather BLOCKED (FileNotFoundError): data\r1\india_climate_1979_2021.nc
```

## Step 3 — Add deterministic trainer controls

Added inclusive climatology years, optional recent-year sampling, buffered
validation folds, a curriculum epoch cap, and per-epoch JSON loss logs. The
trainer uses a fixed clean validation horizon for checkpoint selection and
stops after three consecutive non-improvements. Training window eligibility
comes from the dataset time axis and the selected split. The trainer accepts
one fold per invocation. The smoke runs used CPU and a scratch copy of rows
from the provided archive.

### Smoke command

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
@'
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import torch
import xarray as xr
from engine.engine import Engine
from monsoon_r1 import scoring
from train.common import FOLD_BUFFER_DAYS, day_phase, load_weather, time_block_split, window_ends
from train.train import training_order

archive = Path("tests/_windows/india_climate_train.nc")
root = Path("tests/_windows/trainer-smoke")
root.mkdir(exist_ok=True)
fixture = root / "fixture.nc"
with xr.open_dataset(archive) as source:
    days = source.time.values.astype("datetime64[D]")
    boundary = np.flatnonzero(days.astype("datetime64[Y]")[1:] != days.astype("datetime64[Y]")[:-1])[0] + 1
    source.isel(time=slice(boundary - 100, boundary + 100)).to_netcdf(fixture)
weather = load_weather(fixture)
assert weather.physical.shape == (200, 6, 32, 36)
first_year = str(weather.dates[0].astype("datetime64[Y]"))
last_year = str(weather.dates[-1].astype("datetime64[Y]"))
base = [sys.executable, "-B", "-m", "train.train", str(fixture),
        "--device", "cpu", "--width", "4", "--threads", "4", "--batch-size", "32"]
validation = ["--split", "train", "--val-years", "1"]
opts = ["--seed", "35", "--clim-years", first_year, first_year, "--sample-weight-recent"]

def run(name, flags):
    path = root / name / "airavat.pt"
    command = [*base, "--output", str(path), *flags]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)
    history = json.loads(path.with_suffix(".loss.json").read_text())
    package = torch.load(path, map_location="cpu", weights_only=True)
    assert history["checkpoint_epoch"] == package["epochs"]
    model = Engine(path.parent)
    result = model.simulate(weather.physical[:3], day_phase(weather.day_of_year[2]), 2)
    assert not scoring.validity(result, 2)
    return history, package, path

old, _, old_path = run("legacy", [*validation, "--curriculum", "1"])
a, package, first_path = run("a", [*validation, *opts, "--curriculum", "1,2,2", "--max-epochs", "2"])
b, _, second_path = run("b", [*validation, *opts, "--curriculum", "1,2,2", "--max-epochs", "2"])
assert a == b and len(a["epochs"]) == 2
assert package["clim_years"] == [int(first_year)] * 2
assert torch.equal(torch.load(first_path, weights_only=True)["model"]["input.weight"],
                   torch.load(second_path, weights_only=True)["model"]["input.weight"])
assert old["epochs"][0]["val_loss"] is not None

all_years, all_pkg, all_path = run("all", ["--split", "all", *opts, "--curriculum", "1"])
assert all_years["epochs"][0]["val_loss"] is None
assert all_pkg["fit_period"][-1].startswith(last_year)
assert all_pkg["climatology_fit_periods"][0][-1].startswith(first_year)
seasons = weather.dates.astype("datetime64[Y]")
pool = np.arange(2, 199)
order = training_order(pool, seasons, np.random.default_rng(35), True)
assert len(order) == len(pool) and np.count_nonzero(seasons[order] == seasons[-1]) > 110

stop, stopped_pkg, _ = run("stop", [*validation, "--seed", "35", "--learning-rate", "1e-40",
                                   "--curriculum", "1,1,1,1,1"])
assert stop["stopped_early"] and len(stop["epochs"]) == 4 and stopped_pkg["epochs"] == 1
for start, end in ((2008, 2009), (2014, 2015), (2020, 2021)):
    train, val = time_block_split(days, start, end)
    assert str(days[val.start]) == f"{start}-01-01" and str(days[val.stop - 1]) == f"{end}-12-31"
    for span in train:
        windows = window_ends(span.stop - span.start, 14, "all") + span.start
        assert all((windows + 14 < val.start - FOLD_BUFFER_DAYS) |
                   (windows - 2 >= val.stop + FOLD_BUFFER_DAYS))
    print("fold", start, end, "train spans", [(p.start, p.stop) for p in train], "val", (val.start, val.stop))
print("200-day smoke PASS", "epoch1_train_loss", a["epochs"][0]["train_loss"],
      "epoch1_val_loss", a["epochs"][0]["val_loss"], "early_stop_epoch", len(stop["epochs"]),
      "legacy_and_new_checkpoints_engine_load_and_validity PASS", "same_seed_loss_logs_identical PASS")

resume, resume_pkg, _ = run("resume", [*validation, "--curriculum", "1", "--resume", str(old_path)])
assert resume["epochs"][0]["epoch"] == resume_pkg["epochs"] == 2
fold_year = str(np.unique(days.astype("datetime64[Y]"))[1])
fold_start, fold_stop = np.searchsorted(days, np.array([fold_year, str(int(fold_year) + 1)], dtype="datetime64[Y]"))
fold_file = root / "fold-fixture.nc"
with xr.open_dataset(archive) as source:
    source.isel(time=slice(fold_start - 28, fold_stop + 28)).to_netcdf(fold_file)
fold_path = root / "fold" / "airavat.pt"
fold_cli = [*base[:4], str(fold_file), *base[5:], "--folds", fold_year, fold_year,
            "--curriculum", "1", "--output", str(fold_path)]
print(subprocess.list2cmdline(fold_cli), flush=True)
subprocess.run(fold_cli, check=True)
fold_pkg = torch.load(fold_path, map_location="cpu", weights_only=True)
assert fold_pkg["split"] == "fold" and len(fold_pkg["fit_periods"]) == 2
assert fold_pkg["validation_period"] == [f"{fold_year}-01-01", f"{fold_year}-12-31"]
assert json.loads(fold_path.with_suffix(".loss.json").read_text())["epochs"][0]["val_loss"] is not None
fold_engine = Engine(fold_path.parent)
assert not scoring.validity(fold_engine.simulate(weather.physical[:3], day_phase(weather.day_of_year[2]), 2), 2)
print("legacy resume PASS", "buffered fold CLI and engine load PASS")
'@ | & .venv/Scripts/python.exe -B -
```

### Smoke output

```text
Early stopping after 3 epochs without validation improvement
fold 2008 2009 train spans [(0, 1081), (1840, 6209)] val (1095, 1826)
fold 2014 2015 train spans [(0, 3273), (4031, 6209)] val (3287, 4017)
fold 2020 2021 train spans [(0, 5464)] val (5478, 6209)
200-day smoke PASS epoch1_train_loss 0.6302236969937983 epoch1_val_loss 7985.680989583333 early_stop_epoch 4 legacy_and_new_checkpoints_engine_load_and_validity PASS same_seed_loss_logs_identical PASS
legacy resume PASS buffered fold CLI and engine load PASS
```

The plateau smoke run kept epoch 1. The all-years run trained into the
fixture's second year and fit climatology and scale on the first year. The
short partial-season fixtures exercise trainer behavior. Their losses are
smoke diagnostics. The shipped checkpoint retained SHA-256
`30143677fb684a5e3daa92efcc7042de174d63c2ad07d7d509da9b258899796d`.

## Step 4 — E0 benchmark reference

`tests/benchmark.py` rebuilds windows from the five first-target selections
and ten shock targets recorded in `train/validation.json`. The earlier
evaluator resolved those selections to windows. The repository records the
selected targets and does not contain a separate event-search algorithm.
The benchmark verifies the organizer archive by SHA-256, derives the held-out
years from its time axis, and fits the kit's 15-day smoothed calendar-day
climatology on the pre-holdout rows. Scoring uses `monsoon_r1_kit` 1.0.0.

```powershell
& .venv/Scripts/python.exe -B -m tests.benchmark --weights weights --label E0
& .venv/Scripts/python.exe -B tests/benchmark.py --weights tests/_windows/calibration/reference-final/airavat.pt --label CPU-train-only
& .venv/Scripts/python.exe -B -m tests.benchmark --weights weights/airavat.pt --label E0-repeat
```

The E0 run wrote `tests/_windows/benchmark/benchmark-E0.json` and
`tests/_windows/benchmark/benchmark-E0.md`. The timing measures the first
10-step call to `Engine.simulate` after checkpoint load, on four Torch CPU
threads. It excludes HTTP. The cold rollout had no official validity errors.

| Label | Forecast /45 | Day-8 skill | Anti-blur /10 | Propagation | Locality | Stability | Exact zero | Cold 10-step s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 37.15 | -2.58% | 10.00 | 19/20 | 20/20 | 20/20 | 10/10 | 0.0393 |

The E0 repeat produced the same non-timing results. The recreated train-only
reference scored 30.95/45 and −1.94% day-8 skill with the same protocol.
E0 differs from the archived 28.69/45 and −2.01% because the shipped
checkpoint was trained on the full archive, including the five proxy
windows, and the archived score came from a different, train-only checkpoint.
On the same shipped weights, the earlier evaluator's calendar-neighbor
climatology gave 37.63/45 and −2.01%. The kit climatology alone moved that
replay to 37.15/45 and −2.58%. The archived train-only weights were unavailable.

## Step 5 — E1 and E2 preflight

The extended archive is absent from the workspace. The original organizer
archive is available for benchmarking. E1 and E2 have no training or benchmark
results, and the conditional second-seed run is pending.

```powershell
Test-Path -LiteralPath data/r1/india_climate_1979_2021.nc
Test-Path -LiteralPath tests/_windows/india_climate_train.nc
```

```text
False
True
```

The recipe specifies seed 42, width 16, and batches of 48. The first 11
epochs use the documented Gaussian-augmentation stage and curriculum
`1,1,1,2,2,3,4,4,6,8,8` at learning rate 0.002. The final four epochs use
the compact augmentation stage with curriculum `1,2,3,4`, learning rate
0.00035, and far-field weight 18. E1 and E2 require that fixed epoch budget.
Both exclude the final two calendar years from fitting.

E0 numbers below are from `tests/_windows/benchmark/benchmark-E0.json`.
Its proxies overlap its training data. E1 and E2 will score those proxies as
held-out cases. A comparison of E1 with E2 isolates the climatology span
under the shared extended training dataset and seed.

| Run | Climatology fit years | Forecast /45 | CSI skill | RMSE skill d1 / d4 / d8 | Propagation / locality / stability / exact zero | Cold 10-step engine |
|---|---|---:|---:|---|---|---:|
| E0 shipped | 2005–2021 | 37.15 | 10.29% | 26.79% / 2.36% / −2.58% | 19/20 / 20/20 / 20/20 / 10/10 | 39.3 ms |
| E1 | 1979–2019 | — | — | — | — | — |
| E2 | 2005–2019 | — | — | — | — | — |
