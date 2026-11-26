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
fixture = json.loads(Path("tests/fixtures/proxy_events.json").read_text(encoding="utf-8"))
weather = load_weather(Path("tests/_windows/india_climate_train.nc"))
cut = validation_start(weather.dates, 2)
cases = list(dict.fromkeys(f'{e["event"]}={e["first_target"]}' for e in fixture["event_forecast"]["events"]))
for name, weights in (("shipped", path.parent), ("recreated_train_only", Path("tests/_windows/calibration/reference-final"))):
    checkpoint = weights / "airavat.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    engine = Engine(weights)
    score = evaluate_events(engine, weather, cut, cases)
    shocks = evaluate_shocks(engine, weather, Path("assets/stats.json"), 0, cut, fixture["shock_first_target_dates"])
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
and ten shock targets recorded in `tests/fixtures/proxy_events.json`. The earlier
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

## Step 6 — Prepare the capacity and loss sweep

The extended archive is still absent (`Test-Path -LiteralPath data/r1/india_climate_1979_2021.nc`
returned `False`). E1 and E2 have no result, so the winning data setup,
trained width sweep, candidate ranking, and three-fold confirmation are
pending. The shipped checkpoint remains unchanged.

The trainer now supports opt-in scheduled sampling for unpaired windows, the
existing event-weighted term, soft CSI at the BCE rain threshold, and a
spatial gradient-energy term. The paired-shock and far-field terms retain
their original formulas. A common validation horizon and proxy exclusion
keep screening losses comparable. The 0.1 loss weights below are smoke
inputs. No hyperparameter tuning was performed.

### Smoke commands

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
@'
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import xarray as xr
from engine.engine import Engine
from monsoon_r1 import scoring
from train.common import day_phase, exclude_proxy_windows, load_weather, recorded_proxy_cases, validation_start, window_ends
from train.train import rain_spatial_energy_loss, scheduled_history, soft_csi_loss

root = Path("tests/_windows")
fixture = root / "trainer-smoke/fixture.nc"
base = [sys.executable, "-B", "-m", "train.train", str(fixture), "--device", "cpu",
        "--threads", "4", "--width", "4", "--batch-size", "32", "--split", "train", "--val-years", "1", "--seed", "35"]
out = root / "rollout-smoke/airavat.pt"
command = [*base, "--output", str(out), "--curriculum", "2,8,10", "--scheduled-sampling",
           "--event-rain-weight", "0.1", "--rain-csi-weight", "0.1", "--rain-energy-weight", "0.1",
           "--validation-steps", "10", "--exclude-proxy-windows"]
print(subprocess.list2cmdline(command), flush=True)
subprocess.run(command, check=True)
history = json.loads(out.with_suffix(".loss.json").read_text())
assert [epoch["train_horizon"] for epoch in history["epochs"]] == [2, 8, 10]
assert [epoch["teacher_forcing"] for epoch in history["epochs"]] == [0.5, 0.25, 0.0]
assert all(epoch["validation_horizon"] == 10 and np.isfinite(epoch["val_loss"]) for epoch in history["epochs"])
weather = load_weather(fixture)
engine = Engine(out.parent)
assert not scoring.validity(engine.simulate(weather.physical[:3], day_phase(weather.day_of_year[2]), 10), 10)
print("Scheduled 2/8/10 smoke PASS", [(e["train_loss"], e["val_loss"]) for e in history["epochs"]], flush=True)

paired = torch.arange(6 * 3 * 4, dtype=torch.float32).reshape(6, 3, 1, 2, 2)
actual = torch.zeros(4, 1, 2, 2)
rng = torch.Generator().manual_seed(35)
mixed = scheduled_history(paired, actual, 2, 1.0, rng)
assert torch.equal(mixed[[0, 1, 4, 5]], paired[[0, 1, 4, 5]])
assert torch.equal(mixed[:, :-1], paired[:, :-1]) and torch.equal(mixed[2:4, -1], actual[2:])
truth = torch.tensor([[[0.0, 14.0], [16.0, 0.0]]]).log1p()
candidate = torch.tensor([[[1.0, 5.0], [8.0, 3.0]]]).log1p().requires_grad_()
heavy = (truth > np.log1p(scoring.RAIN_THRESH)).float()
csi = soft_csi_loss(3 * (candidate - np.log1p(scoring.RAIN_THRESH)), heavy)
energy = rain_spatial_energy_loss(candidate, truth)
assert csi > soft_csi_loss(3 * (truth - np.log1p(scoring.RAIN_THRESH)), heavy)
assert energy > rain_spatial_energy_loss(truth, truth) == 0
(csi + energy).backward()
assert torch.isfinite(candidate.grad).all() and candidate.grad.abs().sum() > 0
print("Paired-state preservation and rain loss gradients PASS", float(csi.detach()), float(energy.detach()), flush=True)

with xr.open_dataset(root / "india_climate_train.nc") as archive:
    dates = archive.time.values.astype("datetime64[D]")
cut = validation_start(dates, 2)
fixture = json.loads(Path("tests/fixtures/proxy_events.json").read_text(encoding="utf-8"))
starts = [date for _, date in recorded_proxy_cases(fixture)]
for horizon in (8, 10):
    windows = window_ends(len(dates), horizon, "validation", val_start=cut)
    kept = exclude_proxy_windows(dates, windows, horizon, starts)
    for idx in np.searchsorted(dates, np.asarray(starts, dtype="datetime64[D]")):
        assert ((kept + horizon < idx - 3) | (kept - 2 > idx + 13)).all()
    if horizon == 8:
        assert (len(windows), len(kept)) == (721, 592)
    print("Proxy exclusion PASS", horizon, len(windows), len(kept), flush=True)

legacy = root / "rollout-legacy/airavat.pt"
subprocess.run([*base, "--clim-years", "2005", "2005", "--sample-weight-recent",
                "--curriculum", "1,2,2", "--max-epochs", "2", "--output", str(legacy)], check=True)
log = json.loads(legacy.with_suffix(".loss.json").read_text())
assert log["epochs"][0]["train_loss"] == 0.6302236969937983
assert log["epochs"][0]["val_loss"] == 7985.680989583333
print("Previous defaults reproduced epoch 1 exactly", log["epochs"][0], flush=True)
'@ | & .venv/Scripts/python.exe -B -
```

```text
Scheduled 2/8/10 smoke PASS [(0.9771703084309896, 47386.09375), (1.4254299587673611, 45693.38352272727), (1.5782018141313032, 43075.83238636364)]
Paired-state preservation and rain loss gradients PASS 0.7594814896583557 1.7446913719177246
Proxy exclusion PASS 8 721 592
Proxy exclusion PASS 10 719 585
Previous defaults reproduced epoch 1 exactly {'epoch': 1, 'train_horizon': 1, 'train_loss': 0.6302236969937983, 'teacher_forcing': 0.0, 'validation_horizon': 2, 'val_loss': 7985.680989583333}
```

### Capacity probes

Each probe times one first 10-step engine call on a real fixture in a fresh
process with four Torch CPU threads. Width 16 uses shipped weights. The
wider probes use seeded random network tensors and the shipped static and
normalization tensors. These probes measure capacity and direct CPU
inference. Candidate quality and the cold-server latency gate are pending.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
@'
from pathlib import Path
import subprocess
import sys

program = """
from pathlib import Path
import sys
import time

import numpy as np
import torch
from engine.engine import Engine
from model.unet import AnomalyUNet
from monsoon_r1 import scoring

torch.set_num_threads(4)
torch.manual_seed(42)
width = int(sys.argv[1])
weights = Path("weights/airavat.pt")
package = torch.load(weights, map_location="cpu", weights_only=True)
if width != package["width"]:
    package.update(width=width, model=AnomalyUNet(width).state_dict())
    weights = Path("tests/_windows/width-probe") / str(width) / "airavat.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, weights)
with np.load("tests/_windows/real.npz", allow_pickle=False) as fixture:
    history, phase = fixture["init_states"], fixture["doy_sincos"]
engine = Engine(weights.parent)
start = time.perf_counter()
out = engine.simulate(history, phase, 10)
elapsed = time.perf_counter() - start
assert not scoring.validity(out, 10)
print(width, sum(v.numel() for v in package["model"].values()), weights.stat().st_size, round(elapsed * 1000, 2))
"""
print("width parameters checkpoint_bytes cold_engine_ms", flush=True)
for width in (16, 24, 32, 48):
    subprocess.run([sys.executable, "-B", "-c", program, str(width)], check=True)
'@ | & .venv/Scripts/python.exe -B -
& .venv/Scripts/python.exe -B -m tests.benchmark --weights weights --label E0-capacity
```

| Width | Network | Parameters | Checkpoint bytes | Cold 10-step engine |
|---:|---|---:|---:|---:|
| 16 | Shipped | 32,342 | 344,546 | 35.53 ms |
| 24 | Untrained probe | 69,246 | 492,078 | 83.94 ms |
| 32 | Untrained probe | 119,974 | 695,342 | 49.69 ms |
| 48 | Untrained probe | 262,902 | 1,267,054 | 74.43 ms |

The updated E0 benchmark emitted `16 | 32342 | 344546 | 37.15 | -2.58%` for
width, parameters, bytes, forecast points, and day-8 skill. Its first
10-step engine call took 0.0370 s. The forecast and shock fields matched the
previous E0 JSON.

## Step 7 — Ensemble inference and held-out shrinkage

The extended archive remains unavailable and there is no winning
configuration to train across seeds. E4 and E5 below are smoke bundles.
They share one recreated train-only network with controlled changes to the
member rain biases. They are not independently trained candidates.

Added optional ensemble loading and pre-decode anomaly averaging. Kept the
format at version 1. The fitter used 573 complete 14-step holdout windows
after excluding every window sharing inputs or targets with a proxy.
It fixed the rain multipliers at 1. The benchmark reports both the peak
ratios for individual scored leads and the ratio of the maxima across
days 1, 4, and 8.

### Commands

The legacy snapshot in this command was captured with `Engine("weights")`
before editing the engine. It uses the same `real.npz` initial state and
warm-ocean shock as the comparison.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
@'
from pathlib import Path
import numpy as np
import torch
from engine.engine import Engine, MAX_STEPS
from monsoon_r1 import scoring
from train.fit_shrinkage import check_training_scope

root = Path("tests/_windows/ensemble-smoke")
e4 = root / "e4"
e4.mkdir(exist_ok=True)
torch.set_num_threads(4)
with np.load("tests/_windows/real.npz", allow_pickle=False) as sample:
    history, phase = sample["init_states"], sample["doy_sincos"]
legacy = np.load(root / "legacy.npz", allow_pickle=False)
shipped = Engine("weights")
shock = scoring.make_shock("warm_ocean", history[-1], scoring.STD_DEFAULT)
assert np.array_equal(legacy["null"], shipped.simulate(history, phase, 14))
assert np.array_equal(legacy["shock"], shipped.simulate(history, phase, 14, shock))
print("Legacy forecasts and warm-ocean shocks bit-identical", flush=True)

reference = root.parent / "calibration/reference-final/airavat.pt"
for filename, shift in (("airavat.pt", 0), ("airavat-member-a.pt", .02), ("airavat-member-b.pt", -.025)):
    package = torch.load(reference, map_location="cpu", weights_only=True)
    package["model"]["output.bias"][1] += shift
    torch.save(package, e4 / filename)

engine = Engine(e4)
base = engine.simulate(history, phase, MAX_STEPS)
for delta in (scoring.make_shock("null", history[-1], scoring.STD_DEFAULT), np.full((6, 32, 36), 1e-30, dtype="f4")):
    assert np.array_equal(base, engine.simulate(history, phase, MAX_STEPS, delta))
assert not scoring.validity(base, MAX_STEPS)
physical_members = [Engine(path).simulate(history, phase, MAX_STEPS) for path in sorted(e4.glob("*.pt"))]
from model.unet import decode_physical
batch = torch.from_numpy(history.astype('f4'))[None]
phase_f32 = phase.astype('f4')
phases = torch.from_numpy(phase_f32 / float(np.linalg.norm(phase_f32)))[None]
components = [Engine(path)._encoded_forecast(batch, phases, MAX_STEPS) for path in sorted(e4.glob("*.pt"))]
stack = torch.stack([torch.stack(states, dim=1) for states, _ in components])
expected = np.stack([decode_physical(stack.mean(dim=0)[:, day], day + 1)[0].numpy() for day in range(MAX_STEPS)])
np.testing.assert_allclose(expected[:, 1], base[:, 1], rtol=2e-6, atol=2e-6)
assert np.max(np.abs(np.mean(physical_members, axis=0)[:, 1] - base[:, 1])) > 1e-5
try:
    check_training_scope([Path("weights/airavat.pt")], np.datetime64("2020-01-01"))
except ValueError:
    pass
else:
    raise AssertionError("Expected the fitter to reject in-sample E0")
print("Three members: anomaly-before-decode, exact null/subprecision, validity, and in-sample rejection PASS", flush=True)
'@ | & .venv/Scripts/python.exe -B -
& .venv/Scripts/python.exe -B -m train.fit_shrinkage tests/_windows/india_climate_train.nc --weights tests/_windows/ensemble-smoke/e4 --output tests/_windows/ensemble-smoke/e5
& .venv/Scripts/python.exe -B -m tests.benchmark --weights tests/_windows/ensemble-smoke/e4 --label E4-smoke
& .venv/Scripts/python.exe -B -m tests.benchmark --weights tests/_windows/ensemble-smoke/e5 --label E5-smoke
& .venv/Scripts/python.exe -B -m tests.benchmark --weights weights --label E0-ensemble-check
```

```text
Legacy forecasts and warm-ocean shocks bit-identical
Three members: anomaly-before-decode, exact null/subprecision, validity, and in-sample rejection PASS
E0 official forecast fields and E4/E5 rain peaks match exactly
```

Cold HTTP measurements reused the four-physical-core launcher from Step 1.
The only change to the launched process was routing its `Engine` constructor
to the scratch bundle. The script checked HTTP response validity and the
1.5 s limit. The application source stayed untouched.

```powershell
@'
from pathlib import Path
import subprocess
import sys

doc = Path("train/experiments.md").read_text(encoding="utf-8")
cold = doc.split("### Cold server on four physical cores", 1)[1]
program = cold.split("@'\n", 1)[1].split("\n'@", 1)[0]
program = program.replace("import json, os, socket, subprocess, time",
                          "import json, os, socket, subprocess, sys, time")
program = program.replace("runpy.run_path('serve.py', run_name='__main__')", """
import os
from engine import engine as engine_module
real_engine = engine_module.Engine
engine_module.Engine = lambda _: real_engine(os.environ['AIRAVAT_SMOKE_WEIGHTS'])
runpy.run_path('serve.py', run_name='__main__')""")
program = program.replace('env.pop("MONSOON_DEV_STUB", None)',
    'env.pop("MONSOON_DEV_STUB", None)\nenv["AIRAVAT_SMOKE_WEIGHTS"] = str(Path(sys.argv[1]).resolve())')
program = program.replace("elapsed = time.perf_counter() - start",
                          "elapsed = time.perf_counter() - start\n    assert elapsed < 1.5")
for name in ("e4", "e5"):
    subprocess.run([sys.executable, "-B", "-c", program, str(Path("tests/_windows/ensemble-smoke") / name)], check=True)
'@ | & .venv/Scripts/python.exe -B -
```

### Results

| Run | Forecast /45 | Day-4 skill | Day-8 skill | Day-8 min variance ratio | Propagation / locality / stability / exact zero | Cyclone peak ratio | Heavy-rain peak ratio | First 10-step HTTP |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| E0 shipped | 37.15 | +2.36% | −2.58% | 0.604 | 19/20 / 20/20 / 20/20 / 10/10 | 0.498 | 0.233 | 0.204 s from Step 1 |
| E4 smoke | 30.71 | +1.40% | −1.95% | 0.554 | 19/20 / 20/20 / 20/20 / 10/10 | 0.907 | 0.278 | 0.273 s |
| E5 smoke | 31.37 | +1.76% | −1.50% | 0.554 | 19/20 / 20/20 / 20/20 / 10/10 | 0.907 | 0.278 | 0.286 s |

The benchmark-level rain peak ratios were exactly equal for E4 and E5,
including every reported lead for each rain proxy. Both smoke bundles
stayed below the latency limit and passed the shock gates. Their forecast
scores remained below E0. No checkpoint was promoted.

## Step 8 — Expanded causal audit

Added a repeatable audit and ran it on E0 and the E4 and E5 smoke bundles.
The extended archive is still absent, so there is no trained winner to
audit. The same input draws were used for all three bundles. No model,
transport, checkpoint, or server changes were needed for this audit.

```powershell
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& .venv/Scripts/python.exe -B -m tests.audit_causality --weights weights --output tests/_windows/causality-audit/E0.json
& .venv/Scripts/python.exe -B -m tests.audit_causality --weights tests/_windows/ensemble-smoke/e4 --output tests/_windows/causality-audit/E4-smoke.json
& .venv/Scripts/python.exe -B -m tests.audit_causality --weights tests/_windows/ensemble-smoke/e5 --output tests/_windows/causality-audit/E5-smoke.json
```

The audit reports the original ten official windows separately from the
five proxies plus ten random holdout windows. It uses seed 42 and samples
3, 2, 3, and 2 random windows from DJF, MAM, JJAS, and ON. The random
windows exclude those sharing rows with the proxies or archived shock
windows. The ten judge draws use a raised-cosine bump, signed amplitude
between 0.5 and 3 standard deviations, and radius between 2 and 8 cells.
A draw that would make initial rain or moisture negative uses a positive
amplitude. The kit gets the tight grid-aligned box around the bump as
the judge shock region. Weak wind is the bottom quartile of daily
Arabian Sea box mean wind speed in the holdout, at or below 3.6195 m/s.

| Bundle | Official propagation | Official exact zero | Expanded propagation | Expanded exact zero | Judge failures | Invalid / saturated / growing-drift 14-step rollouts |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 19/20 | 10/10 | 29/30 | 15/15 | 3/10 | 0/20 / 4/20 / 20/20 |
| E4 smoke | 19/20 | 10/10 | 29/30 | 15/15 | 3/10 | 0/20 / 0/20 / 20/20 |
| E5 smoke | 19/20 | 10/10 | 29/30 | 15/15 | 3/10 | 0/20 / 0/20 / 20/20 |

On E5 smoke, both non-null shocks passed nonzero, localized, stable,
and no-teleport on 15/15 windows each. Warm-ocean propagation was 14/15,
with day-one near fraction 99.8848% mean and 99.6729% minimum.
Monsoon-stall propagation was 15/15, with near fraction 99.9835% mean
and 99.9387% minimum. Stall passed in all seasons (DJF 4/4, MAM 4/4,
JJAS 5/5, and ON 2/2) and in all five distinct weak-wind cases across
the two panels. The common official propagation miss was warm ocean
with last input on 2020-05-17.

All 15 expanded null shocks were bit-exact controls. The kit reported
zero nonzero, locality, and propagation flags and 15/15 stability and
no-teleport flags. Its near fraction is zero when response energy is zero.

All three bundles failed outward propagation on the same judge draws.
They remained finite, nonnegative in rain, local, stable, and below
the kit's teleportation limit.

| Last input | Channel | Radius |
|---|---|---:|
| 2021-01-13 | t2m | 6.97 cells |
| 2021-05-09 | t2m | 7.82 cells |
| 2021-10-16 | tcwv | 6.20 cells |

For each 14-step proxy rollout, drift is the RMS across channels of
latitude-weighted domain-mean forecast bias against observations, in
channel standard deviations. Growth compares the mean of days 11–14
with days 1–4. Mean early and late drift across the 20 rollouts was
0.0948 and 0.1938 for E0, 0.0963 and 0.1852 for E4, and 0.0959 and
0.1921 for E5. E0 reached the lower moisture bound in the four winter
disturbance rollouts. Exact output-bound hits were absent in E4 and E5.
The judge and drift goals remain unmet. The weak-wind transport-change
condition did not trigger.

## Step 9 — Submission preflight

The extended training archive is absent and no candidate cleared the earlier
forecast and causality gates. Full-archive retraining was not run. The
existing checkpoint remains the submission file. Added the extended-archive
build commands to the training README and appended the available held-out
diagnostics to `train/validation.json`. The new validation section identifies
the smoke bundles and records their commands, seed, data span, and evaluation
environment. It contains no E0 in-sample forecast numbers.

### Clean-clone verification of the existing checkpoint

```powershell
$clone = Join-Path $env:TEMP ('airavat-submission-' + [Guid]::NewGuid().ToString('N'))
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --no-local --single-branch --branch fix/calibration . $clone
git -C $clone check-attr filter -- weights/airavat.pt
git -C $clone cat-file -s HEAD:weights/airavat.pt
@'
import json, os, socket, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen

from monsoon_r1 import repo_scan

clone = Path(sys.argv[1]).resolve()
runtime = Path("tests/_windows/runtime/Scripts/python.exe").resolve()
status = ["git", "-C", str(clone), "--no-optional-locks", "status", "--porcelain"]
assert not subprocess.check_output(status).strip()
print("clean clone", subprocess.check_output(["git", "-C", str(clone), "rev-parse", "HEAD"], text=True).strip(), flush=True)
scan = repo_scan.scan(clone)
print("repo_scan.scan()", json.dumps(scan), flush=True)
assert scan["points"] == 15 and not scan["flags"]
weights = list((clone / "weights").glob("*.pt"))
for weight in weights:
    with weight.open("rb") as stream:
        assert not stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
total_bytes = sum(weight.stat().st_size for weight in weights)
assert 0 < total_bytes < 50_000_000
print("Git blob weights", [(weight.name, weight.stat().st_size) for weight in weights],
      "total_bytes", total_bytes, "< 50000000 PASS", flush=True)

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"
env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
env.pop("MONSOON_DEV_STUB", None)
command = [str(runtime), "-B", "serve.py", "--host", "127.0.0.1", "--port", str(port)]
print(subprocess.list2cmdline(command), flush=True)
server = subprocess.Popen(command, env=env, cwd=clone)
try:
    for _ in range(150):
        if server.poll() is not None:
            raise RuntimeError("serve.py exited before readiness")
        try:
            with urlopen(url + "/health", timeout=1) as response:
                health = json.load(response)
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise TimeoutError("Server did not start")
    print("GET /health", json.dumps(health), flush=True)
    assert health == {"status": "ready", "model_name": "airavat-anomaly-unet"}
    validator = [sys.executable, "-B", "-m", "monsoon_r1.schema_validator", url]
    print(subprocess.list2cmdline(validator), flush=True)
    subprocess.run(validator, cwd=clone, env=env, check=True)
finally:
    if server.poll() is None:
        server.terminate()
        server.wait(timeout=10)
assert not subprocess.check_output(status).strip()
print("clean clone startup, repo scan, schema validator, and plain weight size PASS", flush=True)
'@ | & .venv/Scripts/python.exe -B - $clone
```

```text
weights/airavat.pt: filter: unspecified
344546
clean clone a0124434d44199d1dcf68e5e272ee7c819e16f58
repo_scan.scan() points=15.0 anti_cheat_points=8.0 code_health_points=7.0 flags=[]
Git blob weights [('airavat.pt', 344546)] total_bytes 344546 < 50000000 PASS
[serve] ready: model_name=airavat-anomaly-unet
GET /health {"status": "ready", "model_name": "airavat-anomaly-unet"}
health: OK
  [JSON lists (canonical)] no perturbation  OK   0.25s
  [JSON lists (canonical)] warm_ocean      OK   0.27s
  [JSON lists (canonical)] zeros           OK   0.20s
  [base64 (also tolerated)] no perturbation OK  0.17s
  [base64 (also tolerated)] warm_ocean      OK  0.22s
  [base64 (also tolerated)] zeros           OK  0.16s
RESULT: all checks passed
clean clone startup, repo scan, schema validator, and plain weight size PASS
```

The clean checkout stayed clean after verification. These checks apply to
the existing Git blob. The new extended-data submission remains pending.

### Local archive-build smoke test

This exercises the documented merge and export using adjacent five-day
samples from the organizer archive. It checks exact preservation of the
second sample's fields and static grid. The historical data download was
outside this local test.

```powershell
@'
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import xarray as xr
from train.common import load_weather

readme = Path("train/README.md").read_text(encoding="utf-8")
section = readme.split("## Build the extended training archive", 1)[1].split("## Reproduce the shipped checkpoint", 1)[0]
convert, export = re.findall(r"<<'PY'\n(.*?)\nPY\n", section, re.S)
with TemporaryDirectory(prefix="airavat-archive-smoke-") as directory:
    work = Path(directory)
    with xr.open_dataset("tests/_windows/india_climate_train.nc") as source:
        source.isel(time=slice(0, 5)).to_netcdf(work / "history.nc")
        source.isel(time=slice(5, 10)).to_netcdf(work / "organizer.nc")
    python = [sys.executable, "-B"]
    for stem in ("history", "organizer"):
        subprocess.run([*python, "-c", convert, str(work / f"{stem}.nc"), str(work / f"{stem}.npz")], check=True)
    subprocess.run([*python, "-m", "monsoon_r1.build_dataset", "merge", "--parts",
                    str(work / "history.npz"), str(work / "organizer.npz"),
                    "--out", str(work / "extended.npz")], check=True)
    subprocess.run([*python, "-c", export, str(work / "organizer.nc"),
                    str(work / "extended.npz"), str(work / "extended.nc")], check=True)
    assert len(load_weather(work / "extended.nc").dates) == 10
    print("Local archive merge/export round trip PASS (10 days)", flush=True)
'@ | & .venv/Scripts/python.exe -B -
```

```text
merged (10, 6, 32, 36) 2005-01-01 2005-01-10
Local archive merge/export round trip PASS (10 days)
```
