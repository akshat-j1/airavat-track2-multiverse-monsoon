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
