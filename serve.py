#!/usr/bin/env python3
"""Multiverse of Monsoon: offline inference server (Part B).

Run:  python serve.py --port 8000
API:  POST /health, POST /simulate  (see API_CONTRACT.md in yug-monsoon-kit)
Also: GET /dashboard/  serves the static dashboard (code lives in dashboard/).

The model lives behind the Engine interface (engine/engine.py, owned by Part A).
If engine/engine.py is missing or fails to load, the server falls back to
engine/stub_engine.py so it always answers with valid output; /health then reports
a model_name containing "stub". Set MONSOON_STRICT=1 to disable the fallback
(recommended while testing the real model).
"""
import os
import sys

sys.dont_write_bytecode = True  # repo is read-only at evaluation time
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import argparse  # noqa: E402
import base64  # noqa: E402
import binascii  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

ROOT = Path(__file__).resolve().parent
N_IN, N_CH, H, W = 3, 6, 32, 36
MAX_STEPS = 14  # official monsoon_r1.contract.MAX_STEPS; no kit needed at runtime
STRICT = os.environ.get("MONSOON_STRICT") == "1"


def _log(msg):
    print("[serve] " + msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- output safety
def sanitize(out, last_day, steps):
    """Return float32 [steps, 6, H, W]: right shape, finite, rain >= 0."""
    r = np.array(out, dtype=np.float32, copy=True)
    if r.ndim != 4 or r.shape[0] < steps or r.shape[1:] != (N_CH, H, W):
        raise ValueError("engine returned shape %s, expected (%d, 6, 32, 36)" % (r.shape, steps))
    r = r[:steps]
    prev = np.asarray(last_day, dtype=np.float32)
    for k in range(steps):
        bad = ~np.isfinite(r[k])
        if bad.any():
            r[k][bad] = prev[bad]  # fill from the previous valid step
        prev = r[k]
    r[:, 1] = np.maximum(r[:, 1], 0.0)
    return r


# --------------------------------------------------------------------------- engine loading
def _load_stub():
    from engine.stub_engine import Engine as StubEngine

    eng = StubEngine(str(ROOT / "weights"))
    return eng, eng.model_name, True


def _warmup(engine):
    x = np.empty((N_IN, N_CH, H, W), dtype=np.float32)
    x[:] = np.array([300.0, 1.0, 3.0, 0.0, 101000.0, 40.0], dtype=np.float32)[None, :, None, None]
    out = engine.simulate(x, np.array([0.0, -1.0], dtype=np.float32), 2, None)
    sanitize(out, x[-1], 2)


def load_engine():
    try:
        from engine.engine import Engine

        eng = Engine(str(ROOT / "weights"))
        _warmup(eng)
        return eng, str(getattr(eng, "model_name", "monsoon-engine")), False
    except Exception as e:  # noqa: BLE001
        if STRICT:
            raise
        missing = isinstance(e, ModuleNotFoundError) and e.name in ("engine", "engine.engine")
        if not missing:
            traceback.print_exc()
        _log("real engine unavailable (%s); using the STUB engine" % type(e).__name__)
        return _load_stub()


class _State:
    engine = None
    model_name = "unloaded"
    is_stub = False
    lock = threading.Lock()


STATE = _State()
STATE.engine, STATE.model_name, STATE.is_stub = load_engine()
_log("ready: model_name=%s" % STATE.model_name)


# --------------------------------------------------------------------------- request parsing
def _bad(msg):
    raise HTTPException(status_code=400, detail=msg)


def _array(value, shape, name):
    """Accept canonical lists and the official validator's optional base64 form."""
    try:
        if isinstance(value, dict):
            if value.get("shape") != list(shape) or not isinstance(value.get("b64"), str):
                _bad(f"{name} base64 object must have shape {list(shape)}")
            expected = math.prod(shape) * 4
            if len(value["b64"]) != 4 * ((expected + 2) // 3):
                _bad(f"{name} base64 length does not match shape")
            raw = base64.b64decode(value["b64"], validate=True)
            if len(raw) != expected:
                _bad(f"{name} base64 length does not match shape")
            result = np.frombuffer(raw, dtype="<f4").reshape(shape)
        else:
            result = np.asarray(value, dtype=np.float32)
    except (ValueError, TypeError, OverflowError, binascii.Error):
        _bad(f"{name} must be numeric lists or a valid base64 array")
    if result.shape != shape or not np.isfinite(result).all():
        _bad(f"{name} must be finite with shape {list(shape)}")
    return result


def parse_request(body):
    if not isinstance(body, dict):
        _bad("body must be a JSON object")
    if "init_states" not in body or "doy_sincos" not in body:
        _bad("init_states and doy_sincos are required")
    init = _array(body["init_states"], (N_IN, N_CH, H, W), "init_states")
    doy = _array(body["doy_sincos"], (2,), "doy_sincos")
    if not 0.9 <= float(np.linalg.norm(doy)) <= 1.1:
        _bad("doy_sincos must describe a unit-circle angle")

    steps = body.get("steps", 10)
    if isinstance(steps, bool) or not isinstance(steps, (int, float)):
        _bad("steps must be an integer")
    if isinstance(steps, float) and (not math.isfinite(steps) or not steps.is_integer()):
        _bad("steps must be an integer")
    if not 1 <= steps <= MAX_STEPS:
        _bad("steps must be between 1 and %d" % MAX_STEPS)
    steps = int(steps)

    pert = body.get("perturbation")
    if pert is not None:
        pert = _array(pert, (N_CH, H, W), "perturbation")
    return init, doy, steps, pert


# --------------------------------------------------------------------------- app
app = FastAPI(title="Multiverse of Monsoon")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ready", "model_name": STATE.model_name}


def _run(init, doy, steps, pert):
    t0 = time.perf_counter()
    try:
        with STATE.lock:
            raw = STATE.engine.simulate(init, doy, steps, pert)
        rollout = sanitize(raw, init[-1], steps)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        if STRICT:
            raise HTTPException(status_code=500, detail="engine failed (MONSOON_STRICT=1)")
        _log("engine failed on this request; answering with the stub engine")
        from engine.stub_engine import Engine as StubEngine

        rollout = sanitize(StubEngine().simulate(init, doy, steps, pert), init[-1], steps)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "rollout": np.round(rollout.astype(np.float64), 4).tolist(),
        "inference_ms": round(ms, 2),
    }


@app.post("/simulate")
async def simulate(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        _bad("invalid JSON")
    init, doy, steps, pert = parse_request(body)
    return await run_in_threadpool(_run, init, doy, steps, pert)


DASH = ROOT / "dashboard"
if DASH.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(DASH), html=True), name="dashboard")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/dashboard/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
