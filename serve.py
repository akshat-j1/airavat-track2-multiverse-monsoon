#!/usr/bin/env python3
"""Multiverse of Monsoon offline inference server.

Run:  python serve.py --port 8000
API:  POST /health, POST /simulate  (see API_CONTRACT.md in yug-monsoon-kit)
Also: GET /dashboard/  serves the static dashboard (code lives in dashboard/).

The trained checkpoint is required by default. MONSOON_DEV_STUB=1 permits the
labelled development stub if the real engine cannot start.
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
import json  # noqa: E402
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
MAX_BODY_BYTES = 8 * 1024 * 1024
PHASE_TOLERANCE = 1e-6
INPUT_BOUNDS = (
    ("t2m", 150, 350), ("tp", 0, 1000), ("u10", -100, 100),
    ("v10", -100, 100), ("msl", 80000, 115000), ("tcwv", 0, 150),
)


def _log(msg):
    print("[serve] " + msg, file=sys.stderr, flush=True)


def checked_rollout(out, steps):
    rollout = np.asarray(out)
    expected = (steps, N_CH, H, W)
    if rollout.shape != expected:
        raise ValueError(f"engine returned shape {rollout.shape}, expected {expected}")
    if not np.issubdtype(rollout.dtype, np.floating) or not np.isfinite(rollout).all():
        raise ValueError("engine returned non-finite or non-floating-point data")
    if (rollout[:, 1] < 0).any():
        raise ValueError("engine returned negative precipitation")
    return rollout


# --------------------------------------------------------------------------- engine loading
def _load_stub():
    from engine.stub_engine import Engine as StubEngine

    eng = StubEngine(str(ROOT / "weights"))
    return eng, eng.model_name, True


def _warmup(engine):
    x = np.empty((N_IN, N_CH, H, W), dtype=np.float32)
    x[:] = np.array([300.0, 1.0, 3.0, 0.0, 101000.0, 40.0], dtype=np.float32)[None, :, None, None]
    out = engine.simulate(x, np.array([0.0, -1.0], dtype=np.float32), 2, None)
    checked_rollout(out, 2)


def load_engine():
    try:
        weights_dir = ROOT / "weights"
        checkpoint = weights_dir / "airavat.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
        with checkpoint.open("rb") as stream:
            if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Expected the trained checkpoint at {checkpoint}, found a Git LFS pointer")
        from engine.engine import Engine

        eng = Engine(str(weights_dir))
        _warmup(eng)
        return eng, str(getattr(eng, "model_name", "monsoon-engine")), False
    except Exception as exc:
        if os.environ.get("MONSOON_DEV_STUB") != "1":
            raise
        _log(f"Development stub enabled after engine startup failed: {exc}")
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


def _numeric_list(value, shape):
    if not shape:
        return type(value) in (int, float)
    return isinstance(value, list) and len(value) == shape[0] and all(
        _numeric_list(item, shape[1:]) for item in value
    )


def _array(value, shape, name):
    """Accept canonical lists and the official validator's optional base64 form."""
    try:
        if isinstance(value, dict):
            dimensions = value.get("shape")
            if (dimensions != list(shape) or any(type(dim) is not int for dim in dimensions)
                    or not isinstance(value.get("b64"), str)):
                _bad(f"{name} base64 object must have shape {list(shape)}")
            expected = math.prod(shape) * 4
            if len(value["b64"]) != 4 * ((expected + 2) // 3):
                _bad(f"{name} base64 length does not match shape")
            raw = base64.b64decode(value["b64"], validate=True)
            if len(raw) != expected:
                _bad(f"{name} base64 length does not match shape")
            result = np.frombuffer(raw, dtype="<f4").reshape(shape)
        else:
            if not _numeric_list(value, shape):
                _bad(f"{name} must be JSON numbers with shape {list(shape)}")
            result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError, binascii.Error):
        _bad(f"{name} must be numeric lists or a valid base64 array")
    if result.shape != shape or not np.isfinite(result).all():
        _bad(f"{name} must be finite with shape {list(shape)}")
    return result


def _validate_state(state, name):
    for channel, (variable, low, high) in enumerate(INPUT_BOUNDS):
        values = state[..., channel, :, :]
        if (values < low).any() or (values > high).any():
            _bad(f"{name} {variable} must be between {low} and {high} in physical units")


def parse_request(body):
    if not isinstance(body, dict):
        _bad("body must be a JSON object")
    if "init_states" not in body or "doy_sincos" not in body:
        _bad("init_states and doy_sincos are required")
    init = _array(body["init_states"], (N_IN, N_CH, H, W), "init_states")
    _validate_state(init, "init_states")
    doy = _array(body["doy_sincos"], (2,), "doy_sincos")
    norm = math.hypot(*doy)
    if not 0.9 - PHASE_TOLERANCE <= norm <= 1.1 + PHASE_TOLERANCE:
        _bad("doy_sincos must describe a unit-circle angle")
    # Normalize in full precision before handing the phase to the float32 Engine.
    phase = (doy.astype(np.float64) / norm).astype(np.float32)

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
        with np.errstate(over="ignore"):
            _validate_state(init[-1].astype(np.float64) + pert, "perturbed last input")
        pert = pert.astype(np.float32, copy=False)
    return init.astype(np.float32, copy=False), phase, steps, pert


async def _read_json(request):
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "application/json" and not (media_type.startswith("application/") and media_type.endswith("+json")):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    content_length = request.headers.get("content-length", "")
    if content_length.isdecimal() and int(content_length) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="JSON request exceeds the 8 MiB limit")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="JSON request exceeds the 8 MiB limit")
        raw.extend(chunk)
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        _bad("invalid JSON")


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
        rollout = checked_rollout(raw, steps)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Forecast engine failed")
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "rollout": np.round(rollout.astype(np.float64), 4).tolist(),
        "inference_ms": round(ms, 2),
    }


@app.post("/simulate")
async def simulate(request: Request):
    body = await _read_json(request)
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
