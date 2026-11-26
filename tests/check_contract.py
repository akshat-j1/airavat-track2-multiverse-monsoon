#!/usr/bin/env python3
"""Black-box API checks; dependencies: Python stdlib and NumPy only.

Usage: python tests/check_contract.py --url http://localhost:8000 --window window.npz
--window accepts .npy (3,6,32,36), or .npz/JSON containing init_states and
optionally doy_sincos, lat, lon. Without a window, use a physical constant field
and July-like seasonal forcing. Channel order: t2m,tp,u10,v10,msl,tcwv.
--require-real rejects a missing model name or a name containing 'stub'.
"""

import argparse
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


SHAPE = (10, 6, 32, 36)


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, name, condition, detail=""):
        passed = bool(condition)
        self.failed += not passed
        print(f"{'PASS' if passed else 'FAIL'} {name}" + (f": {detail}" if detail else ""), flush=True)
        return passed


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--window", type=Path, help=".npy, .npz or JSON initial window")
    parser.add_argument("--require-real", action="store_true", help="fail if /health reports a stub")
    return parser.parse_args()


def load_window(path):
    metadata = {}
    if path is None:
        init = np.broadcast_to(np.array([300., 1., 3., 0., 101000., 40.])[None, :, None, None],
                               (3, 6, 32, 36)).copy()
    elif path.suffix.lower() == ".npy":
        init = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            metadata = {key: data[key] for key in data.files}
        init = metadata["init_states"]
    elif path.suffix.lower() == ".json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(metadata, list):
            init, metadata = metadata, {}
        else:
            init = metadata["init_states"]
    else:
        raise ValueError("--window must be .npy, .npz, or .json")
    init = np.asarray(init, dtype=np.float64)
    doy = np.asarray(metadata.get("doy_sincos", [0., -1.]), dtype=np.float64)
    if init.shape != (3, 6, 32, 36) or not np.isfinite(init).all():
        raise ValueError("window must be finite with shape (3,6,32,36)")
    if doy.shape != (2,) or not np.isfinite(doy).all():
        raise ValueError("doy_sincos must be a finite [sin, cos] pair")
    return {"init_states": init.tolist(), "doy_sincos": doy.tolist(), "steps": 10}, metadata


def request(url, route, payload=None, raw=None, content_type="application/json", extra_headers=None):
    """Return status, decoded body and complete wall latency, including JSON IO."""
    started = time.perf_counter()
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode("utf-8"))
    headers = {"Content-Type": content_type, **(extra_headers or {})}
    req = Request(url.rstrip("/") + route, data=data, headers=headers,
                  method="GET" if data is None else "POST")
    try:
        response = urlopen(req, timeout=15)
    except HTTPError as exc:
        response = exc
    with response:
        status = response.code
        body = response.read()
    try:
        body = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        body = body.decode("utf-8", errors="replace")
    return status, body, time.perf_counter() - started


def health(checks, args):
    try:
        status, body, _ = request(args.url, "/health")
        checks.check("/health ready", status == 200 and isinstance(body, dict) and body.get("status") == "ready")
        if args.require_real:
            name = body.get("model_name") if isinstance(body, dict) else None
            checks.check("real model required", isinstance(name, str) and bool(name.strip())
                         and "stub" not in name.lower(), f"model_name={name!r}")
    except Exception as exc:
        checks.check("/health ready", False, str(exc))


def forecast(checks, url, payload, label):
    try:
        status, body, elapsed = request(url, "/simulate", payload)
        if not checks.check(f"{label}: HTTP 200", status == 200, f"status={status}"):
            return None
        result = np.asarray(body["rollout"], dtype=np.float64)
        if not checks.check(f"{label}: shape", result.shape == SHAPE, str(result.shape)):
            return None
        checks.check(f"{label}: finite", np.isfinite(result).all())
        return result, body, elapsed
    except Exception as exc:
        checks.check(f"{label}: response", False, str(exc))
        return None


def main():
    args = arguments(__doc__)
    checks = Checks()
    health(checks, args)
    try:
        payload, _ = load_window(args.window)
    except Exception as exc:
        checks.check("load window", False, str(exc))
        return 1
    baseline = forecast(checks, args.url, payload, "base")
    if baseline is not None:
        rollout, body, elapsed = baseline
        checks.check("tp >= 0", (rollout[:, 1] >= 0).all())
        ms = body.get("inference_ms")
        checks.check("inference_ms numeric", type(ms) in (int, float) and math.isfinite(ms) and ms >= 0,
                     repr(ms))
        checks.check("latency < 5s", elapsed < 5., f"{elapsed:.3f}s")
        for name, channel, low, high in (("t2m", 0, 180, 340), ("msl", 4, 80000, 110000),
                                         ("tcwv", 5, 0, 120)):
            values = rollout[:, channel]
            checks.check(f"{name} physical range [{low}, {high}]",
                         ((values >= low) & (values <= high)).all(),
                         f"min={values.min():.4g}, max={values.max():.4g}")
    explicit_null = forecast(checks, args.url, dict(payload, perturbation=None), "null perturbation")
    checks.check("null vs omitted perturbation equal", baseline is not None and explicit_null is not None
                 and np.array_equal(baseline[0], explicit_null[0]))
    for norm in (0.9, 1.1):
        status, response, _ = request(args.url, "/simulate", dict(payload, doy_sincos=[norm, 0]))
        checks.check(f"accept phase norm {norm}", status == 200 and "rollout" in response, f"status={status}")

    cases = [("non-object", []), ("missing fields", {}),
             ("missing init_states", {k: v for k, v in payload.items() if k != "init_states"}),
             ("missing doy_sincos", {k: v for k, v in payload.items() if k != "doy_sincos"})]
    for key, values in (
        ("init_states", [[], "bad", None, [[1], [2, 3]]]),
        ("doy_sincos", [[], [0], [0, 1, 2], ["bad", 1], ["0", 1], [float("inf"), 0], [0, 0], [1.11, 0]]),
        ("steps", [0, -1, 15, 31, 1.5, True, "10", None, float("nan"), float("inf")]),
        ("perturbation", [[], "bad", [[[0]]], {"bad": 1}]),
    ):
        for index, value in enumerate(values):
            cases.append((f"{key} invalid {index + 1} ({value!r})", dict(payload, **{key: value})))
    init = np.asarray(payload["init_states"], dtype=object)
    init[0, 0, 0, 0] = str(init[0, 0, 0, 0])
    cases.append(("numeric string in weather grid", dict(payload, init_states=init.tolist())))
    init[0, 0, 0, 0] = 1e38
    cases.append(("temperature outside physical bounds", dict(payload, init_states=init.tolist())))
    excessive_delta = np.zeros((6, 32, 36))
    excessive_delta[4, 0, 0] = -1e10
    cases.append(("perturbed pressure outside physical bounds", dict(payload, perturbation=excessive_delta.tolist())))
    for key, shape in (("init_states", (3, 6, 32, 36)), ("perturbation", (6, 32, 36))):
        for value in (float("nan"), float("inf")):
            invalid = np.zeros(shape)
            invalid.flat[0] = value
            cases.append((f"{key} nonfinite {value}", dict(payload, **{key: invalid.tolist()})))
    for label, invalid in cases:
        try:
            status, _, _ = request(args.url, "/simulate", invalid)
            checks.check(f"bad input: {label}", status in (400, 422), f"status={status}")
        except Exception as exc:
            checks.check(f"bad input: {label}", False, str(exc))
    try:
        status, _, _ = request(args.url, "/simulate", raw=b'{"init_states":')
        checks.check("bad input: malformed JSON", status in (400, 422), f"status={status}")
    except Exception as exc:
        checks.check("bad input: malformed JSON", False, str(exc))
    status, _, _ = request(args.url, "/simulate", payload, content_type="text/plain")
    checks.check("bad input: wrong media type", status == 415, f"status={status}")
    status, _, _ = request(args.url, "/simulate", raw=b"",
                           extra_headers={"Content-Length": str(8 * 1024 * 1024 + 1)})
    checks.check("bad input: body exceeds 8 MiB", status == 413, f"status={status}")
    return int(checks.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
