#!/usr/bin/env python3
"""Check deterministic, localized and spreading shock responses over ten steps.

Usage: python tests/check_shocks.py --url http://localhost:8000 --window window.npz
Window formats and --require-real match check_contract.py. Optional window keys
lat (32,) and lon (36,), or both (32,36), locate the nearest ocean centres.
Without coordinates, use the CONTRACT.md grid: rows south-to-north from 6.5 to
37.5 degrees N; columns from 65.5 to 100.5 degrees E. Bay: 16N,88E; Arabian: 16N,70E.
Radius is 4.5 cells. Energy is the sum of squared response over all six channels;
footprint is the number of cells containing 90% of energy, and mean distance is
energy-weighted distance to the shock centre. All comparisons use shocked minus
base forecasts, never absolute forecast fields.
"""

import numpy as np

from check_contract import Checks, arguments, forecast, health, load_window


def centre(metadata, latitude, longitude):
    lat = np.asarray(metadata.get("lat", np.arange(6.5, 38.5)), dtype=float)
    lon = np.asarray(metadata.get("lon", np.arange(65.5, 101.5)), dtype=float)
    if lat.shape == (32,) and lon.shape == (36,):
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    if lat.shape != (32, 36) or lon.shape != (32, 36):
        raise ValueError("lat/lon must be (32,)/(36,) vectors or (32,36) grids")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("lat/lon must be finite")
    delta_lon = (lon - longitude + 180.) % 360. - 180.
    distance = (lat - latitude) ** 2 + (delta_lon * np.cos(np.deg2rad(latitude))) ** 2
    return np.unravel_index(np.argmin(distance), distance.shape)


def metrics(energy, distance):
    total = energy.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("response has no finite, positive energy")
    descending = np.sort(energy.ravel())[::-1]
    footprint = int(np.searchsorted(np.cumsum(descending), .9 * total) + 1)
    return footprint, float((energy * distance).sum() / total)


def main():
    args = arguments(__doc__)
    checks = Checks()
    health(checks, args)
    try:
        payload, metadata = load_window(args.window)
    except Exception as exc:
        checks.check("load window", False, str(exc))
        return 1
    base = forecast(checks, args.url, payload, "base")
    repeat = forecast(checks, args.url, payload, "base repeat")
    checks.check("base determinism", base is not None and repeat is not None
                 and np.array_equal(base[0], repeat[0]))
    zero = forecast(checks, args.url, dict(payload, perturbation=np.zeros((6, 32, 36)).tolist()), "zero shock")
    checks.check("zero-array shock equals base (array_equal)", base is not None and zero is not None
                 and np.array_equal(base[0], zero[0]))
    rows, columns = np.indices((32, 36))
    radius = 4.5
    for label, channel, amplitude, latitude, longitude in (
        ("warm Bay of Bengal", 0, 2., 16., 88.),
        ("weaker Arabian Sea wind", 2, -3., 16., 70.),
    ):
        try:
            row, column = centre(metadata, latitude, longitude)
            distance = np.hypot(rows - row, columns - column)
            bump = np.where(distance < radius, .5 * (1. + np.cos(np.pi * distance / radius)), 0.)
            perturbation = np.zeros((6, 32, 36))
            perturbation[channel] = amplitude * bump
            shocked_payload = dict(payload, perturbation=perturbation.tolist())
            shock = forecast(checks, args.url, shocked_payload, label)
            again = forecast(checks, args.url, shocked_payload, label + " repeat")
            checks.check(label + ": determinism", shock is not None and again is not None
                         and np.array_equal(shock[0], again[0]))
            if base is None or shock is None:
                checks.check(label + ": response available", False)
                continue
            response = shock[0] - base[0]
            if not checks.check(label + ": response finite (no NaN/Inf)", np.isfinite(response).all()):
                continue
            energy = np.sum(response ** 2, axis=1)
            if not checks.check(label + ": nonzero step-0 response", energy[0].sum() > 0):
                continue
            local_fraction = energy[0][distance <= 2 * radius].sum() / energy[0].sum()
            checks.check(label + ": step-0 energy within 2R >= 80%", local_fraction >= .8,
                         f"{local_fraction:.2%}, centre=({row},{column}), R={radius}")
            first_footprint, first_mean = metrics(energy[0], distance)
            last_footprint, last_mean = metrics(energy[-1], distance)
            checks.check(label + ": footprint or mean distance grows",
                         last_footprint > first_footprint or last_mean > first_mean + 1e-9,
                         f"90% footprint {first_footprint}->{last_footprint}; "
                         f"mean distance {first_mean:.4f}->{last_mean:.4f} cells")
            peak_first = float(np.abs(response[0]).max())
            peak_last = float(np.abs(response[-1]).max())
            checks.check(label + ": last peak <= 3x first peak", peak_last <= 3 * peak_first,
                         f"{peak_first:.6g}->{peak_last:.6g}")
        except Exception as exc:
            checks.check(label + ": response checks", False, str(exc))
    return int(checks.failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
