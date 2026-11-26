"""Evaluate training-only climatology and persistence on held-out daily windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from model.unet import LAT, RAIN
from train.common import (
    DEFAULT_VAL_YEARS, LEADS, PERIOD, fit_climatology, load_weather,
    seasonal_fields, validation_start, window_ends, year_range,
)


def validation_cases(weather, val_start, stride=1):
    ends = window_ends(len(weather.dates), max(LEADS), "validation", val_start=val_start)[::stride]
    if not len(ends):
        raise ValueError("No complete forecast windows in the validation period")
    targets = ends[:, None] + np.asarray(LEADS)
    coefficients = fit_climatology(weather.physical[:val_start], weather.day_of_year[:val_start])
    climatology = seasonal_fields(coefficients)[weather.day_of_year[targets] % PERIOD]
    climatology[:, :, RAIN] = np.maximum(climatology[:, :, RAIN], 0)
    persistence = np.broadcast_to(weather.physical[ends, None], climatology.shape)
    return ends, weather.physical[targets], persistence, climatology


def forecast_metrics(prediction, target):
    weight = np.cos(np.deg2rad(LAT))
    weight /= weight.mean()
    mse = (np.square(prediction - target, dtype="f4") * weight[None, None, None, :, None]).mean((0, 3, 4))
    predicted_event = prediction[:, :, RAIN] > 10
    true_event = target[:, :, RAIN] > 10
    hits = np.count_nonzero(predicted_event & true_event, axis=(0, 2, 3))
    union = np.count_nonzero(predicted_event | true_event, axis=(0, 2, 3))
    csi = np.divide(hits, union, out=np.full(len(LEADS), np.nan), where=union != 0)
    return {"rmse": np.sqrt(mse).tolist(), "heavy_rain_csi": csi.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS)
    args = parser.parse_args()
    weather = load_weather(args.dataset)
    try:
        val_start = validation_start(weather.dates, args.val_years)
        ends, truth, persistence, climatology = validation_cases(weather, val_start)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "training_years": year_range(weather.dates[:val_start]),
        "validation_years": year_range(weather.dates[val_start:]),
        "windows": len(ends), "leads": LEADS,
        "persistence": forecast_metrics(persistence, truth),
        "climatology": forecast_metrics(climatology, truth),
    }, indent=2))


if __name__ == "__main__":
    main()
