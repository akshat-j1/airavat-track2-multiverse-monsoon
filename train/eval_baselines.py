"""Evaluate 2005–2019 climatology and persistence on daily 2020–2021 windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from model.unet import LAT, RAIN
from train.common import LEADS, TRAIN_END, fit_climatology, load_weather, seasonal_fields, window_ends


def validation_cases(weather, stride=1):
    ends = window_ends(weather.dates, max(LEADS), "validation")[::stride]
    targets = ends[:, None] + np.asarray(LEADS)
    train = weather.dates < TRAIN_END
    coefficients = fit_climatology(weather.physical[train], weather.day_of_year[train])
    climatology = seasonal_fields(coefficients)[weather.day_of_year[targets] % 365]
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
    args = parser.parse_args()
    ends, truth, persistence, climatology = validation_cases(load_weather(args.dataset))
    print(json.dumps({
        "windows": len(ends), "leads": LEADS,
        "persistence": forecast_metrics(persistence, truth),
        "climatology": forecast_metrics(climatology, truth),
    }, indent=2))


if __name__ == "__main__":
    main()
