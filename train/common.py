"""Read Track 2 data, fit seasonal climatology, and select forecast windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from model.unet import CHANNELS, LAT, LON, RAIN, STATIC_CHANNELS

TRAIN_END = np.datetime64("2020-01-01")
VALIDATION_END = np.datetime64("2022-01-01")
LEADS = (1, 4, 8)
PERIOD = 365
HARMONICS = 3


@dataclass(frozen=True)
class Weather:
    physical: np.ndarray
    dates: np.ndarray
    day_of_year: np.ndarray
    static: np.ndarray


def load_weather(path: Path) -> Weather:
    with xr.open_dataset(path) as ds:
        np.testing.assert_array_equal(ds.channel.values, CHANNELS)
        np.testing.assert_array_equal(ds.static_channel.values, STATIC_CHANNELS)
        np.testing.assert_array_equal(ds.lat.values, LAT)
        np.testing.assert_array_equal(ds.lon.values, LON)
        dates = ds.time.values.astype("datetime64[D]")
        if len(dates) < 4 or np.any(np.diff(dates) != np.timedelta64(1, "D")):
            raise ValueError("The dataset must have consecutive daily records")
        physical = ds.data.transpose("time", "channel", "lat", "lon").values
        static = ds.static.transpose("static_channel", "lat", "lon").values
        doy = ds.time.dt.dayofyear.values
    if not np.isfinite(physical).all() or not np.isfinite(static).all():
        raise ValueError("Non-finite values in the dataset")
    if np.any(physical[:, RAIN] < 0):
        raise ValueError("The dataset contains negative daily precipitation")
    return Weather(physical.astype("f4", copy=False), dates, doy, static.astype("f4", copy=False))


def day_phase(day_of_year):
    angle = 2 * np.pi * np.asarray(day_of_year) / PERIOD
    return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype("f4")


def seasonal_basis(day_of_year):
    terms = [np.ones_like(day_of_year, dtype="f4")]
    for harmonic in range(1, HARMONICS + 1):
        phase = day_phase(harmonic * day_of_year)
        terms.extend((phase[..., 0], phase[..., 1]))
    return np.stack(terms, axis=-1)


def fit_climatology(fields, day_of_year):
    """Fit the intercept and three annual harmonics at every grid cell."""
    design = seasonal_basis(day_of_year)
    flat = fields.reshape(len(fields), -1)
    coefficients = np.linalg.solve(design.T @ design, design.T @ flat)
    return coefficients.reshape((1 + 2 * HARMONICS, *fields.shape[1:]))


def seasonal_fields(coefficients):
    """One cycle indexed by day_of_year modulo 365, including leap dates."""
    basis = seasonal_basis(np.arange(PERIOD))
    flat = basis @ coefficients.reshape(len(coefficients), -1)
    return flat.reshape(PERIOD, *coefficients.shape[1:])


def window_ends(dates, steps, split):
    last_input = np.arange(2, len(dates) - steps)
    first_day = dates[last_input - 2]
    last_day = dates[last_input + steps]
    if split == "train":
        return last_input[last_day < TRAIN_END]
    if split == "validation":
        return last_input[(first_day >= TRAIN_END) & (last_day < VALIDATION_END)]
    if split == "all":
        return last_input[last_day < VALIDATION_END]
    raise ValueError(f"Unknown split {split!r}")
