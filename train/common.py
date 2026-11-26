"""Read Track 2 data, fit seasonal climatology, and select forecast windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from model.unet import CHANNELS, LAT, LON, RAIN, STATIC_CHANNELS

DEFAULT_VAL_YEARS = 2
LEADS = (1, 4, 8)
PERIOD = 365
HARMONICS = 3
FOLD_BUFFER_DAYS = 14
PROXY_ROLLOUT_STEPS = 14


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


def validation_start(dates, val_years=DEFAULT_VAL_YEARS):
    """Return the row index where the last N represented calendar years begin."""
    if val_years < 1:
        raise ValueError("--val-years must be at least one")
    years = dates.astype("datetime64[Y]")
    first_validation_year = years[-1] - np.timedelta64(val_years - 1, "Y")
    cut = int(np.searchsorted(years, first_validation_year))
    if cut == 0:
        raise ValueError("--val-years must leave at least one calendar year for training")
    return cut


def time_block_split(dates, start_year, end_year):
    """Return training slices and one full calendar-year validation block."""
    if not 1 <= start_year <= end_year <= 9998:
        raise ValueError("Fold years must be ordered and between 1 and 9998")
    first = np.datetime64(f"{start_year:04d}", "Y").astype("datetime64[D]")
    following = np.datetime64(f"{end_year + 1:04d}", "Y").astype("datetime64[D]")
    if dates[0] > first or dates[-1] < following - np.timedelta64(1, "D"):
        raise ValueError("The dataset must cover the entire requested fold")

    start, stop = map(int, np.searchsorted(dates, [first, following]))
    train_stop = max(0, start - FOLD_BUFFER_DAYS)
    train_start = min(len(dates), stop + FOLD_BUFFER_DAYS)
    train = tuple(part for part in (slice(0, train_stop), slice(train_start, len(dates)))
                  if part.start < part.stop)
    return train, slice(start, stop)


def recorded_proxy_cases(audit):
    """Return the original event selections in benchmark order."""
    return list(dict.fromkeys((row["event"], row["first_target"]) for row in audit["event_forecast"]["events"]))


def exclude_proxy_windows(dates, ends, steps, first_targets):
    """Discard loss windows sharing any input or forecast row with a proxy."""
    targets = np.asarray(first_targets, dtype="datetime64[D]")
    positions = np.searchsorted(dates, targets)
    in_range = positions < len(dates)
    positions = positions[in_range]
    positions = positions[dates[positions] == targets[in_range]]
    overlap = ((ends[:, None] - 2 <= positions + PROXY_ROLLOUT_STEPS - 1)
               & (ends[:, None] + steps >= positions - 3))
    return ends[~overlap.any(axis=1)]


def year_range(dates):
    first, last = dates[[0, -1]].astype("datetime64[Y]")
    return str(first) if first == last else f"{first}–{last}"


def window_ends(record_count, steps, split, *, val_start=None):
    """Keep all three input days and all targets inside the same partition."""
    if split == "all":
        return np.arange(2, record_count - steps)
    if val_start is None or not 0 < val_start < record_count:
        raise ValueError("A validation start within the dataset is required")
    if split == "train":
        return np.arange(2, val_start - steps)
    if split == "validation":
        return np.arange(val_start + 2, record_count - steps)
    raise ValueError(f"Unknown split {split!r}")
