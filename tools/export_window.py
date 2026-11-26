"""Export three consecutive observations from an official Track 2 dataset."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def export_window(source, output, last_index):
    with xr.open_dataset(source) as dataset:
        if not 2 <= last_index < dataset.sizes["time"]:
            raise ValueError("--last-index must be a zero-based index with two preceding days")
        window = dataset.isel(time=slice(last_index - 2, last_index + 1))
        dates = window.time.values.astype("datetime64[D]")
        if not np.all(np.diff(dates) == np.timedelta64(1, "D")):
            raise ValueError("The selected observations must be consecutive daily records")
        channels = ["t2m", "tp", "u10", "v10", "msl", "tcwv"]
        data = window["data"].sel(channel=channels).transpose("time", "channel", "lat", "lon")
        lat = data.lat.values
        lon = data.lon.values
        if not np.array_equal(lat, np.arange(6.5, 38.5)) or not np.array_equal(lon, np.arange(65.5, 101.5)):
            raise ValueError("The dataset axes must match the Track 2 contract grid")
        values = data.values.astype(np.float32)
        if values.shape != (3, 6, 32, 36) or not np.isfinite(values).all() or np.any(values[:, 1] < 0):
            raise ValueError("The selected window must be finite with contract shape and nonnegative rain")
        day = int(window.time.dt.dayofyear.values[-1])
        angle = 2 * np.pi * day / 365
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, init_states=values, doy_sincos=[np.sin(angle), np.cos(angle)], lat=lat, lon=lon)
    print(f"Exported observations ending {dates[-1]} (day {day}) to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-index", type=int, required=True, help="Zero-based index of the last input day")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    try:
        export_window(args.input, args.output, args.last_index)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
