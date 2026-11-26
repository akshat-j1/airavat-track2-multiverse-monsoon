"""Report the Track 2 NetCDF schema and data quality before preprocessing."""

import sys
from os.path import dirname, realpath

# Direct invocation puts this file's directory ahead of the standard library,
# where dependencies expect to find the standard library's inspect module.
if __name__ == "__main__" and not __package__:
    if sys.path and realpath(sys.path[0]) == dirname(realpath(__file__)):
        sys.path.pop(0)

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to india_climate_train.nc")
    parser.add_argument(
        "--stats",
        type=Path,
        help="Path to stats.json (defaults to stats.json beside the dataset)",
    )
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")
    if args.stats is not None and not args.stats.is_file():
        parser.error(f"stats file not found: {args.stats}")

    import numpy as np
    import xarray as xr

    with xr.open_dataset(args.dataset) as data:
        print(f"Dimensions: {dict(data.sizes)}")
        for name, coordinate in data.coords.items():
            units = coordinate.attrs.get("units", "")
            print(f"Coordinate {name}: {coordinate.dims} {coordinate.shape} {units}")
            if coordinate.ndim == 1 and coordinate.size:
                values = coordinate.values
                print(f"  first={values[0]}, last={values[-1]}")
                if np.issubdtype(values.dtype, np.number):
                    print(f"  min={np.nanmin(values)}, max={np.nanmax(values)}")

        for name, variable in data.data_vars.items():
            print(
                f"Variable {name}: {variable.dims} {variable.shape} "
                f"{variable.dtype}, units={variable.attrs.get('units', 'unspecified')}"
            )
            print(f"  NaN or missing values: {variable.isnull().sum().item()}")
            if np.issubdtype(variable.dtype, np.number):
                print(f"  Infinite values: {np.isinf(variable).sum().item()}")

    stats_path = args.stats or args.dataset.with_name("stats.json")
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as stats_file:
            stats = json.load(stats_file)
        if not isinstance(stats, dict):
            raise ValueError(f"Expected a JSON object in {stats_path}")
        print(f"stats.json keys: {', '.join(sorted(stats))}")
    else:
        print(f"stats.json unavailable at {stats_path}")


if __name__ == "__main__":
    main()
