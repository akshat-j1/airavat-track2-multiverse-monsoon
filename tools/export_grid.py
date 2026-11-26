#!/usr/bin/env python3
"""Export static NetCDF grid fields for the dashboard.

Run from any directory: python tools/export_grid.py
Override paths with --input FILE.nc --output FILE.json.
Non-spatial dimensions (for example time or level) use their first index.
"""

import argparse
import json
from pathlib import Path
import sys

import xarray as xr


ROOT = Path(__file__).resolve().parents[1]


def find_coordinate(dataset, kind):
    """Recognize common coordinate names and CF coordinate metadata."""
    names = ("lat", "latitude", "nav_lat") if kind == "lat" else ("lon", "longitude", "nav_lon")
    standard_name = "latitude" if kind == "lat" else "longitude"
    units = ("degrees_north", "degree_north") if kind == "lat" else ("degrees_east", "degree_east")
    # Prefer registered coordinates, but accept coordinate arrays stored as variables.
    candidates = list(dataset.coords) + [name for name in dataset.variables if name not in dataset.coords]
    for alias in names:
        for name in candidates:
            if name.lower() == alias:
                return dataset[name]
    for name in candidates:
        variable = dataset[name]
        if (str(variable.attrs.get("standard_name", "")).lower() == standard_name
                or str(variable.attrs.get("units", "")).lower() in units):
            return variable
    return None


def export_grid(input_path, output_path):
    with xr.open_dataset(input_path) as dataset:
        try:
            lat = find_coordinate(dataset, "lat")
            lon = find_coordinate(dataset, "lon")
            missing = []
            if lat is None:
                missing.append("latitude coordinate")
            if lon is None:
                missing.append("longitude coordinate")
            missing.extend(name for name in ("land_sea_mask", "orography_m") if name not in dataset)
            if missing:
                raise ValueError("Missing: " + ", ".join(missing))
            if lat.ndim not in (1, 2) or lon.ndim != lat.ndim:
                raise ValueError("lat/lon must both be 1D axes or 2D coordinate grids")
            if lat.ndim == 2 and set(lat.dims) != set(lon.dims):
                raise ValueError("2D lat/lon must share spatial dimensions")
            if lat.ndim == 1 and lat.dims == lon.dims:
                raise ValueError("1D lat/lon must use distinct spatial dimensions")
            spatial_dims = tuple(dict.fromkeys(lat.dims + lon.dims))
            if lat.ndim == 2:
                lon = lon.transpose(*spatial_dims)
            result = {"lat": lat.values.tolist(), "lon": lon.values.tolist()}
            for name in ("land_sea_mask", "orography_m"):
                field = dataset[name]
                if not set(spatial_dims).issubset(field.dims):
                    raise ValueError(f"{name} must include spatial dimensions {spatial_dims}; got {field.dims}")
                field = field.isel({dim: 0 for dim in field.dims if dim not in spatial_dims}, drop=True)
                result[name] = field.transpose(*spatial_dims).values.tolist()
            # Reject NaN/Inf rather than writing invalid JSON for the dashboard.
            content = json.dumps(result, allow_nan=False, separators=(",", ":"))
        except (ValueError, TypeError, IndexError) as exc:
            print(str(exc), file=sys.stderr)
            print("Available variables: " + ", ".join(dataset.variables), file=sys.stderr)
            return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "india_climate_train.nc")
    parser.add_argument("--output", type=Path, default=ROOT / "dashboard" / "grid.json")
    args = parser.parse_args()
    try:
        return export_grid(args.input, args.output)
    except (OSError, ValueError, ImportError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
