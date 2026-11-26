"""Fit post-rollout anomaly shrinkage on non-proxy held-out windows."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import torch

from engine.engine import CHECKPOINT_NAME, MAX_STEPS, Engine, checkpoint_paths
from model.unet import CHANNELS, LAT, RAIN, encode_physical
from train.common import (
    DEFAULT_VAL_YEARS, day_phase, exclude_proxy_windows, load_weather, recorded_proxy_cases,
    validation_start, window_ends,
)


def check_training_scope(paths, first_validation_day):
    for path in paths:
        package = torch.load(path, map_location="cpu", weights_only=True)
        if "shrinkage_alpha" in package:
            raise ValueError(f"{path} already contains shrinkage")
        training = package.get("fit_periods") or [package.get("fit_period")]
        climate = package.get("climatology_fit_periods") or training
        for period in (*training, *climate):
            if not period or not np.datetime64(period[-1], "D") < first_validation_day:
                raise ValueError(f"{path} must document training and climatology before the holdout")


def fit_alpha(engine, weather, ends):
    latitude = torch.from_numpy(np.cos(np.deg2rad(LAT))).view(1, 1, 1, -1, 1)
    cross = torch.zeros((MAX_STEPS, len(CHANNELS)), dtype=torch.float64)
    energy = torch.zeros_like(cross)
    for batch in np.array_split(ends, (len(ends) + 23) // 24):
        history = torch.from_numpy(weather.physical[batch[:, None] + np.arange(-2, 1)])
        phases = day_phase(weather.day_of_year[batch])
        phases /= np.linalg.norm(phases, axis=1, keepdims=True)
        forecasts, climate = engine._encoded_forecast(history, torch.from_numpy(phases), MAX_STEPS)
        forecast = torch.stack(forecasts, dim=1) - climate
        truth = encode_physical(torch.from_numpy(weather.physical[batch[:, None] + np.arange(1, MAX_STEPS + 1)]))
        observed = truth - climate
        cross += (forecast * observed * latitude).sum(dim=(0, 3, 4), dtype=torch.float64)
        energy += (forecast.square() * latitude).sum(dim=(0, 3, 4), dtype=torch.float64)
    alpha = torch.ones_like(cross)
    has_energy = energy > 0
    alpha[has_energy] = (cross[has_energy] / energy[has_energy]).clamp(0, 1)
    # Preserve E4 rain peaks exactly. Fit the other channels against their stored harmonic climate.
    alpha[:, RAIN] = 1
    if not torch.isfinite(alpha).all():
        raise ValueError("The fitted shrinkage contains non-finite values")
    return alpha.float()


def save_bundle(paths, output, alpha, windows):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="shrinkage-", dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        primary = torch.load(paths[0], map_location="cpu", weights_only=True)
        primary["shrinkage_alpha"] = alpha
        primary["shrinkage_validation_windows"] = windows
        torch.save(primary, bundle / CHECKPOINT_NAME)
        for path in paths[1:]:
            shutil.copyfile(path, bundle / path.name)
        bundle.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    paths = checkpoint_paths(args.weights)
    try:
        weather = load_weather(args.dataset)
        cut = validation_start(weather.dates, args.val_years)
        check_training_scope(paths, weather.dates[cut])
        audit = json.loads(Path(__file__).with_name("validation.json").read_text(encoding="utf-8"))
        targets = [date for _, date in recorded_proxy_cases(audit["release_candidate_audit"])]
        ends = window_ends(len(weather.dates), MAX_STEPS, "validation", val_start=cut)
        ends = exclude_proxy_windows(weather.dates, ends, MAX_STEPS, targets)
        if not len(ends):
            raise ValueError("No complete non-proxy validation windows remain")
        alpha = fit_alpha(Engine(args.weights), weather, ends)
        save_bundle(paths, args.output, alpha, len(ends))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"members": len(paths), "validation_windows": len(ends),
                      "channels": CHANNELS, "alpha": alpha.tolist()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
