"""Measure held-out forecast skill and official counterfactual diagnostics."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from engine.engine import Engine
from model.unet import CHANNELS, STATE_SHAPE
from train.common import DEFAULT_VAL_YEARS, LEADS, day_phase, load_weather, validation_start, window_ends, year_range
from train.eval_baselines import forecast_metrics, validation_cases


def evaluate_forecasts(engine, weather, val_start, stride):
    ends, target, persistence, climatology = validation_cases(weather, val_start, stride)
    histories = ends[:, None] + np.arange(-2, 1)
    phases = day_phase(weather.day_of_year[ends])
    predictions = []
    for offset in range(0, len(ends), 24):
        batch = slice(offset, offset + 24)
        rollout = engine._rollout(
            torch.from_numpy(weather.physical[histories[batch]]),
            torch.from_numpy(phases[batch]), max(LEADS),
        )
        predictions.append(rollout[:, np.asarray(LEADS) - 1].numpy())
    prediction = np.concatenate(predictions)
    scores = {
        name: forecast_metrics(values, target)
        for name, values in (("model", prediction), ("persistence", persistence), ("climatology", climatology))
    }
    model_rmse = np.asarray(scores["model"]["rmse"])
    persistence_rmse = np.asarray(scores["persistence"]["rmse"])
    climatology_rmse = np.asarray(scores["climatology"]["rmse"])
    ratios = prediction[:, -1].var(axis=(-2, -1)) / np.maximum(target[:, -1].var(axis=(-2, -1)), 1e-8)
    return {
        "years": year_range(weather.dates[val_start:]),
        "windows": len(ends), "lead_days": LEADS, "channels": CHANNELS, **scores,
        "mean_skill_vs_persistence": (1 - model_rmse / persistence_rmse).mean(axis=1).tolist(),
        "mean_skill_vs_best_reference": (1 - model_rmse / np.minimum(persistence_rmse, climatology_rmse)).mean(axis=1).tolist(),
        "day8_spatial_variance_ratio": ratios.mean(axis=0).tolist(),
    }


def evaluate_shocks(engine, weather, stats_path, samples, val_start):
    from monsoon_r1 import contract, scoring

    with stats_path.open(encoding="utf-8") as file:
        stats = json.load(file)
    if tuple(stats["channels"]) != CHANNELS:
        raise ValueError("Unexpected stats.json channel order")
    std = np.asarray(stats["std"], dtype="f4")
    ends = window_ends(len(weather.dates), contract.SCORE_STEPS, "validation", val_start=val_start)
    if not len(ends):
        raise ValueError("No complete shock windows in the validation period")
    ends = ends[np.linspace(0, len(ends) - 1, min(samples, len(ends)), dtype=int)]
    zeros = np.zeros(STATE_SHAPE, dtype="f4")
    results = []
    boxes = {"warm_ocean": contract.BOX_BAY_OF_BENGAL, "monsoon_stall": contract.BOX_ARABIAN_SEA}
    for end in ends:
        history = weather.physical[end - 2:end + 1]
        phase = day_phase(weather.day_of_year[end])
        start = time.perf_counter()
        base = engine.simulate(history, phase, contract.SCORE_STEPS)
        seconds = time.perf_counter() - start
        null = engine.simulate(history, phase, contract.SCORE_STEPS, zeros)
        result = {"last_input_day": str(weather.dates[end]), "base_seconds": seconds,
                  "base_valid": not scoring.validity(base, contract.SCORE_STEPS),
                  "zero_shock_exact": bool(np.array_equal(base, null)), "shocks": {}}
        for name, box in boxes.items():
            delta = scoring.make_shock(name, history[-1], std)
            rollout = engine.simulate(history, phase, contract.SCORE_STEPS, delta)
            checks = scoring.causality_checks(base, rollout, box, std)
            result["shocks"][name] = {
                **checks, "valid": not scoring.validity(rollout, contract.SCORE_STEPS),
                "peak_response_in_channel_std": float(np.abs((rollout - base) / std[:, None, None]).max()),
            }
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shock-samples", type=int, default=0)
    parser.add_argument("--shocks-only", action="store_true")
    args = parser.parse_args()
    if args.stride < 1 or args.shock_samples < 0 or (args.shocks_only and not args.shock_samples):
        parser.error("Use a positive stride and request shock samples with --shocks-only")
    weather = load_weather(args.dataset)
    try:
        val_start = validation_start(weather.dates, args.val_years)
    except ValueError as exc:
        parser.error(str(exc))
    engine = Engine(args.weights)
    result = {}
    if not args.shocks_only:
        checkpoint = torch.load(args.weights / "airavat.pt", map_location="cpu", weights_only=True)
        if checkpoint["split"] != "train":
            parser.error("Use a --split train checkpoint for held-out forecast skill")
        fit_period = [str(weather.dates[0]), str(weather.dates[val_start - 1])]
        if checkpoint.get("fit_period", fit_period) != fit_period:
            parser.error("The checkpoint's fitting period does not match this dataset and --val-years")
        if "fit_period" not in checkpoint and args.val_years != DEFAULT_VAL_YEARS:
            parser.error("A checkpoint with split metadata is required to override --val-years")
        result["forecast"] = evaluate_forecasts(engine, weather, val_start, args.stride)
    if args.shock_samples:
        stats = args.stats or args.dataset.with_name("stats.json")
        result["shocks"] = evaluate_shocks(engine, weather, stats, args.shock_samples, val_start)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
