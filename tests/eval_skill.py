"""Measure held-out forecast skill and official counterfactual diagnostics."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from engine.engine import Engine
from model.unet import CHANNELS
from train.common import (
    DEFAULT_VAL_YEARS, LEADS, PERIOD, PROXY_ROLLOUT_STEPS, day_phase, load_weather,
    validation_start, window_ends, year_range,
)
from train.eval_baselines import forecast_metrics, validation_cases


def probe_ends(weather, dates, val_start, horizon):
    """Resolve externally selected first-target dates to complete held-out windows."""
    targets = np.asarray(dates, dtype="datetime64[D]")
    indices = np.searchsorted(weather.dates, targets)
    for target, index in zip(targets, indices):
        if index < val_start + 3 or index + horizon > len(weather.dates) or weather.dates[index] != target:
            raise ValueError(f"{target} does not start a complete {horizon}-day held-out forecast")
    return indices - 1


def evaluate_events(engine, weather, val_start, events, *, climatology_for=None):
    from monsoon_r1 import scoring

    labels, dates = zip(*(item.split("=", 1) for item in events))
    indices = probe_ends(weather, dates, val_start, PROXY_ROLLOUT_STEPS)
    if climatology_for is None:
        train_days = weather.day_of_year[:val_start]

        def climatology_for(index):
            day = weather.day_of_year[index]
            neighbor_days = abs((train_days - day + PERIOD // 2) % PERIOD - PERIOD // 2) <= 7
            return weather.physical[:val_start][neighbor_days].mean(axis=0)

    details, skill_values, rain_skill_values, ratios = [], [], [], []
    for name, date, end in zip(labels, dates, indices):
        initial = weather.physical[end - 2:end + 1]
        rollout = engine.simulate(initial, day_phase(weather.day_of_year[end]), PROXY_ROLLOUT_STEPS)
        for lead in LEADS:
            climatology = climatology_for(end + lead)
            truth = weather.physical[end + lead]
            prediction, persistence = rollout[lead - 1], initial[-1]
            energy = scoring.skill_vs_best_ref(prediction, truth, persistence, climatology)
            csis = [scoring.csi(fields[1], truth[1]) for fields in (prediction, persistence, climatology)]
            rain_skill_values.append(scoring.csi_skill(*csis))
            skill_values.append(float(np.nanmean(energy["ss"])))
            if lead == max(LEADS):
                ratios.append(scoring.variance_ratio(prediction, truth))
            details.append({
                "event": name, "first_target": date, "lead": lead,
                "mean_skill_vs_best": skill_values[-1],
                "mean_skill_vs_persistence": float(np.nanmean(energy["ss_persist"])),
                "rmse": {key: scoring.rmse_lat(value, truth).tolist() for key, value in
                         (("model", prediction), ("persistence", persistence), ("climatology", climatology))},
                "csi": [value if np.isfinite(value) else None for value in csis],
                "peak_rain_model": float(prediction[1].max()), "peak_rain_observed": float(truth[1].max()),
                "heavy_cells_model": int((prediction[1] > scoring.RAIN_THRESH).sum()),
                "heavy_cells_observed": int((truth[1] > scoring.RAIN_THRESH).sum()),
                "valid_14_day_rollout": not scoring.validity(rollout, PROXY_ROLLOUT_STEPS),
            })
    return {
        **scoring.forecast_points(skill_values, rain_skill_values), "events": details,
        "mean_skill_vs_best_by_lead": np.asarray(skill_values).reshape(-1, len(LEADS)).mean(axis=0).tolist(),
        "day8_variance_ratio": np.mean(ratios, axis=0).tolist(),
        "anti_blur_points": scoring.anti_blur_points(np.asarray(ratios)),
    }


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


def evaluate_shocks(engine, weather, stats_path, samples, val_start, dates=()):
    from monsoon_r1 import contract, scoring

    with stats_path.open(encoding="utf-8") as file:
        stats = json.load(file)
    if tuple(stats["channels"]) != CHANNELS:
        raise ValueError("Unexpected stats.json channel order")
    std = np.asarray(stats["std"], dtype="f4")
    ends = window_ends(len(weather.dates), contract.SCORE_STEPS, "validation", val_start=val_start)
    if not len(ends):
        raise ValueError("No complete shock windows in the validation period")
    if dates:
        ends = probe_ends(weather, dates, val_start, 10)
    else:
        ends = ends[np.linspace(0, len(ends) - 1, min(samples, len(ends)), dtype=int)]
    results = []
    boxes = {"warm_ocean": contract.BOX_BAY_OF_BENGAL, "monsoon_stall": contract.BOX_ARABIAN_SEA}
    for end in ends:
        history = weather.physical[end - 2:end + 1]
        phase = day_phase(weather.day_of_year[end])
        start = time.perf_counter()
        base = engine.simulate(history, phase, contract.SCORE_STEPS)
        seconds = time.perf_counter() - start
        delta = scoring.make_shock("null", history[-1], std)
        null = engine.simulate(history, phase, contract.SCORE_STEPS, delta)
        tiny = delta.copy()
        tiny[0, 0, 0] = 1e-30
        tiny_result = engine.simulate(history, phase, contract.SCORE_STEPS, tiny)
        result = {"last_input_day": str(weather.dates[end]), "base_seconds": seconds,
                  "base_valid": not scoring.validity(base, contract.SCORE_STEPS),
                  "null_consistent": bool(scoring.null_consistent(base, null, std)),
                  "zero_shock_exact": bool(np.array_equal(base, null)),
                  "subprecision_shock_exact": bool(np.array_equal(base, tiny_result)), "shocks": {}}
        for name, box in boxes.items():
            delta = scoring.make_shock(name, history[-1], std)
            start = time.perf_counter()
            rollout = engine.simulate(history, phase, contract.SCORE_STEPS, delta)
            elapsed = time.perf_counter() - start
            checks = scoring.causality_checks(base, rollout, box, std)
            result["shocks"][name] = {
                **checks, "valid": not scoring.validity(rollout, contract.SCORE_STEPS),
                "seconds": elapsed,
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
    parser.add_argument("--event", action="append", default=[], metavar="LABEL=FIRST_TARGET_DAY")
    parser.add_argument("--shock-target", action="append", default=[], metavar="FIRST_TARGET_DAY")
    parser.add_argument("--shocks-only", action="store_true")
    args = parser.parse_args()
    if args.stride < 1 or args.shock_samples < 0 or (args.shocks_only and not (args.shock_samples or args.shock_target)):
        parser.error("Use a positive stride and request shock samples with --shocks-only")
    if args.event and (args.shocks_only or any("=" not in event for event in args.event)):
        parser.error("--event requires a train-only forecast evaluation and LABEL=DATE arguments")
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
        if args.event:
            result["event_forecast"] = evaluate_events(engine, weather, val_start, args.event)
    if args.shock_samples or args.shock_target:
        stats = args.stats or args.dataset.with_name("stats.json")
        result["shocks"] = evaluate_shocks(engine, weather, stats, args.shock_samples, val_start, args.shock_target)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
