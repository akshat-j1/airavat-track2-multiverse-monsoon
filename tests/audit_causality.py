"""Audit the official and random shock responses on reproducible windows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

import numpy as np
from monsoon_r1 import contract, scoring

from engine.engine import ENCODED_BOUNDS, Engine
from model.unet import RAIN
from tests.benchmark import ARCHIVE_SHA256, HOLDOUT_YEARS, SCRATCH, sha256
from tests.eval_skill import evaluate_shocks, probe_ends
from train.common import (
    day_phase, exclude_proxy_windows, load_proxy_events, load_weather, recorded_proxy_cases, validation_start,
    window_ends,
)


CHECKS = ("nonzero", "localized", "propagates", "stable", "no_teleport")
BOXES = {"warm_ocean": contract.BOX_BAY_OF_BENGAL, "monsoon_stall": contract.BOX_ARABIAN_SEA}
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)}
SEED = 42


def season(date):
    return next(name for name, months in SEASONS.items() if date.astype(object).month in months)


def choose_random_ends(weather, cut, excluded, rng):
    ends = window_ends(len(weather.dates), contract.MAX_STEPS, "validation", val_start=cut)
    ends = exclude_proxy_windows(weather.dates, ends, contract.MAX_STEPS, excluded)
    by_season = {name: [] for name in SEASONS}
    for end in ends:
        by_season[season(weather.dates[end])].append(end)
    return np.concatenate([rng.choice(by_season[name], count, replace=False)
                           for name, count in zip(SEASONS, (3, 2, 3, 2))])


def summarize_checks(rows):
    near = [row["near_frac_day1"] for row in rows if "near_frac_day1" in row]
    return {
        "n": len(rows),
        **{key: sum(row.get(key, False) for row in rows) for key in CHECKS},
        "near_frac_day1_mean": float(np.mean(near)) if near else None,
        "near_frac_day1_min": min(near) if near else None,
    }


def audit_official(engine, weather, cut, stats_path, ends, proxies, std, wind_threshold):
    dates = [str(weather.dates[end + 1]) for end in ends]
    results = evaluate_shocks(engine, weather, stats_path, 0, cut, dates, audit_null=True)
    arabian = contract.box_mask(contract.BOX_ARABIAN_SEA)
    for end, result in zip(ends, results):
        winds = weather.physical[end, 2:4][:, arabian]
        speed = float(np.hypot(*winds).mean())
        result.update(season=season(weather.dates[end]), proxy=proxies.get(end),
                      arabian_wind_mps=speed, weak_wind=speed <= wind_threshold)
    return results


def judge_shock(history, std, rng):
    _, channels, height, width = history.shape
    channel = int(rng.integers(channels))
    row, column = int(rng.integers(height)), int(rng.integers(width))
    radius = float(rng.uniform(2, 8))
    magnitude = float(rng.uniform(0.5, 3))
    sign = int(rng.choice((-1, 1)))
    y, x = np.indices((height, width))
    distance = np.hypot(y - row, x - column)
    support = distance < radius
    bump = np.where(support, 0.5 * (1 + np.cos(np.pi * distance / radius)), 0)
    if channel in (RAIN, contract.CHANNELS.index("tcwv")) and np.any(
            history[-1, channel] + sign * magnitude * std[channel] * bump < 0):
        sign = 1
    delta = np.zeros_like(history[-1])
    delta[channel] = sign * magnitude * std[channel] * bump
    rows, cols = np.where(support)
    box = (contract.LAT[rows.min()], contract.LAT[rows.max()], contract.LON[cols.min()], contract.LON[cols.max()])
    return delta, box, {"channel": contract.CHANNELS[channel], "cell": [row, column],
                        "radius_cells": radius, "signed_std": sign * magnitude}


def audit_judge(engine, weather, ends, std, rng):
    reports = []
    for end in ends:
        history = weather.physical[end - 2:end + 1]
        phase = day_phase(weather.day_of_year[end])
        delta, box, report = judge_shock(history, std, rng)
        base = engine.simulate(history, phase, contract.SCORE_STEPS)
        shocked = engine.simulate(history, phase, contract.SCORE_STEPS, delta)
        validity = scoring.validity(shocked, contract.SCORE_STEPS)
        checks = scoring.causality_checks(base, shocked, box, std)
        reports.append({**report, **checks, "last_input_day": str(weather.dates[end]),
                        "validity_errors": validity, "nonnegative_rain": bool((shocked[:, RAIN] >= 0).all()),
                        "pass": not validity and bool((shocked[:, RAIN] >= 0).all())
                        and all(checks.get(name, False) for name in ("finite", "localized", "propagates"))})
    return reports


def audit_rollouts(engine, weather, proxies, std):
    limits = np.asarray(ENCODED_BOUNDS, dtype=np.float32)
    limits[RAIN, 1] = np.expm1(limits[RAIN, 1]).round()
    latitude = contract.lat_weights()[None, None, :, None]
    reports = []
    for end, event in proxies.items():
        history = weather.physical[end - 2:end + 1]
        phase = day_phase(weather.day_of_year[end])
        observations = weather.physical[end + 1:end + 1 + contract.MAX_STEPS]
        for shock in ("none", "null", *BOXES):
            delta = None if shock == "none" else scoring.make_shock(shock, history[-1], std)
            rollout = engine.simulate(history, phase, contract.MAX_STEPS, delta)
            lower_hits = (rollout == limits[None, :, 0, None, None]).sum(axis=(0, 2, 3))
            upper_hits = (rollout == limits[None, :, 1, None, None]).sum(axis=(0, 2, 3))
            mean_bias = (((rollout - observations) * latitude).mean(axis=(-2, -1)) / std[None, :])
            drift = np.sqrt(np.mean(mean_bias ** 2, axis=1))
            reports.append({
                "event": event, "shock": shock, "validity_errors": scoring.validity(rollout, contract.MAX_STEPS),
                "lower_bound_hits": lower_hits.tolist(), "upper_bound_hits": upper_hits.tolist(),
                "drift_by_lead_std": drift.tolist(),
                "early_drift": float(drift[:4].mean()), "late_drift": float(drift[-4:].mean()),
                "drift_grew": bool(drift[-4:].mean() > drift[:4].mean()),
            })
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=SCRATCH / "india_climate_train.nc")
    args = parser.parse_args()
    if sha256(args.dataset) != ARCHIVE_SHA256:
        parser.error("Use the original organizer archive")
    weather = load_weather(args.dataset)
    cut = validation_start(weather.dates, HOLDOUT_YEARS)
    fixture = load_proxy_events()
    events = recorded_proxy_cases(fixture)
    proxy_dates = [date for _, date in events]
    proxy_ends = probe_ends(weather, proxy_dates, cut, contract.MAX_STEPS)
    proxies = {end: name for end, (name, _) in zip(proxy_ends, events)}
    rng = np.random.default_rng(SEED)
    shock_dates = fixture["shock_first_target_dates"]
    random_ends = choose_random_ends(weather, cut, [*proxy_dates, *shock_dates], rng)
    stats_path = ROOT / "assets/stats.json"
    std = np.asarray(json.loads(stats_path.read_text(encoding="utf-8"))["std"], dtype=np.float32)
    arabian = contract.box_mask(contract.BOX_ARABIAN_SEA)
    winds = weather.physical[cut:, 2:4][:, :, arabian]
    wind_threshold = float(np.quantile(np.hypot(winds[:, 0], winds[:, 1]).mean(axis=1), .25))
    engine = Engine(args.weights)
    anchors = probe_ends(weather, shock_dates, cut, contract.SCORE_STEPS)
    official = audit_official(engine, weather, cut, stats_path, anchors, proxies, std, wind_threshold)
    expanded = audit_official(engine, weather, cut, stats_path, np.r_[proxy_ends, random_ends], proxies, std, wind_threshold)
    judge = audit_judge(engine, weather, random_ends, std, rng)
    rollouts = audit_rollouts(engine, weather, proxies, std)
    weak = {row["last_input_day"]: row["shocks"]["monsoon_stall"]
            for row in (*official, *expanded) if row["weak_wind"]}
    summary = {
        "official_propagation": sum(r["shocks"][name]["propagates"] for r in official for name in BOXES),
        "official_exact_zero": sum(r["zero_shock_exact"] for r in official),
        "expanded_exact_zero": sum(r["zero_shock_exact"] for r in expanded),
        "expanded": {name: summarize_checks([row["shocks"][name] for row in expanded]) for name in BOXES},
        "expanded_null": summarize_checks([row["null_checks"] for row in expanded]),
        "stall_by_season": {
            name: summarize_checks([row["shocks"]["monsoon_stall"] for row in expanded if row["season"] == name])
            for name in SEASONS
        },
        "weak_wind_stall": summarize_checks(list(weak.values())),
        "judge": summarize_checks(judge),
        "judge_failures": sum(not row["pass"] for row in judge),
        "invalid_14_step_rollouts": sum(bool(row["validity_errors"]) for row in rollouts),
        "saturated_14_step_rollouts": sum(any(row["upper_bound_hits"]) or any(row["lower_bound_hits"]) for row in rollouts),
        "growing_14_step_drift": sum(row["drift_grew"] for row in rollouts),
    }
    result = {"seed": SEED, "weak_wind_threshold_mps": wind_threshold, "summary": summary,
              "official": official, "expanded": expanded, "judge": judge, "rollouts": rollouts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
