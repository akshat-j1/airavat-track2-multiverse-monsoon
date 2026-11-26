"""Score a checkpoint on the archived Airavat proxy and shock protocol."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from monsoon_r1 import contract, scoring, stats

from engine.engine import Engine, checkpoint_paths
from tests.eval_skill import evaluate_events, evaluate_shocks, probe_ends
from train.common import LEADS, day_phase, load_proxy_events, load_weather, recorded_proxy_cases, validation_start


ARCHIVE_SHA256 = "f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b"
SCRATCH = ROOT / "tests" / "_windows"
CPU_THREADS = 4
HOLDOUT_YEARS = 2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_rain_peaks(events):
    ratios = {}
    for event in ("cyclone", "heavy_rain"):
        rows = [row for row in events if row["event"] == event]
        observed = max(row["peak_rain_observed"] for row in rows)
        ratios[event] = {
            "peak_ratio": max(row["peak_rain_model"] for row in rows) / observed if observed else None,
            "ratios_by_lead": {
                row["lead"]: row["peak_rain_model"] / row["peak_rain_observed"] if row["peak_rain_observed"] else None
                for row in rows
            },
        }
    return ratios


def benchmark(dataset, weights, label):
    if sha256(dataset) != ARCHIVE_SHA256:
        raise ValueError("Use the original organizer archive for a comparable benchmark")
    fixture = load_proxy_events()
    # The prior audit records first-target selections. Keep those same five cases.
    cases = [f"{event}={date}" for event, date in recorded_proxy_cases(fixture)]
    shock_dates = fixture["shock_first_target_dates"]
    if len(cases) != 5 or len(set(shock_dates)) != 10 or len(shock_dates) != 10:
        raise ValueError("The archived protocol must contain five proxies and ten distinct shock targets")

    weather = load_weather(dataset)
    val_start = validation_start(weather.dates, HOLDOUT_YEARS)
    first_shock = probe_ends(weather, shock_dates, val_start, contract.SCORE_STEPS)[0]
    history = weather.physical[first_shock - 2:first_shock + 1]
    phase = day_phase(weather.day_of_year[first_shock])
    torch.set_num_threads(CPU_THREADS)
    start = time.perf_counter()
    engine = Engine(weights)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cold = engine.simulate(history, phase, contract.SCORE_STEPS)
    cold_seconds = time.perf_counter() - start

    train_dates = np.char.replace(weather.dates[:val_start].astype(str), "-", "").astype(int)
    climate = stats.climatology(weather.physical[:val_start], train_dates)

    def climatology_for(index):
        return stats.clim_for(climate, weather.dates[index].astype(object))

    forecast = evaluate_events(engine, weather, val_start, cases, climatology_for=climatology_for)
    forecast["max_rain_ratios"] = paired_rain_peaks(forecast["events"])
    shocks = evaluate_shocks(engine, weather, ROOT / "assets" / "stats.json", 0, val_start, shock_dates)
    responses = [response for window in shocks for response in window["shocks"].values()]
    members = []
    for path in checkpoint_paths(weights):
        package = torch.load(path, map_location="cpu", weights_only=True)
        members.append({
            "file": path.name,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "width": package["width"],
            "network_parameters": sum(tensor.numel() for tensor in package["model"].values()),
            "format_version": package["format_version"],
            "split": package.get("split"),
            "fit_period": package.get("fit_period"),
            "fit_periods": package.get("fit_periods"),
            "shrinkage": "shrinkage_alpha" in package,
        })
    primary = members[0]
    return {
        "label": label,
        "checkpoint": {
            **primary,
            "bytes": sum(member["bytes"] for member in members),
            "network_parameters": sum(member["network_parameters"] for member in members),
            "member_count": len(members),
            "members": members,
        },
        "protocol": {
            "archive_sha256": ARCHIVE_SHA256,
            "kit_version": version("monsoon-r1-kit"),
            "held_out_period": [str(weather.dates[val_start]), str(weather.dates[-1])],
            "lead_days": LEADS,
            "channels": contract.CHANNELS,
            "climatology": "monsoon_r1.stats.climatology on pre-holdout rows",
            "event_cases": cases,
            "shock_first_target_dates": shock_dates,
        },
        "forecast": forecast,
        "shocks": {
            "propagation": sum(response["propagates"] for response in responses),
            "locality": sum(response["localized"] for response in responses),
            "stability": sum(response["stable"] for response in responses),
            "response_count": len(responses),
            "zero_exact": sum(window["zero_shock_exact"] for window in shocks),
            "null_consistent": sum(window["null_consistent"] for window in shocks),
            "window_count": len(shocks),
            "windows": shocks,
        },
        "timing": {
            "cpu_threads": CPU_THREADS,
            "checkpoint_load_s": load_seconds,
            "cold_engine_simulate_s": cold_seconds,
            "cold_steps": contract.SCORE_STEPS,
            "cold_first_target": shock_dates[0],
            "validity_errors": scoring.validity(cold, contract.SCORE_STEPS),
        },
    }


def markdown(result):
    forecast, shocks = result["forecast"], result["shocks"]
    checkpoint = result["checkpoint"]
    label = " ".join(result["label"].split()).replace("|", "&#124;")
    day8 = forecast["mean_skill_vs_best_by_lead"][-1]
    rain = forecast["max_rain_ratios"]
    peak_cells = [f'{rain[event]["peak_ratio"]:.3f}' if rain[event]["peak_ratio"] is not None else "—"
                  for event in ("cyclone", "heavy_rain")]
    return (
        "| Label | Members | Width | Parameters | Bytes | Forecast /45 | Day-8 skill | Anti-blur /10 | Propagation | Locality | Stability | Exact zero | Cold 10-step s | Cyclone peak ratio | Heavy-rain peak ratio |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| {label} | {checkpoint['member_count']} | {checkpoint['width']} | {checkpoint['network_parameters']} | {checkpoint['bytes']} | "
        f"{forecast['points']:.2f} | {day8:.2%} | {forecast['anti_blur_points']:.2f} | "
        f"{shocks['propagation']}/{shocks['response_count']} | {shocks['locality']}/{shocks['response_count']} | "
        f"{shocks['stability']}/{shocks['response_count']} | {shocks['zero_exact']}/{shocks['window_count']} | "
        f"{result['timing']['cold_engine_simulate_s']:.4f} | {' | '.join(peak_cells)} |\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint file or its directory")
    parser.add_argument("--label", required=True, help="Name for the result row")
    parser.add_argument("--dataset", type=Path, default=SCRATCH / "india_climate_train.nc",
                        help="Path to the original organizer NetCDF archive")
    parser.add_argument("--output-dir", type=Path, default=SCRATCH / "benchmark")
    args = parser.parse_args()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.label).strip("-_")
    if not stem:
        parser.error("--label must contain at least one ASCII letter or digit")
    try:
        result = benchmark(args.dataset, args.weights, args.label)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = json.dumps(result, indent=2, allow_nan=False)
    table = markdown(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"benchmark-{stem}.json"
    table_path = result_path.with_suffix(".md")
    result_path.write_text(report + "\n", encoding="utf-8")
    table_path.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Wrote {result_path} and {table_path}")


if __name__ == "__main__":
    main()
