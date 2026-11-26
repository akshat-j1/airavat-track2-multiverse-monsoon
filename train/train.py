"""Train the Track 2 autoregressive model and write an inference checkpoint."""

import argparse
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import torch
from torch.nn import functional as F

from model.unet import CHANNELS, LAT, RAIN, AnomalyUNet, encode_physical, forecast_step, static_features
from train.common import (
    DEFAULT_VAL_YEARS, PERIOD, day_phase, fit_climatology, load_weather,
    seasonal_fields, validation_start, window_ends,
)

DEFAULT_CURRICULUM = "1,1,1,2,2,3,4,4,6,8,8"
DEFAULT_FAR_FIELD_WEIGHT = 18.0
PRESSURE = CHANNELS.index("msl")


def synthetic_anomalies(batch_size, height, width, device):
    """Sample localized, spatially coherent anomalies in normalized units."""
    rows = torch.arange(height, device=device)[None, :, None]
    cols = torch.arange(width, device=device)[None, None, :]
    center_y = torch.rand((batch_size, 1, 1), device=device) * (height - 1)
    center_x = torch.rand((batch_size, 1, 1), device=device) * (width - 1)
    radius = 2.0 + torch.rand((batch_size, 1, 1), device=device) * 4.0
    distance = ((rows - center_y).square() + (cols - center_x).square()).sqrt()
    envelope = 0.5 * (1 + (math.pi * (distance / radius).clamp_max(1)).cos())
    magnitude = torch.randn((batch_size, 6, 1, 1), device=device).clamp(-2, 2) * 0.65
    return envelope[:, None] * magnitude, (distance > radius + 3)[:, None]


def transport_anomaly(delta, wind):
    """Diffuse and advect a training perturbation using the local wind."""
    u, v = (wind[:, :, None] * 0.22).clamp(-0.26, 0.26).unbind(dim=1)
    for _ in range(4):
        edge = F.pad(delta, (1, 1, 1, 1), mode="replicate")
        west, east = edge[..., 1:-1, :-2], edge[..., 1:-1, 2:]
        south, north = edge[..., :-2, 1:-1], edge[..., 2:, 1:-1]
        zonal = u.clamp_min(0) * (west - delta) + (-u).clamp_min(0) * (east - delta)
        meridional = v.clamp_min(0) * (south - delta) + (-v).clamp_min(0) * (north - delta)
        diffusion = 0.09 * (west + east + south + north - 4 * delta)
        delta = 0.987 * (delta + zonal + meridional + diffusion)
    return delta


def rollout_loss(model, weather, doy, phase, climate, ends, horizon, scale, features, latitude_weight,
                 augment, far_field_weight):
    history_idx = ends[:, None] + torch.arange(-2, 1, device=ends.device)
    history = (weather[history_idx] - climate[doy[history_idx]]) / scale
    clean_count = len(ends)
    shock_count = max(1, clean_count // 4) if augment else 0
    if augment:
        height, width = history.shape[-2:]
        delta, far_field = synthetic_anomalies(shock_count, height, width, history.device)
        shock_history = torch.cat((history[:shock_count, :-1],
                                   (history[:shock_count, -1] + delta)[:, None]), dim=1)
        history = torch.cat((history, shock_history), dim=0)
    loss = history.new_zeros(())
    for lead in range(1, horizon + 1):
        target_idx = ends + lead
        paired_idx = torch.cat((target_idx, target_idx[:shock_count])) if augment else target_idx
        history, predictions = forecast_step(model, history, phase[paired_idx], climate[doy[paired_idx]], scale, features)
        predicted = predictions[:clean_count]
        target = weather[target_idx]
        paired_loss = loss.new_zeros(())
        if augment:
            delta = transport_anomaly(delta, weather[target_idx[:shock_count] - 1, 2:4])
            response = (predictions[clean_count:] - predicted[:shock_count]) / scale
            # The paired objective teaches response dynamics independently of the weather error.
            paired_loss = 3.5 * F.mse_loss(response, delta)
            if lead == 1:
                leakage = (response * far_field).square()
                # Small remote pressure errors can dominate response energy in physical units.
                paired_loss += far_field_weight * (leakage.mean() + 6 * leakage[:, PRESSURE].mean())
        rain = target[:, RAIN].clamp_min(0)
        error = (predicted - target) / scale
        error = (error.square() * latitude_weight).mean()
        predicted_rain = predicted[:, RAIN]
        heavy = (rain > np.log1p(10)).to(predicted.dtype)
        # Learn the threshold as well as the amount, so isolated wet extremes have a training signal.
        logits = 3 * (predicted_rain - np.log1p(10))
        rain_weight = 1 + 1.7 * heavy
        rain_loss = (F.binary_cross_entropy_with_logits(logits, heavy, reduction="none") * rain_weight).mean()
        pred_std = (predicted / scale).flatten(2).std(dim=-1, correction=0)
        true_std = (target / scale).flatten(2).std(dim=-1, correction=0)
        variance_loss = F.relu(0.76 * true_std - pred_std).square().mean()
        rain_std = rain.expm1().flatten(1).std(dim=-1, correction=0)
        predicted_rain_std = predicted_rain.expm1().flatten(1).std(dim=-1, correction=0)
        rain_variance_loss = (F.relu(0.76 * rain_std - predicted_rain_std) / 10).square().mean()
        loss = loss + error + 0.13 * rain_loss + 0.16 * variance_loss + 0.09 * rain_variance_loss + paired_loss
    return loss / horizon


def save_checkpoint(path, model, coefficients, scale, raw_static, width, epochs, fit_metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "format_version": 1,
        "model_name": "airavat-anomaly-unet",
        "width": width,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "climatology": torch.from_numpy(coefficients.copy()),
        "anomaly_scale": scale.detach().cpu().reshape(6),
        "static": torch.from_numpy(raw_static.copy()),
        "epochs": epochs,
        **fit_metadata,
    }
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        pending = Path(file.name)
    try:
        torch.save(package, pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS,
                        help="Latest calendar years to hold out with --split train")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM, help="Comma-separated rollout lengths, one per epoch")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--far-field-weight", type=float, default=DEFAULT_FAR_FIELD_WEIGHT,
                        help="Penalty on day-one shock energy outside the perturbed neighborhood")
    parser.add_argument("--resume", type=Path, help="Initialize the network from an earlier checkpoint")
    args = parser.parse_args()
    try:
        horizons = [int(value) for value in args.curriculum.split(",")]
    except ValueError:
        parser.error("--curriculum must be a comma-separated list of integers")
    if min(horizons) < 1 or max(horizons) > 14 or args.width < 4 or args.width % 4:
        parser.error("Rollout lengths must be 1–14 and width must be a positive multiple of four")
    if args.batch_size < 1 or args.threads < 1 or args.learning_rate <= 0:
        parser.error("Batch size, thread count, and learning rate must be positive")
    if args.val_years < 1 or not math.isfinite(args.far_field_weight) or args.far_field_weight < 0:
        parser.error("Validation years must be positive and far-field weight must be finite and nonnegative")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    data = load_weather(args.dataset)
    try:
        val_start = validation_start(data.dates, args.val_years) if args.split == "train" else None
    except ValueError as exc:
        parser.error(str(exc))
    fit_rows = slice(None, val_start)
    fit_dates = data.dates[fit_rows]
    fit_metadata = {
        "split": args.split,
        "val_years": args.val_years if val_start is not None else None,
        "fit_period": [str(fit_dates[0]), str(fit_dates[-1])],
        "far_field_weight": args.far_field_weight,
    }
    windows = {steps: window_ends(len(data.dates), steps, args.split, val_start=val_start)
               for steps in set(horizons)}
    for steps, ends in windows.items():
        if not len(ends):
            parser.error(f"No {steps}-day windows in split {args.split}")
    encoded = encode_physical(torch.from_numpy(data.physical)).numpy()
    coefficients = fit_climatology(encoded[fit_rows], data.day_of_year[fit_rows])
    climatology_np = seasonal_fields(coefficients)
    season_idx = data.day_of_year % PERIOD
    residual = encoded[fit_rows] - climatology_np[season_idx[fit_rows]]
    scale_np = np.maximum(residual.std(axis=(0, 2, 3)), 1e-4)
    del residual

    weather = torch.as_tensor(encoded, device=device)
    day_idx = torch.as_tensor(season_idx, dtype=torch.long, device=device)
    phase = torch.as_tensor(day_phase(data.day_of_year), device=device)
    climate = torch.as_tensor(climatology_np, device=device)
    scale = torch.as_tensor(scale_np, device=device).view(1, 6, 1, 1)
    features = static_features(torch.as_tensor(data.static, device=device))[None]
    latitude_weight = torch.as_tensor(np.cos(np.deg2rad(LAT)), device=device)
    latitude_weight = (latitude_weight / latitude_weight.mean())[None, None, :, None]

    model = AnomalyUNet(args.width).to(device)
    completed_epochs = 0
    if args.resume:
        package = torch.load(args.resume, map_location="cpu", weights_only=True)
        if package["split"] != args.split or package["width"] != args.width:
            parser.error("--resume requires the same training split and model width")
        if not np.array_equal(package["climatology"].numpy(), coefficients):
            parser.error("--resume requires the same dataset and fitted climatology")
        model.load_state_dict(package["model"])
        completed_epochs = package["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    print(f"Device={device}, fit={fit_metadata['fit_period']}, split={args.split}, "
          f"rollout curriculum={horizons}, far-field weight={args.far_field_weight:g}", flush=True)
    for epoch, horizon in enumerate(horizons, start=1):
        start = time.monotonic()
        ends = rng.permutation(windows[horizon])
        loss_total = torch.zeros((), device=device)
        for offset in range(0, len(ends), args.batch_size):
            batch = torch.as_tensor(ends[offset:offset + args.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(model, weather, day_idx, phase, climate, batch, horizon,
                                scale, features, latitude_weight, augment=epoch > 1 or args.resume is not None,
                                far_field_weight=args.far_field_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            loss_total += loss.detach() * len(batch)
        average = float(loss_total.cpu()) / len(ends)
        if not np.isfinite(average):
            raise RuntimeError("Non-finite training loss")
        save_checkpoint(args.output, model, coefficients, scale, data.static, args.width,
                        completed_epochs + epoch, fit_metadata)
        print(f"Epoch {epoch:02d} lead={horizon} loss={average:.4f} seconds={time.monotonic() - start:.1f}", flush=True)
        if epoch + 2 == len(horizons) and len(horizons) > 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5


if __name__ == "__main__":
    main()
