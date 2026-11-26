"""Train the Track 2 autoregressive model and write an inference checkpoint."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from model.unet import (
    LAT, RAIN, AnomalyUNet, encode_physical, forecast_step, static_features,
)
from train.common import (
    PERIOD, TRAIN_END, VALIDATION_END, day_phase, fit_climatology, load_weather,
    seasonal_fields, window_ends,
)

DEFAULT_CURRICULUM = "1,1,1,2,2,3,3,4,4,5,6,8"


def synthetic_anomalies(batch_size, height, width, device):
    """Sample localized, spatially coherent anomalies in normalized units."""
    rows = torch.arange(height, device=device)[None, :, None]
    cols = torch.arange(width, device=device)[None, None, :]
    center_y = torch.rand((batch_size, 1, 1), device=device) * (height - 1)
    center_x = torch.rand((batch_size, 1, 1), device=device) * (width - 1)
    radius = 1.0 + torch.rand((batch_size, 1, 1), device=device) * 2.5
    envelope = torch.exp(-((rows - center_y).square() + (cols - center_x).square()) / (2 * radius.square()))
    magnitude = torch.randn((batch_size, 6, 1, 1), device=device).clamp(-2, 2) * 0.65
    active = (torch.rand((batch_size, 1, 1, 1), device=device) < 0.30)
    return envelope[:, None] * magnitude * active


def transport_anomaly(delta, wind):
    """Diffuse and advect a training perturbation using the local wind."""
    edge = F.pad(delta, (1, 1, 1, 1), mode="replicate")
    middle = edge[..., 1:-1, 1:-1]
    west, east = edge[..., 1:-1, :-2], edge[..., 1:-1, 2:]
    south, north = edge[..., :-2, 1:-1], edge[..., 2:, 1:-1]
    u, v = (wind / 20).clamp(-0.24, 0.24).unbind(dim=1)
    zonal = u.clamp_min(0)[:, None] * (west - middle) + (-u).clamp_min(0)[:, None] * (east - middle)
    meridional = v.clamp_min(0)[:, None] * (south - middle) + (-v).clamp_min(0)[:, None] * (north - middle)
    return 0.94 * (middle + zonal + meridional + 0.1 * (west + east + south + north - 4 * middle))


def rollout_loss(model, weather, doy, phase, climate, ends, horizon, scale, features, latitude_weight, augment):
    history_idx = ends[:, None] + torch.arange(-2, 1, device=ends.device)
    history = (weather[history_idx] - climate[doy[history_idx]]) / scale
    height, width = history.shape[-2:]
    delta = torch.zeros_like(history[:, -1])
    if augment:
        delta = synthetic_anomalies(len(ends), height, width, history.device)
    history = torch.cat((history[:, :-1], (history[:, -1] + delta)[:, None]), dim=1)
    loss = history.new_zeros(())
    for lead in range(1, horizon + 1):
        target_idx = ends + lead
        history, predicted = forecast_step(model, history, phase[target_idx], climate[doy[target_idx]], scale, features)
        target = weather[target_idx]
        if augment:
            delta = transport_anomaly(delta, weather[target_idx - 1, 2:4])
            target = target + delta * scale
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
        loss = loss + error + 0.13 * rain_loss + 0.16 * variance_loss
    return loss / horizon


def save_checkpoint(path, model, coefficients, scale, raw_static, width, epochs, split):
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
        "split": split,
    }
    pending = path.with_suffix(".tmp")
    torch.save(package, pending)
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM, help="Comma-separated rollout lengths, one per epoch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--resume", type=Path, help="Initialize the network from an earlier checkpoint")
    args = parser.parse_args()
    horizons = [int(value) for value in args.curriculum.split(",")]
    if min(horizons) < 1 or max(horizons) > 14 or args.width < 4 or args.width % 4:
        parser.error("Rollout lengths must be 1–14 and width must be a positive multiple of four")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)

    data = load_weather(args.dataset)
    cutoff = TRAIN_END if args.split == "train" else VALIDATION_END
    train_rows = data.dates < cutoff
    encoded = encode_physical(torch.from_numpy(data.physical)).numpy()
    coefficients = fit_climatology(encoded[train_rows], data.day_of_year[train_rows])
    climatology_np = seasonal_fields(coefficients)
    season_idx = data.day_of_year % PERIOD
    residual = encoded[train_rows] - climatology_np[season_idx[train_rows]]
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
    if args.resume:
        package = torch.load(args.resume, map_location="cpu", weights_only=True)
        if package["split"] != args.split or not np.array_equal(package["climatology"].numpy(), coefficients):
            parser.error("--resume requires the same training split and climatology")
        model.load_state_dict(package["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    print(f"Device={device}, records={len(data.dates)}, split={args.split}, rollout curriculum={horizons}", flush=True)
    for epoch, horizon in enumerate(horizons, start=1):
        start = time.monotonic()
        ends = rng.permutation(window_ends(data.dates, horizon, args.split))
        loss_total = torch.zeros((), device=device)
        for offset in range(0, len(ends), args.batch_size):
            batch = torch.as_tensor(ends[offset:offset + args.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(model, weather, day_idx, phase, climate, batch, horizon,
                                scale, features, latitude_weight, augment=epoch > 1 or args.resume is not None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            loss_total += loss.detach() * len(batch)
        average = float(loss_total.cpu()) / len(ends)
        if not np.isfinite(average):
            raise RuntimeError("Non-finite training loss")
        save_checkpoint(args.output, model, coefficients, scale, data.static, args.width, epoch, args.split)
        print(f"Epoch {epoch:02d} lead={horizon} loss={average:.4f} seconds={time.monotonic() - start:.1f}", flush=True)
        if epoch + 2 == len(horizons) and len(horizons) > 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5


if __name__ == "__main__":
    main()
