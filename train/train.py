"""Train the Track 2 autoregressive model and write an inference checkpoint."""

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import torch
from torch.nn import functional as F

from model.unet import CHANNELS, LAT, RAIN, AnomalyUNet, encode_physical, forecast_step, static_features
from train.common import (
    DEFAULT_VAL_YEARS, PERIOD, day_phase, exclude_proxy_windows, fit_climatology, load_proxy_events, load_weather,
    recorded_proxy_cases, seasonal_fields, time_block_split, validation_start, window_ends,
)

DEFAULT_CURRICULUM = "1,1,1,2,2,3,4,4,6,8,8"
DEFAULT_FAR_FIELD_WEIGHT = 18.0
EARLY_STOP_PATIENCE = 3
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


def rain_event_loss(predicted, observed):
    """Retain intense rain and local gradients while tolerating small displacements."""
    predicted_mm, observed_mm = predicted.expm1(), observed.expm1()
    pred_root = (predicted_mm + 1).sqrt()
    obs_root = (observed_mm + 1).sqrt()
    weights = 1 + (observed_mm / 30).clamp_max(3)
    intensity = (F.smooth_l1_loss(pred_root, obs_root, reduction="none") * weights).mean()
    nearby = F.max_pool2d(pred_root[:, None], 3, stride=1, padding=1)[:, 0]
    extreme = observed_mm >= 30
    shortfall = F.relu(obs_root - nearby - 0.5)
    peaks = (shortfall.square() * extreme).sum() / extreme.sum().clamp_min(1)
    row_energy = F.l1_loss(pred_root.diff(dim=1).abs(), obs_root.diff(dim=1).abs())
    column_energy = F.l1_loss(pred_root.diff(dim=2).abs(), obs_root.diff(dim=2).abs())
    return 0.45 * intensity + 0.10 * peaks + 0.20 * (row_energy + column_energy)


def soft_csi_loss(logits, heavy):
    probability = logits.sigmoid()
    intersection = (probability * heavy).sum(dim=(-2, -1))
    union = (probability + heavy).sum(dim=(-2, -1)) - intersection
    return 1 - (intersection / union.clamp_min(1e-6)).mean()


def rain_spatial_energy_loss(predicted, observed):
    def log_energy(encoded):
        rain = encoded.expm1()
        return sum(rain.diff(dim=dim).square().mean(dim=(-2, -1)) for dim in (-2, -1)).log1p()

    return F.smooth_l1_loss(log_energy(predicted), log_energy(observed))


def scheduled_history(history, observed_anomaly, paired_count, probability, rng):
    """Mix in observations for unpaired clean windows only."""
    clean_count = len(observed_anomaly)
    eligible = slice(paired_count, clean_count)
    use_observed = torch.rand((clean_count - paired_count, 1, 1, 1), generator=rng)
    use_observed = use_observed.to(history.device) < probability
    latest = torch.where(use_observed, observed_anomaly[eligible], history[eligible, -1])
    latest = torch.cat((history[:paired_count, -1], latest, history[clean_count:, -1]))
    return torch.cat((history[:, :-1], latest[:, None]), dim=1)


def rollout_loss(model, weather, doy, phase, climate, ends, horizon, scale, features, latitude_weight,
                 augment, far_field_weight, event_rain_weight, *,
                 rain_csi_weight=0.0, rain_energy_weight=0.0, teacher_forcing=0.0, sampling_rng=None):
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
        if event_rain_weight:
            loss = loss + event_rain_weight * rain_event_loss(predicted_rain, rain)
        if rain_csi_weight:
            loss = loss + rain_csi_weight * soft_csi_loss(logits, heavy)
        if rain_energy_weight:
            loss = loss + rain_energy_weight * rain_spatial_energy_loss(predicted_rain, rain)
        if teacher_forcing and lead < horizon and clean_count > shock_count:
            observed_anomaly = (target - climate[doy[target_idx]]) / scale
            history = scheduled_history(history, observed_anomaly, shock_count, teacher_forcing, sampling_rng)
    return loss / horizon


def save_checkpoint(path, model, coefficients, scale, raw_static, width, epochs, fit_metadata):
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
    write_atomically(path, lambda pending: torch.save(package, pending))


def write_atomically(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        pending = Path(file.name)
    try:
        write(pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def training_order(ends, years, rng, weight_recent):
    if not weight_recent or years[ends[-1]] == years[ends[0]]:
        return rng.permutation(ends)
    age = years[ends] - years[ends[0]]
    weights = 1 + age / age[-1]
    return rng.choice(ends, size=len(ends), replace=True, p=weights / weights.sum())


def climatology_partitions(parts, years, clim_years):
    if clim_years is None:
        return parts
    first, last = clim_years
    if not 1 <= first <= last <= 9998:
        raise ValueError("--clim-years requires ordered calendar years between 1 and 9998")
    bounds = [np.datetime64(f"{year:04d}", "Y") for year in (first, last + 1)]
    left, right = map(int, np.searchsorted(years, bounds))
    selected = tuple(slice(max(left, part.start), min(right, part.stop)) for part in parts
                     if max(left, part.start) < min(right, part.stop))
    if not selected:
        raise ValueError("--clim-years contains no eligible training records")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--val-years", type=int, default=DEFAULT_VAL_YEARS,
                        help="Latest calendar years to hold out with --split train")
    parser.add_argument("--clim-years", nargs=2, type=int, metavar=("START", "END"),
                        help="Fit climatology and scale on eligible training records in this inclusive range")
    parser.add_argument("--folds", nargs=2, type=int, metavar=("START", "END"),
                        help="Use one calendar-year validation block with a 14-day buffer on each side")
    parser.add_argument("--sample-weight-recent", action="store_true",
                        help="Sample with replacement, with calendar-year weights rising from 1 to 2")
    parser.add_argument("--max-epochs", type=int, help="Cap the number of epochs in the supplied curriculum")
    parser.add_argument("--curriculum", default=DEFAULT_CURRICULUM, help="Comma-separated rollout lengths, one per epoch")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--far-field-weight", type=float, default=DEFAULT_FAR_FIELD_WEIGHT,
                        help="Penalty on day-one shock energy outside the perturbed neighborhood")
    parser.add_argument("--event-rain-weight", type=float, default=0,
                        help="Weight for the fine-tuning loss on rain intensity, peaks and gradients")
    parser.add_argument("--rain-csi-weight", type=float, default=0, help="Weight for the soft heavy-rain CSI loss")
    parser.add_argument("--rain-energy-weight", type=float, default=0, help="Weight for rain spatial gradient energy")
    parser.add_argument("--scheduled-sampling", action="store_true",
                        help="Anneal observation feedback from 50%% to zero in unpaired training windows")
    parser.add_argument("--validation-steps", type=int, help="Use a common held-out rollout horizon across runs")
    parser.add_argument("--exclude-proxy-windows", action="store_true",
                        help="Reserve the five benchmark proxies and use the base loss for validation")
    parser.add_argument("--resume", type=Path, help="Initialize the network from an earlier checkpoint")
    args = parser.parse_args()
    try:
        horizons = [int(value) for value in args.curriculum.split(",")]
    except ValueError:
        parser.error("--curriculum must be a comma-separated list of integers")
    if min(horizons) < 1 or max(horizons) > 14 or args.width < 4 or args.width % 4:
        parser.error("Rollout lengths must be 1–14 and width must be a positive multiple of four")
    if args.batch_size < 1 or args.threads < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Batch size, thread count, and learning rate must be positive")
    if args.val_years < 1 or not math.isfinite(args.far_field_weight) or args.far_field_weight < 0:
        parser.error("Validation years must be positive and far-field weight must be finite and nonnegative")
    for weight in (args.event_rain_weight, args.rain_csi_weight, args.rain_energy_weight):
        if not math.isfinite(weight) or weight < 0:
            parser.error("Rain loss weights must be finite and nonnegative")
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            parser.error("--max-epochs must be positive")
        horizons = horizons[:args.max_epochs]
    if args.folds is not None and args.split == "all":
        parser.error("--folds requires a held-out split")
    if args.validation_steps is not None and not 1 <= args.validation_steps <= 14:
        parser.error("--validation-steps must be between 1 and 14")
    if args.split == "all" and (args.validation_steps is not None or args.exclude_proxy_windows):
        parser.error("Validation controls require a held-out split")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    torch.use_deterministic_algorithms(True, warn_only=device.type == "mps")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sampling_rng = torch.Generator().manual_seed(args.seed) if args.scheduled_sampling else None

    data = load_weather(args.dataset)
    record_count = len(data.dates)
    years = data.dates.astype("datetime64[Y]")
    try:
        if args.folds is not None:
            train_parts, validation = time_block_split(data.dates, *args.folds)
        elif args.split == "train":
            val_start = validation_start(data.dates, args.val_years)
            train_parts, validation = (slice(0, val_start),), slice(val_start, record_count)
        else:
            train_parts, validation = (slice(0, record_count),), None
        climate_parts = climatology_partitions(train_parts, years, args.clim_years)
    except ValueError as exc:
        parser.error(str(exc))
    if not train_parts:
        parser.error("No training records remain outside the validation fold and buffer")

    def period(part):
        return [str(data.dates[part.start]), str(data.dates[part.stop - 1])]

    fit_periods = [period(part) for part in train_parts]
    fit_metadata = {
        "split": "fold" if args.folds is not None else args.split,
        "val_years": args.val_years if validation is not None and args.folds is None else None,
        "fit_period": [fit_periods[0][0], fit_periods[-1][-1]],
        "fit_periods": fit_periods,
        "validation_period": period(validation) if validation is not None else None,
        "clim_years": args.clim_years,
        "climatology_fit_periods": [period(part) for part in climate_parts],
        "sample_weight_recent": args.sample_weight_recent,
        "far_field_weight": args.far_field_weight,
        "event_rain_weight": args.event_rain_weight,
        "rain_csi_weight": args.rain_csi_weight,
        "rain_energy_weight": args.rain_energy_weight,
        "scheduled_sampling": args.scheduled_sampling,
        "exclude_proxy_windows": args.exclude_proxy_windows,
        "seed": args.seed,
        "curriculum": horizons,
        "learning_rate": args.learning_rate,
    }
    windows = {
        steps: np.concatenate([window_ends(part.stop - part.start, steps, "all") + part.start
                               for part in train_parts])
        for steps in set(horizons)
    }
    for steps, ends in windows.items():
        if not len(ends):
            parser.error(f"No complete {steps}-day training windows")
    validation_horizon = args.validation_steps or max(horizons)
    validation_batches = ()
    validation_count = 0
    if validation is not None:
        max_validation_horizon = validation.stop - validation.start - 3
        if args.validation_steps is not None and validation_horizon > max_validation_horizon:
            parser.error("--validation-steps exceeds the available validation window length")
        validation_horizon = min(validation_horizon, max_validation_horizon)
        if validation_horizon < 1:
            parser.error("The validation partition needs at least three input days and one target day")
        validation_ends = window_ends(validation.stop - validation.start, validation_horizon, "all") + validation.start
        if args.exclude_proxy_windows:
            targets = [date for _, date in recorded_proxy_cases(load_proxy_events())]
            validation_ends = exclude_proxy_windows(
                data.dates, validation_ends, validation_horizon, targets
            )
        validation_count = len(validation_ends)
        if not validation_count:
            parser.error("No complete validation windows remain")
        validation_batches = torch.as_tensor(validation_ends, device=device).split(args.batch_size)

    fit_rows = climate_parts[0] if len(climate_parts) == 1 else np.concatenate(
        [np.arange(part.start, part.stop) for part in climate_parts]
    )
    encoded = encode_physical(torch.from_numpy(data.physical)).numpy()
    season_idx = data.day_of_year % PERIOD
    climate_fields = encoded[fit_rows]
    coefficients = fit_climatology(climate_fields, data.day_of_year[fit_rows])
    climatology_np = seasonal_fields(coefficients)
    residual = climate_fields - climatology_np[season_idx[fit_rows]]
    scale_np = np.maximum(residual.std(axis=(0, 2, 3)), 1e-4)
    del climate_fields, residual

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
        if package["split"] != fit_metadata["split"] or package["width"] != args.width:
            parser.error("--resume requires the same training split and model width")
        if args.folds is not None and package.get("validation_period") != fit_metadata["validation_period"]:
            parser.error("--resume requires the same validation fold")
        if not np.array_equal(package["climatology"].numpy(), coefficients):
            parser.error("--resume requires the same dataset and fitted climatology")
        model.load_state_dict(package["model"])
        completed_epochs = package["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
    loss_path = args.output.with_suffix(".loss.json")
    history = {"checkpoint_epoch": None, "stopped_early": False, "validation_windows": validation_count, "epochs": []}
    best_loss, stale_epochs = math.inf, 0
    print(f"Device={device}, fit={fit_metadata['fit_period']}, split={fit_metadata['split']}, "
          f"rollout curriculum={horizons}, far-field weight={args.far_field_weight:g}", flush=True)
    validation_event_weight = 0.0 if args.exclude_proxy_windows else args.event_rain_weight
    for epoch, horizon in enumerate(horizons, start=1):
        start = time.monotonic()
        model.train()
        ends = training_order(windows[horizon], years, rng, args.sample_weight_recent)
        batches = torch.as_tensor(ends, device=device).split(args.batch_size)
        teacher_forcing = 0.0
        if args.scheduled_sampling and len(horizons) > 1:
            teacher_forcing = 0.5 * (len(horizons) - epoch) / (len(horizons) - 1)
        loss_total = torch.zeros((), device=device)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = rollout_loss(model, weather, day_idx, phase, climate, batch, horizon,
                                scale, features, latitude_weight, augment=epoch > 1 or args.resume is not None,
                                far_field_weight=args.far_field_weight, event_rain_weight=args.event_rain_weight,
                                rain_csi_weight=args.rain_csi_weight, rain_energy_weight=args.rain_energy_weight,
                                teacher_forcing=teacher_forcing, sampling_rng=sampling_rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            loss_total += loss.detach() * len(batch)
        average = float(loss_total.cpu()) / len(ends)
        if not np.isfinite(average):
            raise RuntimeError("Non-finite training loss")
        val_loss = None
        if validation_batches:
            model.eval()
            with torch.inference_mode():
                val_total = torch.zeros((), device=device)
                for batch in validation_batches:
                    loss = rollout_loss(model, weather, day_idx, phase, climate, batch, validation_horizon,
                                        scale, features, latitude_weight, augment=False,
                                        far_field_weight=args.far_field_weight, event_rain_weight=validation_event_weight)
                    val_total += loss * len(batch)
                val_loss = float(val_total.cpu()) / validation_count
            if not math.isfinite(val_loss):
                raise RuntimeError("Non-finite validation loss")
        current_epoch = completed_epochs + epoch
        if val_loss is None or val_loss < best_loss:
            save_checkpoint(args.output, model, coefficients, scale, data.static, args.width, current_epoch, fit_metadata)
            history["checkpoint_epoch"] = current_epoch
            if val_loss is not None:
                best_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        history["stopped_early"] = stale_epochs >= EARLY_STOP_PATIENCE and epoch < len(horizons)
        history["epochs"].append({"epoch": current_epoch, "train_horizon": horizon, "train_loss": average,
                                  "teacher_forcing": teacher_forcing,
                                  "validation_horizon": validation_horizon if validation_batches else None,
                                  "val_loss": val_loss})
        write_atomically(loss_path, lambda pending: pending.write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        ))
        validation_text = f" val_loss={val_loss:.4f}" if val_loss is not None else ""
        print(f"Epoch {epoch:02d} lead={horizon} loss={average:.4f}{validation_text} "
              f"seconds={time.monotonic() - start:.1f}", flush=True)
        if history["stopped_early"]:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement", flush=True)
            break
        if epoch + 2 == len(horizons) and len(horizons) > 3:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5


if __name__ == "__main__":
    main()
