"""Workflow A's CPU engine for the Track 2 server."""

import math
from operator import index
from pathlib import Path

import numpy as np
import torch

from model.transport import transport_response
from model.unet import (
    STATE_SHAPE, AnomalyUNet, advance_phase, climatology_at, decode_physical,
    encode_physical, forecast_step, static_features,
)

CHECKPOINT_NAME = "airavat.pt"
MAX_STEPS = 14


class Engine:
    def __init__(self, weights_dir: str):
        path = Path(weights_dir) / CHECKPOINT_NAME
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError(f"Unsupported checkpoint format in {path}")
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)
        self.model_name = checkpoint["model_name"]
        self._model = AnomalyUNet(checkpoint["width"]).float()
        self._model.load_state_dict(checkpoint["model"])
        self._model.eval()
        self._coefficients = checkpoint["climatology"].float()
        self._scale = checkpoint["anomaly_scale"].float().view(1, 6, 1, 1)
        self._features = static_features(checkpoint["static"].float())[None]

    @torch.inference_mode()
    def _rollout(self, windows, phases, steps, perturbed=None):
        """Run an independent forecast for each window in a CPU batch."""
        if perturbed is not None:
            windows = torch.cat((windows, perturbed))
            phases = torch.cat((phases, phases))
        encoded = encode_physical(windows)
        if perturbed is not None:
            previous, branched = encoded[:, -1].chunk(2)
            response = branched - previous
            # Bound the first-day learned correction to the shock and nearby cells.
            # Wind-driven transport still moves and diffuses the response each day.
            initial_support = response.ne(0).any(dim=1, keepdim=True).to(response.dtype)
            first_day_support = torch.nn.functional.max_pool2d(initial_support, 7, stride=1, padding=3)
        initial_phases = torch.stack([advance_phase(phases, day) for day in (-2, -1, 0)], dim=1)
        initial_climatology = climatology_at(self._coefficients, initial_phases)
        history = (encoded - initial_climatology) / self._scale
        future = torch.stack([advance_phase(phases, day) for day in range(1, steps + 1)], dim=1)
        climate = climatology_at(self._coefficients, future)
        forecast = []
        for day in range(steps):
            history, encoded = forecast_step(
                self._model, history, future[:, day], climate[:, day], self._scale, self._features,
            )
            if perturbed is not None:
                reference, candidate = encoded.chunk(2)
                wind = 0.5 * (previous[:, 2:4] + reference[:, 2:4])
                transported = transport_response(response, wind)
                learned_response = candidate - reference
                if day == 0:
                    learned_response = learned_response * first_day_support
                branch = reference + 0.20 * learned_response + 0.80 * transported
                branch[:, 0].clamp_(233.15, 333.15)
                branch[:, 1].clamp_(0, math.log1p(400))
                updated = torch.cat((reference, branch))
                history[:, -1] = (updated - climate[:, day]) / self._scale
                response = branch - reference
                previous = reference
                encoded = branch
            forecast.append(decode_physical(encoded, lead=day + 1))
        return torch.stack(forecast, dim=1)

    def simulate(self, init_states, doy_sincos, steps, perturbation=None):
        steps = index(steps)
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
        states = _finite_array(init_states, (3, *STATE_SHAPE), "init_states")
        phase = _finite_array(doy_sincos, (2,), "doy_sincos")
        length = float(np.linalg.norm(phase))
        if not 0.9 <= length <= 1.1:
            raise ValueError("doy_sincos must describe a unit-circle angle")

        states = np.array(states, order="C", copy=True)
        perturbed = None
        if perturbation is not None:
            change = _finite_array(perturbation, STATE_SHAPE, "perturbation")
            if np.any(change):
                branch = states.copy()
                branch[-1] += change
                if not np.isfinite(branch[-1]).all():
                    raise ValueError("perturbed last input contains a non-finite value")
                perturbed = torch.from_numpy(branch)[None]
        direction = torch.from_numpy(phase / length)
        forecast = self._rollout(torch.from_numpy(states)[None], direction[None], steps, perturbed)[0].numpy()
        if not np.isfinite(forecast).all():
            raise RuntimeError("The model produced a non-finite forecast")
        return forecast


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return array
