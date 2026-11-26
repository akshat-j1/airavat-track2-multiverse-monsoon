"""Workflow A's CPU engine for the Track 2 server."""

from operator import index
from pathlib import Path

import numpy as np
import torch

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
    def _rollout(self, windows, phases, steps):
        """Run an independent forecast for each window in a CPU batch."""
        encoded = encode_physical(windows)
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
            forecast.append(decode_physical(encoded))
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

        # Copy before perturbing the last input, including when the input is a read-only view.
        states = np.array(states, order="C", copy=True)
        if perturbation is not None:
            change = _finite_array(perturbation, STATE_SHAPE, "perturbation")
            if np.any(change):
                states[-1] += change
                if not np.isfinite(states[-1]).all():
                    raise ValueError("perturbed last input contains a non-finite value")
        direction = torch.from_numpy(phase / length)
        forecast = self._rollout(torch.from_numpy(states)[None], direction[None], steps)[0].numpy()
        if not np.isfinite(forecast).all():
            raise RuntimeError("The model produced a non-finite forecast")
        return forecast


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return array
