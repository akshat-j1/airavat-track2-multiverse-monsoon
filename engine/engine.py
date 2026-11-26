"""Workflow A's CPU engine for the Track 2 server."""

import math
from operator import index
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model.transport import advance_adjustment, transport_response
from model.unet import (
    STATE_SHAPE, AnomalyUNet, advance_phase, climatology_at, decode_physical,
    encode_physical, forecast_step, static_features,
)

CHECKPOINT_NAME = "airavat.pt"
MAX_STEPS = 14
ENCODED_BOUNDS = (
    (233.15, 333.15), (0, math.log1p(400)), (-100, 100), (-100, 100),
    (80000, 115000), (0, 150),
)


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
        self._lower, self._upper = torch.tensor(ENCODED_BOUNDS, dtype=torch.float32).T.reshape(2, 1, 6, 1, 1)

    @torch.inference_mode()
    def _rollout(self, windows, phases, steps, perturbed=None):
        """Run an independent forecast for each window in a CPU batch."""
        encoded = encode_physical(windows)
        initial_phases = torch.stack([advance_phase(phases, day) for day in (-2, -1, 0)], dim=1)
        initial_climatology = climatology_at(self._coefficients, initial_phases)
        history = (encoded - initial_climatology) / self._scale
        if perturbed is not None:
            branch_encoded = encode_physical(perturbed)
            branch_history = (branch_encoded - initial_climatology) / self._scale
            previous = encoded[:, -1]
            response = branch_encoded[:, -1] - previous
            support = response.ne(0).any(dim=1, keepdim=True).to(response.dtype)
            adjustment = response[:, 2:5]
            tendency = torch.zeros_like(adjustment)
        future = torch.stack([advance_phase(phases, day) for day in range(1, steps + 1)], dim=1)
        climate = climatology_at(self._coefficients, future)
        forecast = []
        for day in range(steps):
            phase, seasonal = future[:, day], climate[:, day]
            history, encoded = forecast_step(self._model, history, phase, seasonal, self._scale, self._features)
            reference = encoded.clamp(self._lower, self._upper)
            output = reference
            if perturbed is not None:
                branch_history, candidate = forecast_step(
                    self._model, branch_history, phase, seasonal, self._scale, self._features,
                )
                wind = 0.5 * (previous[:, 2:4] + reference[:, 2:4])
                transported = transport_response(response, wind)
                adjustment, tendency = advance_adjustment(adjustment, tendency)
                # Moisture, temperature and rain retain purely advective mixing.
                transported[:, 2:5] = 0.65 * transported[:, 2:5] + 0.35 * adjustment
                # Advance a bounded domain of influence for both response terms.
                support = F.max_pool2d(support, 7, stride=1, padding=3)
                learned_weight = 0.20 if day == 0 else 0.08
                shift = support * (learned_weight * (candidate - encoded) + (1 - learned_weight) * transported)
                output = self._apply_response(reference, shift)
                branch_history[:, -1] = (output - seasonal) / self._scale
                response = output - reference
                previous = reference
            forecast.append(decode_physical(output, lead=day + 1))
        return torch.stack(forecast, dim=1)

    def _apply_response(self, reference, shift):
        """Compress extreme changes smoothly near a physical bound."""
        headroom = torch.where(shift >= 0, self._upper - reference, reference - self._lower)
        magnitude = shift.abs()
        shoulder = 0.75 * headroom
        tail = 0.25 * headroom
        excess = (magnitude - shoulder).clamp_min(0)
        soft = shoulder - tail * torch.expm1(-excess / tail.clamp_min(1e-6))
        bounded = torch.where(magnitude <= shoulder, magnitude, soft)
        return (reference + shift.sign() * bounded).clamp(self._lower, self._upper)

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
                if not np.array_equal(branch[-1], states[-1]):
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
