"""Workflow A's CPU engine for the Track 2 server."""

import math
from operator import index
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model.transport import advance_adjustment, transport_response
from model.unet import (
    CHANNELS, STATE_SHAPE, AnomalyUNet, advance_phase, climatology_at, decode_physical,
    encode_physical, forecast_step, static_features,
)

CHECKPOINT_NAME = "airavat.pt"
MEMBER_PATTERN = "airavat-member-*.pt"
MAX_STEPS = 14
ENCODED_BOUNDS = (
    (233.15, 333.15), (0, math.log1p(400)), (-100, 100), (-100, 100),
    (80000, 115000), (0, 150),
)


def checkpoint_paths(weights):
    path = Path(weights)
    if path.is_dir():
        return [path / CHECKPOINT_NAME, *sorted(path.glob(MEMBER_PATTERN))]
    return [path]


class Engine:
    def __init__(self, weights_dir: str):
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)
        paths = checkpoint_paths(weights_dir)
        self._models = []
        primary = None
        shared_tensors = ("climatology", "anomaly_scale", "static")
        for path in paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["format_version"] != 1:
                raise ValueError(f"Unsupported checkpoint format in {path}")
            if primary is None:
                primary = checkpoint
            elif ("shrinkage_alpha" in checkpoint or any(
                    not torch.equal(checkpoint[name], primary[name]) for name in shared_tensors)):
                raise ValueError(f"Ensemble member {path} must share the primary climatology, scale and static fields "
                                 "and leave shrinkage to the primary checkpoint")
            model = AnomalyUNet(checkpoint["width"]).float()
            model.load_state_dict(checkpoint["model"])
            self._models.append(model.eval())
        self.model_name = primary["model_name"]
        self._coefficients = primary["climatology"].float()
        self._scale = primary["anomaly_scale"].float().view(1, 6, 1, 1)
        self._features = static_features(primary["static"].float())[None]
        self._alpha = primary.get("shrinkage_alpha")
        if self._alpha is not None:
            alpha = self._alpha
            if (not isinstance(alpha, torch.Tensor) or not alpha.is_floating_point()
                    or alpha.shape != (MAX_STEPS, len(CHANNELS)) or not torch.isfinite(alpha).all()
                    or (alpha < 0).any() or (alpha > 1).any()):
                raise ValueError(f"shrinkage_alpha must be a finite {MAX_STEPS}×{len(CHANNELS)} tensor in [0, 1]")
            self._alpha = alpha.float()[:, :, None, None]
        self._lower, self._upper = torch.tensor(ENCODED_BOUNDS, dtype=torch.float32).T.reshape(2, 1, 6, 1, 1)

    @torch.inference_mode()
    def _rollout(self, windows, phases, steps, perturbed=None):
        """Run an independent forecast for each window in a CPU batch."""
        encoded, climate = self._encoded_forecast(windows, phases, steps, perturbed)
        forecast = []
        for day, output in enumerate(encoded):
            if self._alpha is not None:
                alpha = self._alpha[day]
                seasonal = climate[:, day]
                shrunk = seasonal + alpha * (output - seasonal)
                output = torch.where(alpha == 1, output, shrunk).clamp(self._lower, self._upper)
            forecast.append(decode_physical(output, lead=day + 1))
        return torch.stack(forecast, dim=1)

    @torch.inference_mode()
    def _encoded_forecast(self, windows, phases, steps, perturbed=None):
        encoded = encode_physical(windows)
        initial_phases = torch.stack([advance_phase(phases, day) for day in (-2, -1, 0)], dim=1)
        initial_climatology = climatology_at(self._coefficients, initial_phases)
        branch = encode_physical(perturbed) if perturbed is not None else None
        future = torch.stack([advance_phase(phases, day) for day in range(1, steps + 1)], dim=1)
        climate = climatology_at(self._coefficients, future)
        members = [self._member_rollout(model, encoded, initial_climatology, future, climate, branch)
                   for model in self._models]
        if len(members) == 1:
            return members[0], climate
        outputs = []
        for day, states in enumerate(zip(*members)):
            seasonal = climate[:, day]
            anomaly = torch.stack([state - seasonal for state in states]).mean(dim=0)
            outputs.append((seasonal + anomaly).clamp(self._lower, self._upper))
        return outputs, climate

    def _member_rollout(self, model, encoded, initial_climatology, future, climate, branch_encoded):
        history = (encoded - initial_climatology) / self._scale
        if branch_encoded is not None:
            branch_history = (branch_encoded - initial_climatology) / self._scale
            previous = encoded[:, -1]
            response = branch_encoded[:, -1] - previous
            support = response.ne(0).any(dim=1, keepdim=True).to(response.dtype)
            adjustment = response[:, 2:5]
            tendency = torch.zeros_like(adjustment)
        forecast = []
        for day in range(future.shape[1]):
            phase, seasonal = future[:, day], climate[:, day]
            history, encoded = forecast_step(model, history, phase, seasonal, self._scale, self._features)
            reference = encoded.clamp(self._lower, self._upper)
            output = reference
            if branch_encoded is not None:
                branch_history, candidate = forecast_step(
                    model, branch_history, phase, seasonal, self._scale, self._features,
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
            forecast.append(output)
        return forecast

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
