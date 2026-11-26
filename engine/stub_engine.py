"""Stub engine so serve.py and the dashboard can be developed before the real model exists.

Implements the Engine interface from CONTRACT.md. This is NOT a forecast model:
it repeats the last input day, adds the perturbation to step 0, then spreads the
perturbation field with a 3x3 blur and mild decay at every later step.

Properties the tests rely on:
  * zero / null perturbation returns exactly the repeated last day
  * a localized shock is local at step 0 and spreads outward afterwards
  * deterministic, tp >= 0, no NaN/Inf
"""
import numpy as np
from scipy.ndimage import uniform_filter


class Engine:
    model_name = "stub-persistence-spread"
    DECAY = 0.95

    def __init__(self, weights_dir=None):
        self.weights_dir = weights_dir

    def simulate(self, init_states, doy_sincos, steps, perturbation=None):
        x = np.asarray(init_states, dtype=np.float32)
        out = np.repeat(x[-1][None], int(steps), axis=0).astype(np.float32)
        if perturbation is not None:
            p = np.asarray(perturbation, dtype=np.float32).copy()
            if np.any(p != 0):
                for k in range(int(steps)):
                    if k > 0:
                        p = np.stack([uniform_filter(c, size=3, mode="nearest") for c in p])
                        p = (p * np.float32(self.DECAY)).astype(np.float32)
                    out[k] += p
        out[:, 1] = np.maximum(out[:, 1], 0.0)  # tp never negative
        return out
