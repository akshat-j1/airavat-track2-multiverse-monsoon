"""Residual U-Net with an autoregressive seasonal-anomaly state."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
STATIC_CHANNELS = ("land_sea_mask", "orography_m")
LAT = np.arange(6.5, 38.5, dtype="f4")
LON = np.arange(65.5, 101.5, dtype="f4")
STATE_SHAPE = (len(CHANNELS), len(LAT), len(LON))
RAIN = CHANNELS.index("tp")


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
        )

    def forward(self, x):
        return F.silu(x + self.layers(x))


class AnomalyUNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.input = nn.Conv2d(3 * len(CHANNELS) + 6, width, 3, padding=1)
        self.fine = ResidualBlock(width)
        self.down = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.middle = ResidualBlock(width)
        self.down_coarse = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.coarse = ResidualBlock(width)
        self.up_middle = nn.Conv2d(2 * width, width, 3, padding=1)
        self.up_fine = nn.Conv2d(2 * width, width, 3, padding=1)
        self.output = nn.Conv2d(width, len(CHANNELS), 3, padding=1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, history, features, phase):
        size = history.shape[-2:]
        phase_maps = phase[:, :, None, None].expand(-1, -1, *size)
        x = torch.cat((history.flatten(1, 2), features.expand(history.shape[0], -1, -1, -1), phase_maps), dim=1)
        fine = self.fine(self.input(x))
        middle = self.middle(self.down(fine))
        coarse = self.coarse(self.down_coarse(middle))
        coarse = F.interpolate(coarse, size=middle.shape[-2:], mode="bilinear", align_corners=False)
        middle = F.silu(self.up_middle(torch.cat((middle, coarse), dim=1)))
        middle = F.interpolate(middle, size=size, mode="bilinear", align_corners=False)
        correction = self.output(F.silu(self.up_fine(torch.cat((fine, middle), dim=1))))
        return 0.84 * history[:, -1] + correction


def encode_physical(state):
    return torch.cat((state[..., :1, :, :], state[..., 1:2, :, :].clamp_min(0).log1p(), state[..., 2:, :, :]), dim=-3)


def finalize_encoded(state):
    rain = F.softplus(state[..., 1:2, :, :], beta=3).clamp_max(math.log1p(400))
    return torch.cat((state[..., :1, :, :].clamp(233.15, 333.15), rain, state[..., 2:, :, :]), dim=-3)


def decode_physical(state, lead):
    # The held-out split showed increasing underdispersion in rain at longer leads.
    rain_gain = min(1.23, 1.06 + 0.023 * (lead - 1))
    rain = (state[..., 1:2, :, :].expm1() * rain_gain).clamp_max(400)
    return torch.cat((state[..., :1, :, :], rain, state[..., 2:, :, :]), dim=-3)


def static_features(static):
    lat = torch.as_tensor(LAT, device=static.device)[:, None].expand(len(LAT), len(LON))
    lon = torch.as_tensor(LON, device=static.device)[None, :].expand(len(LAT), len(LON))
    return torch.stack((2 * static[0] - 1, static[1] / 2500, (lat - 22) / 16, (lon - 83) / 18))


def advance_phase(phase, days):
    angle = 2 * math.pi * days / 365
    sine, cosine = phase.unbind(dim=-1)
    return torch.stack((sine * math.cos(angle) + cosine * math.sin(angle),
                        cosine * math.cos(angle) - sine * math.sin(angle)), dim=-1)


def climatology_at(coefficients, phase):
    sine, cosine = phase.unbind(dim=-1)
    basis = torch.stack((torch.ones_like(sine), sine, cosine, 2 * sine * cosine,
                         2 * cosine.square() - 1, (3 - 4 * sine.square()) * sine,
                         (4 * cosine.square() - 3) * cosine), dim=-1)
    return (basis @ coefficients.flatten(1)).reshape(*basis.shape[:-1], *coefficients.shape[1:])


def forecast_step(model, history, phase, climatology, scale, features):
    predicted_anomaly = model(history, features, phase)
    encoded = finalize_encoded(climatology + scale * predicted_anomaly)
    anomaly = (encoded - climatology) / scale
    return torch.cat((history[:, 1:], anomaly[:, None]), dim=1), encoded
