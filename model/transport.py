"""Wind-driven transport for localized counterfactual weather anomalies."""

import torch
from torch.nn import functional as F

from model.unet import LAT


def transport_response(response, wind):
    """Advance a perturbation one day with semi-Lagrangian advection and diffusion."""
    height, width = response.shape[-2:]
    options = {"dtype": response.dtype, "device": response.device}
    rows = torch.linspace(-1, 1, height, **options)
    cols = torch.linspace(-1, 1, width, **options)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    cos_lat = torch.as_tensor(LAT, **options).deg2rad().cos()[None, :, None]

    # One m/s sustained for a day is about 0.78 degrees of latitude. Use a
    # damped displacement because 10 m wind only approximates column transport.
    east = 0.58 * wind[:, 0] / cos_lat
    north = 0.58 * wind[:, 1]
    limit = 3 / torch.sqrt(east.square() + north.square()).clamp_min(3)
    departure = torch.stack((grid_x - east * limit * 2 / (width - 1),
                             grid_y - north * limit * 2 / (height - 1)), dim=-1)
    moved = F.grid_sample(response, departure, align_corners=True, padding_mode="border")
    # The symmetric term represents unresolved lateral mixing even in calm flow.
    mixed = F.avg_pool2d(F.pad(response, (4, 4, 4, 4), mode="replicate"), kernel_size=9, stride=1)
    return 0.99 * (0.40 * moved + 0.60 * mixed)


def advance_adjustment(field, tendency):
    """Advance a damped lateral pressure and momentum adjustment in four substeps."""
    for _ in range(4):
        edge = F.pad(field, (1, 1, 1, 1), mode="replicate")
        laplacian = (
            edge[..., :-2, 1:-1] + edge[..., 2:, 1:-1]
            + edge[..., 1:-1, :-2] + edge[..., 1:-1, 2:] - 4 * field
        )
        tendency = 0.985 * (tendency + 0.20 * laplacian)
        field = field + tendency
    return field, tendency
