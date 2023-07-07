import torch
import math
import torch.nn.functional as F
from torch import nn


class PolarWarp(nn.Module):
    def __init__(self, radius, mode='bilinear'):
        """
        Args:
            radius (float): The radius to use for the warping operation
            mode (string): The interpolation mode
        """
        super().__init__()
        self.radius = radius
        self.mode = mode
        self.base_x_idxs, self.base_y_idxs = get_base_polar_flow_field(radius)

    def forward(self, t, center):
        """
        Apply a polar warp to the input tensor.

        Args:
            t (torch.Tensor): The batch of images to warp $(N, C, H, W)$, or
                the image itself $(C, H, W)$.
            center (list): List with length 2. The first element contains the
                $N$ x-coordinates of the respective centers to warp around. The
                second element contains the $N$ y-coordinates. If the input
                tensor has shape $(C, H, W)$, center should be a tuple with the
                (x, y)-coordinate.
        """
        if t.ndim == 3:
            in_t = t[None, ...]
            in_center = [
                torch.tensor([center[0]]).type_as(in_t),
                torch.tensor([center[1]]).type_as(in_t)
            ]
        elif t.ndim == 4:
            if len(center[0]) != len(center[1]):
                raise ValueError(
                    'The number of x and y coordinates should be equal.'
                )
            if len(t) != len(center[0]):
                raise ValueError(
                    'The number of tensor channels should be equal to the '
                    'number of x and y coordinates'
                )
            in_t = t
            in_center = center
        else:
            raise ValueError('Expected input tensor with 3 or 4 dims, '
                             f'but got {t.ndim} dims')

        x_idxs, y_idxs = self.get_rel_polar_flow_field(t.shape[-2:], in_center)
        grid = torch.cat([x_idxs[..., None],
                          y_idxs[..., None]],
                         dim=-1).type_as(in_t)
        out_t = F.grid_sample(in_t, grid, mode=self.mode,
                              align_corners=False)

        return out_t if t.ndim == 4 else out_t[0, ...]

    def get_abs_polar_flow_field(self, centers: list):
        """
        Return the absolute x and y indices necessary for indexing an image
        to obtain a polar warp transform.

        Args:
            centers (list): List of (x, y)-coordinates of centers to warp
                around.
            radius (float): The radius to use for the warping operation
        """
        self.base_x_idxs = self.base_x_idxs.type_as(centers[0])
        self.base_y_idxs = self.base_y_idxs.type_as(centers[1])

        x_idxs = (
            self.base_x_idxs
            .expand(centers[0].shape[0], -1, -1)
            + centers[0][:, None, None]
        )
        y_idxs = (
            self.base_y_idxs
            .expand(centers[1].shape[0], -1, -1)
            + centers[1][:, None, None]
        )
        return x_idxs, y_idxs

    def get_rel_polar_flow_field(self, input_shape: tuple, centers: list):
        """
        Return the relative x and y indices ((-1, -1) if top-left of the input
        image, (1, 1) is bottom-right) necessary for indexing an image to
        obtain a polar warp transform.

        Args:
            input_shape (tuple): The (height, width) of the batch
            centers (list): List with length 2. The first element contains the
                    x-coordinates of the centers to warp around. The second
                    element contains the y-coordinates.
        """
        x_idxs, y_idxs = self.get_abs_polar_flow_field(centers)
        x_idxs = (x_idxs / input_shape[-1]) * 2 - 1
        y_idxs = (y_idxs / input_shape[-2]) * 2 - 1
        return x_idxs, y_idxs


def get_base_polar_flow_field(radius: float):
    out_height = round(radius)
    out_width = round(radius * math.pi)

    rhos = torch.arange(0, radius, radius / out_height)[None, :]
    phis = torch.arange(0, math.tau, math.tau / out_width)[None, :]

    x_idxs = torch.mm(rhos.T, torch.cos(phis))[None, ...]
    y_idxs = torch.mm(rhos.T, torch.sin(phis))[None, ...]

    return x_idxs, y_idxs
