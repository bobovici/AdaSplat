"""Differentiable losses used by AdaSplat V4 structure-first maturation."""
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class MaturationPhase(str, Enum):
    STRUCTURE = "structure"
    APPEARANCE = "appearance"
    JOINT = "joint"
    MATURE = "mature"


def _luminance(image):
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("expected [B,3,H,W]")
    weights = image.new_tensor((.2126, .7152, .0722)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def sobel_gradients(image):
    value = _luminance(image)
    kx = value.new_tensor(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.))).view(1, 1, 3, 3) / 8.
    ky = kx.transpose(-1, -2)
    return F.conv2d(value, kx, padding=1), F.conv2d(value, ky, padding=1)


def differentiable_ssim(x, y, window=7, data_range=1.0):
    if int(window) < 3 or int(window) % 2 != 1:
        raise ValueError("SSIM window must be odd and >= 3")
    pad = int(window) // 2
    mean_x = F.avg_pool2d(x, window, 1, pad)
    mean_y = F.avg_pool2d(y, window, 1, pad)
    var_x = F.avg_pool2d(x * x, window, 1, pad) - mean_x.square()
    var_y = F.avg_pool2d(y * y, window, 1, pad) - mean_y.square()
    cov = F.avg_pool2d(x * y, window, 1, pad) - mean_x * mean_y
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    value = ((2 * mean_x * mean_y + c1) * (2 * cov + c2) /
             ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)).clamp_min(1e-12))
    return value.mean()


@dataclass(frozen=True)
class StructureWeights:
    ssim: float = 1.0
    gradient: float = .2
    edge: float = .1
    mse: float = .1
    joint_ssim: float = .1
    joint_gradient: float = .02


class StructureObjective:
    def __init__(self, weights=None, ssim_window=7):
        self.weights = weights or StructureWeights()
        self.ssim_window = int(ssim_window)

    def components(self, prediction, target):
        mse = F.mse_loss(prediction.float(), target.float())
        ssim_loss = 1.0 - differentiable_ssim(prediction.float(), target.float(), self.ssim_window)
        px, py = sobel_gradients(prediction.float())
        tx, ty = sobel_gradients(target.float())
        gradient = (px - tx).abs().mean() + (py - ty).abs().mean()
        edge_weight = torch.sqrt(tx.square() + ty.square() + 1e-12)
        edge_weight = (edge_weight / edge_weight.mean().clamp_min(1e-8)).clamp(0, 4)
        edge = (edge_weight * (_luminance(prediction.float()) - _luminance(target.float())).abs()).mean()
        return {"mse": mse, "ssim": ssim_loss, "gradient": gradient, "edge": edge}

    def __call__(self, prediction, target, phase):
        phase = MaturationPhase(phase)
        values = self.components(prediction, target)
        w = self.weights
        if phase == MaturationPhase.STRUCTURE:
            total = w.ssim * values["ssim"] + w.gradient * values["gradient"] + w.edge * values["edge"] + w.mse * values["mse"]
        elif phase == MaturationPhase.APPEARANCE:
            total = values["mse"]
        else:
            total = values["mse"] + w.joint_ssim * values["ssim"] + w.joint_gradient * values["gradient"]
        values["total"] = total
        return total, values
