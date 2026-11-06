"""PRODUCTION VNEXT native six-tensor model, extracted without formula changes.
Legacy trainers are never imported. Fresh init positions are supplied by trainer.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from optimizer import Adan
from gsplat.rasterize_sum import rasterize_lp
from primix_v4_2_parameterization import q_for, exact_l2_norm


class RectLpBase(nn.Module):
    """Phase-15 CUDA screen-space Lp parameters generalized only to H != W."""

    def __init__(self, target, count, args):
        super().__init__()
        self.height, self.width, self.count, self.block = (
            target.shape[-2],
            target.shape[-1],
            count,
            16,
        )
        device = target.device
        xys = torch.rand(count, 2, device=device)
        xys[:, 0] *= self.width - 1
        xys[:, 1] *= self.height - 1
        self.xys = nn.Parameter(xys)
        self.scale_raw = nn.Parameter(
            torch.empty(count, 2, device=device).uniform_(0.2, 2.0)
        )
        self.rotation_angle = nn.Parameter(
            torch.empty(count, device=device).uniform_(-math.pi, math.pi)
        )
        self.opacity = nn.Parameter(torch.full((count,), 0.10, device=device))
        self.p = nn.Parameter(torch.full((count,), 2.0, device=device))
        with torch.no_grad():
            xs = self.xys[:, 0].round().long().clamp(0, self.width - 1)
            ys = self.xys[:, 1].round().long().clamp(0, self.height - 1)
            self.colors = nn.Parameter(
                target[0, :, ys, xs].transpose(0, 1).contiguous()
            )
        self.optimizer = Adan(
            [
                {"params": [self.xys], "lr": args.xyz_lr},
                {"params": [self.scale_raw], "lr": args.scale_lr},
                {"params": [self.rotation_angle], "lr": args.rotation_lr},
                {"params": [self.opacity], "lr": args.opacity_lr},
                {"params": [self.colors], "lr": args.color_lr},
                {"params": [self.p], "lr": args.p_lr},
            ]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20000, gamma=0.5
        )

    @property
    def scales(self):
        return F.softplus(self.scale_raw) + 0.25

    @property
    def rotations(self):
        return torch.stack(
            (self.rotation_angle.cos(), self.rotation_angle.sin()), dim=-1
        )

    def gaussian_equivalent_radii(self):
        return torch.ceil(3 * self.scales.detach().amax(dim=-1)).to(torch.int32)


class RectCudaPMN(RectLpBase):
    def __init__(self, target, count, args, gamma):
        super().__init__(target, count, args)
        self.gamma = gamma
        self.mass = self.opacity
        del self.opacity
        self.optimizer.param_groups[3]["params"] = [self.mass]

    def Z(self):
        p = self.p
        return (
            (2 ** (1 + 1 / p) * torch.exp(torch.lgamma(1 + 1 / p))).square()
            * self.scales[:, 0]
            * self.scales[:, 1]
        )

    def effective_opacity(self):
        return self.mass / self.Z().pow(self.gamma)

    def forward(self):
        depths = torch.zeros(self.count, device=self.xys.device)
        image = rasterize_lp(
            self.xys,
            depths,
            self.gaussian_equivalent_radii(),
            self.scales,
            self.rotations,
            self.colors,
            self.effective_opacity(),
            self.p,
            self.height,
            self.width,
            self.block,
            self.block,
            background=torch.zeros(3, device=self.xys.device),
            lp_mass_normalize=False,
        )
        return image.clamp(0, 1).permute(2, 0, 1).unsqueeze(0).contiguous()

    @torch.no_grad()
    def project_parameters_(self):
        self.xys[:, 0].clamp_(0, self.width - 1)
        self.xys[:, 1].clamp_(0, self.height - 1)
        self.mass.clamp_(1e-4, 1.0) if self.gamma == 0 else self.mass.clamp_(1e-4)
        self.p.clamp_(1.05, 2.0)


class V42PMN(RectCudaPMN):
    def __init__(self, target, count, args, mode):
        self.v42_mode = mode
        self.alpha_cutoff = float(getattr(args, "alpha_cutoff", 1.0 / 255.0))
        self.support_mode = getattr(args,'support_mode','alpha')
        super().__init__(target, count, args, 0.5)

    @property
    def nominal_scales(self):
        return F.softplus(self.scale_raw) + (0. if self.v42_mode=='continuous_l2_rms_free_scale' else .25)

    @property
    def scales(self):
        return self.nominal_scales * q_for(self.v42_mode, self.p)[:, None]

    @property
    def amplitude(self):
        return self.mass

    def effective_opacity(self):
        return self.amplitude / exact_l2_norm(self.p, self.scales)

    def forward(self):
        return self.render_unclipped().clamp(0,1)

    def render_unclipped(self):
        depths = torch.zeros(self.count, device=self.xys.device)
        image = rasterize_lp(
            self.xys,
            depths,
            self.gaussian_equivalent_radii(),
            self.scales,
            self.rotations,
            self.colors,
            self.effective_opacity(),
            self.p,
            self.height,
            self.width,
            self.block,
            self.block,
            background=torch.zeros(3, device=self.xys.device),
            lp_mass_normalize=False,
            alpha_cutoff=self.alpha_cutoff,
            support_mode=self.support_mode,
            p_min=float(getattr(self, "p_min", 1.05)),
            p_max=float(getattr(self, "p_max", 2.0)),
        )
        return image.permute(2, 0, 1).unsqueeze(0).contiguous()

    @torch.no_grad()
    def project_parameters_(self):
        self.xys[:, 0].clamp_(0, self.width - 1)
        self.xys[:, 1].clamp_(0, self.height - 1)
        self.mass.clamp_(1e-6, 1e4)
        if self.v42_mode=='continuous_l2_rms_free_scale':self.scale_raw.clamp_(min=-15.)
        self.p.clamp_(
            float(getattr(self, "p_min", 1.05)), float(getattr(self, "p_max", 2.0))
        )
