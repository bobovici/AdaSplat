"""Gaussian-sum and Lp CUDA rasterizer Python interfaces."""

import math
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C
from .utils import bin_and_sort_gaussians, compute_cumulative_intersects, compute_lp_radii, compute_num_tiles_hit


def lp_mass_normalization_factor(p: Tensor) -> Tensor:
    """Return ``I(2) / I(p)`` for the unnormalised 2-D Lp kernel.

    ``I(p) = [2 ** (1 + 1 / p) * Gamma(1 + 1 / p)] ** 2`` is the
    integral of ``exp(-0.5 * (|u|**p + |v|**p))`` in normalized kernel
    coordinates.  The computation is deliberately a regular PyTorch graph:
    its derivative with respect to ``p`` is composed with the CUDA kernel's
    native Lp derivative by autograd.
    """
    p = p.reshape(-1)
    log_i_p = 2.0 * ((1.0 + 1.0 / p) * math.log(2.0) + torch.lgamma(1.0 + 1.0 / p))
    log_i_2 = math.log(2.0 * math.pi)
    return torch.exp(log_i_2 - log_i_p)


def rasterize_gaussians_sum(
    xys: Float[Tensor, "*batch 2"], depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"], conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"], colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"], img_height: int, img_width: int,
    BLOCK_H: int = 16, BLOCK_W: int = 16,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: bool = False, p: Optional[Tensor] = None,
) -> Tensor:
    """Original Gaussian/conic rasterizer entry point."""
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255.0
    if colors.shape[-1] != 3:
        raise ValueError("rasterize_gaussians_sum requires RGB colors with shape (N, 3)")
    if background is None:
        background = torch.ones(3, dtype=torch.float32, device=colors.device)
    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    # Existing baseline bindings retain an unused p pointer for ABI compatibility.
    if p is None:
        p = torch.zeros(xys.size(0), dtype=xys.dtype, device=xys.device)
    return _RasterizeGaussiansSum.apply(
        xys.contiguous(), depths.contiguous(), radii.contiguous(), conics.contiguous(),
        num_tiles_hit.contiguous(), colors.contiguous(), opacity.contiguous(),
        img_height, img_width, BLOCK_H, BLOCK_W, background.contiguous(), return_alpha,
        p.contiguous(),
    )


class _RasterizeGaussiansSum(Function):
    @staticmethod
    def forward(
        ctx, xys, depths, radii, conics, num_tiles_hit, colors, opacity,
        img_height, img_width, BLOCK_H, BLOCK_W, background, return_alpha, p,
    ):
        tile_bounds = (
            (img_width + BLOCK_W - 1) // BLOCK_W,
            (img_height + BLOCK_H - 1) // BLOCK_H,
            1,
        )
        num_intersects, cumulative = compute_cumulative_intersects(num_tiles_hit)
        if num_intersects < 1:
            out_img = torch.ones(img_height, img_width, 3, device=xys.device) * background
            gaussian_ids_sorted = torch.zeros(0, dtype=torch.int32, device=xys.device)
            tile_bins = torch.zeros(tile_bounds[0] * tile_bounds[1] * tile_bounds[2], 2, dtype=torch.int32, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, dtype=torch.int32, device=xys.device)
        else:
            _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
                xys.size(0), num_intersects, xys, depths, radii, cumulative, tile_bounds
            )
            out_img, final_Ts, final_idx = _C.rasterize_sum_forward(
                tile_bounds, (BLOCK_W, BLOCK_H, 1), (img_width, img_height, 1),
                gaussian_ids_sorted, tile_bins, xys, conics, colors, opacity, background, p,
            )
        ctx.img_height, ctx.img_width = img_height, img_width
        ctx.BLOCK_H, ctx.BLOCK_W, ctx.num_intersects = BLOCK_H, BLOCK_W, num_intersects
        ctx.save_for_backward(
            gaussian_ids_sorted, tile_bins, xys, conics, colors, opacity,
            background, final_Ts, final_idx, p,
        )
        return (out_img, 1 - final_Ts) if return_alpha else out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])
        (ids, tile_bins, xys, conics, colors, opacity, background, final_Ts, final_idx, p) = ctx.saved_tensors
        if ctx.num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
        else:
            v_xy, v_conic, v_colors, v_opacity = _C.rasterize_sum_backward(
                ctx.img_height, ctx.img_width, ctx.BLOCK_H, ctx.BLOCK_W,
                ids, tile_bins, xys, conics, colors, opacity, background,
                final_Ts, final_idx, v_out_img.contiguous(), v_out_alpha.contiguous(), p,
            )
            v_opacity = v_opacity.reshape(opacity.shape)
        return (
            v_xy, None, None, v_conic, None, v_colors, v_opacity,
            None, None, None, None, None, None, None,
        )


def rasterize_lp(
    xys: Tensor, depths: Tensor, radii: Tensor, scales: Tensor, rotations: Tensor,
    colors: Tensor, opacity: Tensor, p: Tensor, img_height: int, img_width: int,
    BLOCK_H: int = 16, BLOCK_W: int = 16, background: Optional[Tensor] = None,
    return_alpha: bool = False, lp_mass_normalize: bool = False,
    alpha_cutoff: float = 1.0 / 255.0,
    p_min: float = 1.05,
    p_max: float = 2.0,
    support_mode: str = "alpha",
    affine_color: bool = False,
) -> Tensor:
    """Lp-kernel rasterizer entry point.

    When ``lp_mass_normalize`` is true, the effective opacity passed to the
    CUDA kernel is multiplied by ``I(2) / I(p)``.  This keeps the p=2 path
    byte-for-byte equivalent while normalizing the kernel mass in its local
    coordinates.  CUDA itself remains unchanged.
    """
    alpha_cutoff = float(alpha_cutoff)
    if not (0.0 < alpha_cutoff < 1.0):
        raise ValueError("alpha_cutoff must be finite and in (0, 1)")
    p_min, p_max = float(p_min), float(p_max)
    if not (math.isfinite(p_min) and math.isfinite(p_max) and 0.1 <= p_min <= p_max <= 8):
        raise ValueError("p range must be finite and satisfy 0.1 <= p_min <= p_max <= 8")
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255.0
    if background is None:
        background = torch.ones(colors.shape[-1], dtype=torch.float32, device=colors.device)
    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, C)")
    if affine_color:
        if support_mode != 'kernel' or colors.shape[1] != 9 or background.numel() != 3:
            raise ValueError('affine mode requires unit kernel, 9 coefficients and RGB background')
    elif colors.shape[1] != background.numel():
        raise ValueError('background channels must match colors')
    if lp_mass_normalize:
        # Keep this outside the custom Function.  Its backward result for the
        # effective opacity then composes through this product and contributes
        # the normalization-factor term to dL/dp automatically.
        opacity = opacity * lp_mass_normalization_factor(p).reshape_as(opacity)
    if support_mode not in ('alpha','kernel'):
        raise ValueError('support_mode must be alpha or kernel')
    if support_mode == 'kernel':
        # The basis has unit peak; its signed RGB coefficient remains outside
        # support selection and alpha saturation, with autograd through L2 norm.
        colors = colors * opacity.reshape(-1,1)
        opacity = torch.ones_like(opacity)
    return _RasterizeLp.apply(
        xys.contiguous(), depths.contiguous(), radii.contiguous(),
        scales.reshape(-1, 2).contiguous(), rotations.reshape(-1, 2).contiguous(),
        p.reshape(-1).contiguous(), colors.contiguous(), opacity.contiguous(),
        img_height, img_width, BLOCK_H, BLOCK_W, background.contiguous(), return_alpha,
        alpha_cutoff, p_min, p_max,
    )


class _RasterizeLp(Function):
    @staticmethod
    def forward(
        ctx, xys, depths, radii, scales, rotations, p, colors, opacity,
        img_height, img_width, BLOCK_H, BLOCK_W, background, return_alpha,
        alpha_cutoff,
        p_min, p_max,
    ):
        tile_bounds = (
            (img_width + BLOCK_W - 1) // BLOCK_W,
            (img_height + BLOCK_H - 1) // BLOCK_H,
            1,
        )
        lp_radii = compute_lp_radii(radii, scales, rotations, p, opacity, alpha_cutoff, p_min, p_max)
        num_tiles_hit = compute_num_tiles_hit(xys, lp_radii, img_height, img_width, BLOCK_H, BLOCK_W)
        num_intersects, cumulative = compute_cumulative_intersects(num_tiles_hit)
        if num_intersects < 1:
            out_img = torch.ones(img_height, img_width, background.numel(), device=xys.device) * background
            gaussian_ids_sorted = torch.zeros(0, dtype=torch.int32, device=xys.device)
            tile_bins = torch.zeros(tile_bounds[0] * tile_bounds[1] * tile_bounds[2], 2, dtype=torch.int32, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, dtype=torch.int32, device=xys.device)
        else:
            _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(
                xys.size(0), num_intersects, xys, depths, lp_radii, cumulative, tile_bounds
            )
            out_img, final_Ts, final_idx = _C.nd_rasterize_lp_forward(
                tile_bounds, (BLOCK_W, BLOCK_H, 1), (img_width, img_height, 1),
                gaussian_ids_sorted, tile_bins, xys, scales, rotations, p,
                colors, opacity, background, alpha_cutoff, p_min, p_max,
            )
        ctx.img_height, ctx.img_width = img_height, img_width
        ctx.BLOCK_H, ctx.BLOCK_W, ctx.num_intersects = BLOCK_H, BLOCK_W, num_intersects
        ctx.alpha_cutoff = alpha_cutoff
        ctx.p_min, ctx.p_max = p_min, p_max
        ctx.save_for_backward(
            gaussian_ids_sorted, tile_bins, xys, scales, rotations, colors,
            opacity, background, final_Ts, final_idx, p,
        )
        return (out_img, 1 - final_Ts) if return_alpha else out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])
        (ids, tile_bins, xys, scales, rotations, colors, opacity, background, final_Ts, final_idx, p) = ctx.saved_tensors
        if ctx.num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_scale = torch.zeros_like(scales)
            v_rotation = torch.zeros_like(rotations)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)
            v_p = torch.zeros_like(p)
        else:
            v_xy, v_scale, v_rotation, v_colors, v_opacity, v_p = _C.nd_rasterize_lp_backward(
                ctx.img_height, ctx.img_width, ctx.BLOCK_H, ctx.BLOCK_W,
                ids, tile_bins, xys, scales, rotations, colors, opacity, background,
                final_Ts, final_idx, v_out_img.contiguous(), v_out_alpha.contiguous(), p,
                ctx.alpha_cutoff, ctx.p_min, ctx.p_max,
            )
            v_opacity = v_opacity.reshape(opacity.shape)
        return (
            v_xy, None, None, v_scale, v_rotation, v_p, v_colors, v_opacity,
            None, None, None, None, None, None, None, None, None,
        )
