"""Python bindings for binning and sorting gaussians"""

from typing import Optional, Tuple

from jaxtyping import Float, Int
from torch import Tensor
import torch

import gsplat.cuda as _C


def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _C.map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
    )
    return (isect_ids, gaussian_ids)


def get_tile_bin_edges(
    num_intersects: int, isect_ids_sorted: Int[Tensor, "num_intersects 1"],
    num_tiles: Optional[int] = None,
) -> Int[Tensor, "num_tiles 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): one half-open intersection range per global
          tile. Empty tiles are ``[0, 0]``.
    """
    if num_tiles is None:
        # Compatibility for callers that do not know the complete tile grid.
        # The sorted last id provides a safe lower bound for allocation.
        num_tiles = 0 if num_intersects == 0 else int((isect_ids_sorted[-1] >> 32).item()) + 1
    if num_tiles < 0:
        raise ValueError("num_tiles must be non-negative")
    return _C.get_tile_bin_edges(num_intersects, num_tiles, isect_ids_sorted.contiguous())


def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _C.compute_cov2d_bounds(num_pts, cov2d.contiguous())


def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def compute_num_tiles_hit(
    xys: Float[Tensor, "batch 2"],
    radii: Int[Tensor, "batch 1"],
    img_height: int,
    img_width: int,
    block_h: int,
    block_w: int,
) -> Int[Tensor, "batch 1"]:
    """Return the number of raster tiles covered by each 2D Gaussian.

    This is the Python equivalent of ``get_tile_bbox`` in ``helpers.cuh``:
    tile minima are inclusive, maxima are exclusive, and both are clamped to
    the image's tile grid.  Gaussians with non-positive radii produce zero
    intersections, matching the CUDA intersection-mapping kernel.
    """
    tile_bounds_x = (img_width + block_w - 1) // block_w
    tile_bounds_y = (img_height + block_h - 1) // block_h
    radius = radii.to(dtype=xys.dtype)
    tile_center_x = xys[:, 0] / float(block_w)
    tile_center_y = xys[:, 1] / float(block_h)
    tile_radius_x = radius / float(block_w)
    tile_radius_y = radius / float(block_h)

    tile_min_x = torch.trunc(tile_center_x - tile_radius_x).to(torch.int32)
    tile_min_y = torch.trunc(tile_center_y - tile_radius_y).to(torch.int32)
    tile_max_x = torch.trunc(tile_center_x + tile_radius_x + 1.0).to(torch.int32)
    tile_max_y = torch.trunc(tile_center_y + tile_radius_y + 1.0).to(torch.int32)
    tile_min_x = tile_min_x.clamp(0, tile_bounds_x)
    tile_min_y = tile_min_y.clamp(0, tile_bounds_y)
    tile_max_x = tile_max_x.clamp(0, tile_bounds_x)
    tile_max_y = tile_max_y.clamp(0, tile_bounds_y)

    tile_area = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y)
    return torch.where(radii > 0, tile_area, torch.zeros_like(tile_area))


def compute_lp_radii(
    gaussian_radii: Int[Tensor, "batch 1"],
    scales: Float[Tensor, "batch 2"],
    rotations: Float[Tensor, "batch 2"],
    p: Float[Tensor, "batch 1"],
    opacity: Float[Tensor, "batch 1"],
    alpha_min: float = 1.0 / 255.0,
    p_min: float = 1.05,
    p_max: float = 2.0,
) -> Int[Tensor, "batch 1"]:
    """Return conservative Lp tile radii at the rasterizer alpha cutoff.

    ``gaussian_radii`` is retained only as an existing validity sentinel.  At
    every valid ``p`` (including ``p=2``), the support is derived from the
    same renderer contribution condition ``opacity * K >= alpha_min``.  This
    computes the axis-aligned extrema of the rotated Lp support and then uses
    their maximum as the scalar tile radius expected by the existing binning
    implementation.

    This is intentionally non-differentiable: tile assignment was already
    non-differentiable for Gaussian radii.
    """
    if scales.ndim != 2 or scales.shape[-1] != 2:
        raise ValueError("scales must have shape (N, 2)")
    if rotations.ndim != 2 or rotations.shape[-1] != 2:
        raise ValueError("rotations must have shape (N, 2)")
    if scales.shape[0] != gaussian_radii.numel():
        raise ValueError("gaussian_radii and scales must have matching batch size")

    with torch.no_grad():
        # The CUDA Lp kernel accepts finite p>0.  Do not silently use a
        # p=2 support box when an upper-bound diagnostic renders p>2.
        p_value = p.reshape(-1).detach().to(scales.dtype).clamp(float(p_min), float(p_max))
        opacity_value = opacity.reshape(-1).detach().to(scales.dtype)
        valid = (gaussian_radii > 0) & (opacity_value > alpha_min)
        # alpha = opacity * exp(-0.5 * ||u||_p^p) >= alpha_min.
        support_power = (-2.0 * torch.log(alpha_min / opacity_value.clamp_min(alpha_min))).clamp_min(0.0)
        q = support_power.pow(1.0 / p_value)
        convex = p_value > 1.0
        # For 0<p<=1 the linear extrema are at the axis vertices, hence max.
        # Evaluate a harmless dual exponent for the inactive nonconvex branch.
        dual_p = torch.where(convex, p_value / (p_value - 1.0).clamp_min(1e-7), torch.ones_like(p_value))
        sx, sy = scales.detach().abs().unbind(dim=-1)
        cosine, sine = rotations.detach().abs().unbind(dim=-1)
        def stable_dual_norm(a, b):
            # Near p=1 the dual exponent is large. Unscaled powers overflow
            # in FP32 and can turn a valid support into a negative int radius.
            maximum = torch.maximum(a, b)
            denominator = maximum.clamp_min(torch.finfo(a.dtype).tiny)
            convex_extent = maximum * ((a / denominator).pow(dual_p) +
                              (b / denominator).pow(dual_p)).pow(1.0 / dual_p)
            return torch.where(convex, convex_extent, maximum)
        extent_x = q * stable_dual_norm(cosine * sx, sine * sy)
        extent_y = q * stable_dual_norm(sine * sx, cosine * sy)
        # Avoid wrapping enormous support radii into negative int32 values.
        # Maximum below INT_MAX remains exactly representable in FP32.
        lp_radii = torch.ceil(torch.maximum(extent_x, extent_y)).clamp(max=2147483520.).to(torch.int32)
        return torch.where(valid, lp_radii, torch.zeros_like(lp_radii))


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points, num_intersects, xys, depths, radii, cum_tiles_hit, tile_bounds
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    num_tiles = tile_bounds[0] * tile_bounds[1] * tile_bounds[2]
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, num_tiles=num_tiles)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins
