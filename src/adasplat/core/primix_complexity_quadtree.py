"""Handcrafted Complexity Quadtree used only for initial PMN placement.

This module has no learned-model, teacher, PMN, or residual-growth dependency.
Its output is a fixed-size set of initial pixel-space coordinates.
"""
import math

import torch
import torch.nn.functional as F


def _unit_normalize(value, epsilon):
    value = value.clamp_min(0)
    value = value - value.min()
    return value / value.max().clamp_min(float(epsilon))


@torch.no_grad()
def compute_complexity_map(image, variance_window=5, epsilon=1e-8):
    """Return gradient-normalized + variance-normalized luminance complexity."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("image must be [3,H,W] or [1,3,H,W]")
    if int(variance_window) < 1 or int(variance_window) % 2 == 0:
        raise ValueError("variance_window must be a positive odd integer")
    value = image.float()
    luminance = (.299 * value[:, 0:1] + .587 * value[:, 1:2] +
                 .114 * value[:, 2:3])
    gx = F.pad(luminance[..., :, 1:] - luminance[..., :, :-1], (0, 1, 0, 0)).abs()
    gy = F.pad(luminance[..., 1:, :] - luminance[..., :-1, :], (0, 0, 0, 1)).abs()
    gradient = torch.sqrt(gx.square() + gy.square() + float(epsilon))
    radius = int(variance_window) // 2
    padded = F.pad(luminance, (radius, radius, radius, radius), mode="replicate")
    local_mean = F.avg_pool2d(padded, int(variance_window), stride=1)
    local_square_mean = F.avg_pool2d(padded.square(), int(variance_window), stride=1)
    variance = (local_square_mean - local_mean.square()).clamp_min(0)
    score = (_unit_normalize(gradient[0, 0], epsilon) +
             _unit_normalize(variance[0, 0], epsilon) + float(epsilon))
    if not torch.isfinite(score).all():
        raise RuntimeError("complexity map contains non-finite values")
    return score


@torch.no_grad()
def build_complexity_quadtree(complexity, min_depth=3, max_depth=5,
                              split_factor=1.5, min_size=32):
    """Partition a complexity map using a global-relative mean score rule."""
    if complexity.ndim != 2:
        raise ValueError("complexity must be [H,W]")
    if int(min_depth) < 0 or int(max_depth) < int(min_depth):
        raise ValueError("complexity quadtree depths are invalid")
    if float(split_factor) <= 0 or int(min_size) < 1:
        raise ValueError("complexity split factor and minimum size must be positive")
    height, width = complexity.shape
    global_mean = float(complexity.mean().item())
    threshold = global_mean * float(split_factor)
    leaves = []

    def visit(x0, y0, x1, y1, depth):
        patch = complexity[y0:y1, x0:x1]
        mean = float(patch.mean().item())
        w, h = x1 - x0, y1 - y0
        splittable = (depth < int(max_depth) and w > int(min_size) and
                      h > int(min_size))
        should_split = depth < int(min_depth) or mean > threshold
        if splittable and should_split:
            xm, ym = x0 + w // 2, y0 + h // 2
            if x0 < xm < x1 and y0 < ym < y1:
                visit(x0, y0, xm, ym, depth + 1)
                visit(xm, y0, x1, ym, depth + 1)
                visit(x0, ym, xm, y1, depth + 1)
                visit(xm, ym, x1, y1, depth + 1)
                return
        leaves.append({"leaf_id": len(leaves), "x0": int(x0), "x1": int(x1),
                       "y0": int(y0), "y1": int(y1), "depth": int(depth),
                       "area": int(w * h), "mean_complexity": mean})

    visit(0, 0, int(width), int(height), 0)
    return leaves, {"global_mean_complexity": global_mean,
                    "split_threshold": threshold}


def allocate_initial_budget(leaves, count, min_per_leaf=1, epsilon=1e-8):
    """Largest-remainder exact-N allocation with a feasible leaf minimum."""
    total = int(count)
    minimum = int(min_per_leaf)
    if total < 1 or minimum < 0 or not leaves:
        raise ValueError("initial allocation inputs are invalid")
    weights = [max(float(epsilon), float(leaf["area"]) *
                   (float(leaf["mean_complexity"]) + float(epsilon)))
               for leaf in leaves]
    allocations = [0] * len(leaves)
    if total >= len(leaves) * minimum:
        allocations = [minimum] * len(leaves)
        remaining = total - len(leaves) * minimum
    else:
        # The requested minimum is infeasible. Allocate the complete budget by
        # score without failing; zero-allocation leaves are explicitly retained.
        remaining = total
    weight_sum = sum(weights)
    raw = [remaining * weight / weight_sum for weight in weights]
    floors = [int(math.floor(value)) for value in raw]
    allocations = [base + floor for base, floor in zip(allocations, floors)]
    remainder = total - sum(allocations)
    order = sorted(range(len(leaves)),
                   key=lambda index: (raw[index] - floors[index], -index),
                   reverse=True)
    for index in order[:remainder]:
        allocations[index] += 1
    if sum(allocations) != total:
        raise RuntimeError("complexity initial allocation is not exact-N")
    rows = []
    for leaf, weight, allocated in zip(leaves, weights, allocations):
        row = dict(leaf); row.update(weight=float(weight), allocated_N=int(allocated))
        rows.append(row)
    return allocations, rows


@torch.no_grad()
def sample_initial_points(complexity, leaves, allocations, beta=1.0, seed=0,
                          epsilon=1e-8):
    """Sample exact-N subpixel positions within independent complexity leaves."""
    if complexity.ndim != 2 or len(leaves) != len(allocations):
        raise ValueError("complexity sampling inputs are inconsistent")
    if float(beta) < 0:
        raise ValueError("complexity pixel beta must be nonnegative")
    generator = torch.Generator(device=complexity.device)
    generator.manual_seed(int(seed))
    points = []
    for leaf, count in zip(leaves, allocations):
        count = int(count)
        if count <= 0:
            continue
        patch = complexity[leaf["y0"]:leaf["y1"], leaf["x0"]:leaf["x1"]]
        probability = (patch.reshape(-1).clamp_min(0) + float(epsilon)).pow(float(beta))
        probability = probability / probability.sum().clamp_min(float(epsilon))
        selected = torch.multinomial(probability, count, replacement=True,
                                     generator=generator)
        patch_width = int(leaf["x1"] - leaf["x0"])
        x = (selected % patch_width).float() + int(leaf["x0"])
        y = torch.div(selected, patch_width, rounding_mode="floor").float() + int(leaf["y0"])
        jitter = .05 + .90 * torch.rand((count, 2), device=complexity.device,
                                        generator=generator)
        points.append(torch.stack((x, y), dim=1) + jitter)
    result = torch.cat(points, dim=0) if points else torch.empty((0, 2), device=complexity.device)
    if len(result) != sum(int(value) for value in allocations):
        raise RuntimeError("complexity initial sampling is not exact-N")
    return result


@torch.no_grad()
def complexity_quadtree_initialization(image, count, seed=0, min_depth=3,
                                       max_depth=5, split_factor=1.5,
                                       min_size=32, min_per_leaf=1,
                                       pixel_beta=1.0):
    """Pure handcrafted initialization entry point used by the trainer."""
    complexity = compute_complexity_map(image)
    leaves, tree = build_complexity_quadtree(complexity, min_depth, max_depth,
                                              split_factor, min_size)
    allocations, rows = allocate_initial_budget(leaves, count, min_per_leaf)
    points = sample_initial_points(complexity, leaves, allocations, pixel_beta, seed)
    return points, complexity, rows, tree
