"""PRODUCTION VNEXT frozen distortion and acceptance arithmetic."""
import math
import torch
from adasplat_vnext_config import DEFAULTS, settings


def unified_loss(runtime, target):
    cfg = settings(runtime)
    image = runtime.model()
    return reconstruction_loss(image, target, cfg['loss'],step=runtime.metadata['clock']['logical_iteration'])


def reconstruction_loss(image, target, config, step=0):
    """Target-relative structure fitting. Zero weights bypass extra graphs.

    First differences penalize reconstruction errors, preserving true edges.
    This is not image smoothing or a hard constraint on primitive geometry.
    """
    if config.get('structure_mode') == 'coverage_multiscale':
        from adasplat_multiscale_objective import multiscale_reconstruction_loss
        return multiscale_reconstruction_loss(image,target,config,step)
    loss = (image-target).square().mean()
    patch=config.get('structure_patch_size',0)
    if patch and (image.shape[-2]>patch or image.shape[-1]>patch):
        h,w=image.shape[-2:];ph,pw=min(patch,h),min(patch,w)
        # Counter-based local PRNG: A/B use the same old-clock patches; warmup
        # does not consume global RNG or advance the old optimization clock.
        import random
        generator=random.Random((int(config.get('structure_seed',20261002))<<32)+int(step))
        y=generator.randrange(h-ph+1);x=generator.randrange(w-pw+1)
        image=image[...,y:y+ph,x:x+pw].contiguous()
        target=target[...,y:y+ph,x:x+pw].contiguous()
    if config['ssim_weight']:
        if config.get('ssim_gaussian', False):
            from pytorch_msssim import ssim
            structural = ssim(image, target, data_range=1, win_size=config['ssim_window'], win_sigma=1.5)
        else:
            from primix_structure_objective import differentiable_ssim
            structural = differentiable_ssim(image, target, window=config['ssim_window'])
        loss = loss + config['ssim_weight'] * (1-structural)
    weight = config.get('gradient_weight', 0.0)
    if weight:
        dx = (image[...,1:]-image[...,:-1])-(target[...,1:]-target[...,:-1])
        dy = (image[...,1:,:]-image[...,:-1,:])-(target[...,1:,:]-target[...,:-1,:])
        loss = loss + weight*(dx.square().mean()+dy.square().mean())
    return loss*config.get('scale',1.0)


@torch.no_grad()
def control_metrics(image, target, config=None):
    cfg = config or DEFAULTS
    ev = cfg["evaluation"]
    size = ev["core_size"]
    from pytorch_msssim import ms_ssim, ssim
    from primix_structure_objective import differentiable_ssim

    error = (image - target).square().mean(1, keepdim=True)
    h, w = error.shape[-2:]
    sums = (
        torch.nn.functional.avg_pool2d(
            error, size, size, ceil_mode=True, divisor_override=1
        )
        .flatten()
        .cpu()
        .double()
    )
    areas = torch.tensor(
        [
            min(size, h - y) * min(size, w - x)
            for y in range(0, h, size)
            for x in range(0, w, size)
        ],
        dtype=torch.float64,
    )
    values = sums / areas
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    remaining = ev["tail_area"] * h * w
    tail = 0.0
    for i in order:
        weight = min(remaining, float(areas[i]))
        tail += weight * float(values[i])
        remaining -= weight
        if remaining <= 0:
            break
    tail /= ev["tail_area"] * h * w
    mse = float((image - target).square().mean())
    metrics = dict(
        MSE=mse,
        PSNR=-10 * math.log10(max(mse, 1e-20)),
        D_tail=tail,
        D_ctl=ev["mse_weight"] * mse + ev["tail_weight"] * tail,
        loss=float(reconstruction_loss(image,target,cfg['loss'])),
        SSIM=float(ssim(image,target,data_range=1,size_average=True,
                        win_size=ev['ms_window'],win_sigma=ev['ms_sigma'],K=tuple(ev['ms_K']))),
        MS_SSIM=float(
            ms_ssim(
                image,
                target,
                data_range=1,
                size_average=True,
                win_size=ev["ms_window"],
                win_sigma=ev["ms_sigma"],
                weights=ev["ms_weights"],
                K=tuple(ev["ms_K"]),
            )
        ),
    )
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite metric")
    return metrics


def capacity_decision(a, b, k, d_ref, epsilon, complete, invariants, config=None):
    ac = (config or DEFAULTS)["acceptance"]
    if d_ref < 1e-8 or not ac["epsilon_min"] <= epsilon <= ac["epsilon_max"]:
        raise ValueError("invalid registered distortion scale/noise")
    gain = (a["D_ctl"] - b["D_ctl"]) / d_ref
    guards = dict(
        structural=b["MS_SSIM"] >= a["MS_SSIM"] - ac["ms_tolerance"],
        psnr=b["PSNR"] >= a["PSNR"] - ac["psnr_tolerance"],
        tail=b["D_tail"]
        <= (1 + ac["tail_relative"]) * a["D_tail"] + ac["tail_absolute"],
    )
    threshold = max(ac["lambda"] * k / 1000, ac["delta"]) + epsilon
    return dict(
        approved=bool(
            k > 0
            and complete
            and invariants
            and all(guards.values())
            and gain >= threshold
        ),
        gain=gain,
        U=gain / (k / 1000) if k else None,
        threshold=threshold,
        guards=guards,
        complete=complete,
        invariants=invariants,
    )
