"""PRODUCTION VNEXT: proposal eligibility, function-preserving birth, cohort maturation."""
import copy, math
import torch
from optimizer import Adan
from adasplat_vnext_primitives import (
    PRIMITIVE_FIELDS,
    inverse_scale,
    context_leaf_birth_scales,
)
from primix_v4_2_parameterization import exact_l2_norm
from adasplat_vnext_config import DEFAULTS, settings
from adasplat_vnext_objective import unified_loss


def micro_budget(n, remaining, config=None):
    cfg = (config or DEFAULTS)["micro_growth"]
    if n < 0 or remaining < 0:
        raise ValueError("negative capacity budget")
    return min(cfg["max_k"], math.ceil(n * cfg["fraction"]), int(remaining))


def core_grid(height, width, config=None):
    cfg = (config or DEFAULTS)["micro_growth"]
    size = cfg["core_size"]
    return [
        dict(
            x0=x,
            y0=y,
            x1=min(x + size, width),
            y1=min(y + size, height),
            area=min(size, width - x) * min(size, height - y),
            cap=max(
                1,
                cfg["core_cap"]
                * min(size, width - x)
                * min(size, height - y)
                // (size * size),
            ),
        )
        for y in range(0, height, size)
        for x in range(0, width, size)
    ]


def _rank_sampled_points(r, cores, points, complexity, config=None):
    cfg = config or DEFAULTS
    birth = cfg["birth"]
    cutoff = cfg["renderer"]["cutoff"]
    h, w = r.shape[-2:]
    total = float(r.square().sum())
    # Recompute context scales if exact-zero projected candidates were removed.
    # Each pass only removes candidates, so this terminates without resampling.
    while points:
        counts = [sum(p["core"] == c["id"] for p in points) for c in cores]
        scales, _ = context_leaf_birth_scales(
            cores,
            counts,
            complexity.detach().cpu().double(),
            scale_min=birth["scale_min"],
            scale_max=birth["scale_max"],
            coefficient=birth["context_coefficient"],
            complexity_beta=birth["context_complexity_beta"],
            factor_min=birth["context_factor_min"],
            factor_max=birth["context_factor_max"],
        )
        ordered = [p for c in cores for p in points if p["core"] == c["id"]]
        work = r.clone()
        selected = []
        offset = 0
        for c, count in zip(cores, counts):
            pending = ordered[offset : offset + count]
            scale = float(scales[offset, 0]) if count else 0
            offset += count
            while pending:
                previews = []
                for p in pending:
                    # Exact support of p=2, opacity=.05; contained in the 64 halo.
                    radius = math.ceil(
                        scale * math.sqrt(2 * math.log(birth["opacity"] / cutoff))
                    )
                    x0 = max(0, p["x"] - radius)
                    x1 = min(w, p["x"] + radius + 1)
                    y0 = max(0, p["y"] - radius)
                    y1 = min(h, p["y"] + radius + 1)
                    yy, xx = torch.meshgrid(
                        torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij"
                    )
                    alpha = birth["opacity"] * torch.exp(
                        -0.5
                        * (
                            (xx - p["x"]).double().square()
                            + (yy - p["y"]).double().square()
                        )
                        / scale**2
                    )
                    alpha[alpha < cutoff] = 0
                    projection = (work[:, y0:y1, x0:x1] * alpha).sum((1, 2))
                    norm = alpha.square().sum() + 1e-12
                    gain = float(projection.square().sum() / norm) / max(total, 1e-30)
                    previews.append((gain, projection / norm, alpha, (x0, y0, x1, y1)))
                best = max(range(len(pending)), key=lambda i: previews[i][0])
                p = pending.pop(best)
                gain, color, alpha, box = previews[best]
                if gain <= 0:
                    continue
                x0, y0, x1, y1 = box
                work[:, y0:y1, x0:x1] -= color[:, None, None] * alpha
                selected.append(dict(p, preview_gain=gain, physical_scale=scale))
        if len(selected) == len(points):
            points = selected
            break
        points = selected
    return points


@torch.no_grad()
def propose_from_residual(
    residual, n, remaining, origin_hash, generator_state, complexity, config=None
):
    """Pure bounded CPU proposal; signed RGB data, no renderer-size kernel bank.

    Sampling is weighted by original unblurred SSE. Sequential local projection
    only ranks sampled candidates. It is never an acceptance or capacity proof.
    """
    from adasplat_training_state import canonical_hash

    cfg = config or DEFAULTS
    micro = cfg["micro_growth"]
    halo = micro["halo"]
    retries = micro["sampling_multiplier"]
    r = residual.detach().cpu().double().clone()
    if r.ndim != 3 or r.shape[0] != 3 or not torch.isfinite(r).all():
        raise ValueError("finite RGB residual required")
    h, w = r.shape[-2:]
    cap = micro_budget(n, remaining, cfg)
    seed = int(
        canonical_hash({"S0": origin_hash, "proposal_rng": generator_state})[:16], 16
    ) % (2**63 - 1)
    generator = torch.Generator().manual_seed(seed)
    cores = core_grid(h, w, cfg)
    for i, c in enumerate(cores):
        c["id"] = i
        c["sse"] = float(r[:, c["y0"] : c["y1"], c["x0"] : c["x1"]].square().sum())
        c["halo"] = [
            max(0, c["x0"] - halo),
            max(0, c["y0"] - halo),
            min(w, c["x1"] + halo),
            min(h, c["y1"] + halo),
        ]
    cores.sort(key=lambda c: (-c["sse"], c["id"]))
    points = []
    attempts = 0
    for c in cores:
        wanted = min(c["cap"], cap - len(points))
        if not wanted or not c["sse"] or attempts >= retries * cap:
            continue
        weights = r[:, c["y0"] : c["y1"], c["x0"] : c["x1"]].square().sum(0).flatten()
        for slot in range(wanted):
            for retry in range(min(retries, retries * cap - attempts)):
                index = int(torch.multinomial(weights, 1, generator=generator))
                attempts += 1
                x = c["x0"] + index % (c["x1"] - c["x0"])
                y = c["y0"] + index // (c["x1"] - c["x0"])
                if any(
                    (x - p["x"]) ** 2 + (y - p["y"]) ** 2 < micro["min_distance"] ** 2
                    for p in points
                ):
                    continue
                points.append(dict(x=x, y=y, core=c["id"]))
                break
    points = _rank_sampled_points(r, cores, points, complexity, cfg)
    counts = [sum(p["core"] == c["id"] for p in points) for c in cores]
    result = dict(
        schema="adasplat_micro_proposal_v1",
        origin_hash=origin_hash,
        residual_hash=canonical_hash(r),
        seed=seed,
        K_cap=cap,
        K=len(points),
        sampling_attempts=attempts,
        cores=cores,
        counts=counts,
        points=points,
        ranking_only=True,
    )
    result["proposal_hash"] = canonical_hash(result)
    return result


@torch.no_grad()
def propose_micro_growth(runtime, target, remaining):
    from adasplat_training_state import (
        snapshot_active,
        restore_active,
        canonical_hash,
        capture_rng,
        restore_rng,
    )
    from adasplat_vnext_runtime import v42_state_factory, validate_v42_state
    from primix_complexity_quadtree import compute_complexity_map

    cfg = settings(runtime)
    birth = cfg["birth"]
    before = snapshot_active(runtime)
    if "proposal" not in runtime.named_generators:
        raise ValueError("independent proposal generator required")
    residual = (target - runtime.model())[0]
    complexity = compute_complexity_map(target)
    proposal = propose_from_residual(
        residual,
        runtime.model.count,
        remaining,
        before.state_hash,
        runtime.named_generators["proposal"].get_state(),
        complexity,
        cfg,
    )
    initial_k = proposal["K"]
    passes = 0
    rng = capture_rng()
    try:
        # Eligibility uses the actual frozen loss and sparse renderer on a restored
        # staging copy. It never changes ranking into a capacity acceptance test.
        while proposal["K"]:
            staged = restore_active(
                before, v42_state_factory(target), runtime.external, validate_v42_state
            )
            apply_micro_proposal(staged, target, proposal)
            with torch.enable_grad():
                unified_loss(staged, target).backward()
            norms = (
                staged.model.colors.grad[-proposal["K"] :].norm(dim=1).detach().cpu()
            )
            staged.model.zero_grad(set_to_none=True)
            del staged
            passes += 1
            if not torch.isfinite(norms).all():
                raise ValueError("nonfinite proposal preview gradient")
            keep = norms > birth["live_norm"]
            # Preserve the frozen batch mean gate, removing weakest candidates
            # only if necessary. No replacement draws or forced filling.
            while keep.any() and float(norms[keep].mean()) <= birth["live_mean"]:
                indices = keep.nonzero().flatten()
                keep[indices[norms[indices].argmin()]] = False
            if bool(keep.all()):
                break
            points = [p for p, valid in zip(proposal["points"], keep.tolist()) if valid]
            points = _rank_sampled_points(
                residual.detach().cpu().double(),
                proposal["cores"],
                points,
                complexity,
                cfg,
            )
            proposal.update(
                points=points,
                K=len(points),
                counts=[
                    sum(p["core"] == c["id"] for p in points) for c in proposal["cores"]
                ],
            )
            proposal["proposal_hash"] = canonical_hash(
                {k: v for k, v in proposal.items() if k != "proposal_hash"}
            )
        proposal["eligibility_preview"] = dict(
            initial_K=initial_k,
            removed=initial_k - proposal["K"],
            passes=passes,
            loss="MSE + 0.05 * (1 - SSIM7)",
            resampled=False,
        )
        proposal["proposal_hash"] = canonical_hash(
            {k: v for k, v in proposal.items() if k != "proposal_hash"}
        )
    finally:
        restore_rng(rng)
        if snapshot_active(runtime).state_hash != before.state_hash:
            raise ValueError("proposal mutated active state")
    return proposal


def apply_micro_proposal(runtime, target, proposal):
    from adasplat_training_state import canonical_hash, snapshot_active
    from primix_complexity_quadtree import compute_complexity_map

    if (
        canonical_hash({k: v for k, v in proposal.items() if k != "proposal_hash"})
        != proposal["proposal_hash"]
    ):
        raise ValueError("proposal hash mismatch")
    if snapshot_active(runtime).state_hash != proposal["origin_hash"]:
        raise ValueError("stale proposal origin")
    if not proposal["K"]:
        return runtime
    xy = torch.tensor(
        [[p["x"], p["y"]] for p in proposal["points"]],
        device=target.device,
        dtype=torch.float32,
    )
    zero_color_bridge(
        runtime,
        target,
        xy,
        proposal["cores"],
        proposal["counts"],
        compute_complexity_map(target),
    )
    runtime.metadata["proposal"] = copy.deepcopy(proposal)
    return runtime


class CohortBank:
    """Compact independent Adan leaves; the renderer retains its six tensors."""

    def __init__(self, model, rows):
        self.rows = torch.as_tensor(rows, device=model.xys.device, dtype=torch.long)
        self.parameters = {
            name: torch.nn.Parameter(getattr(model, name).detach()[self.rows].clone())
            for name in PRIMITIVE_FIELDS
        }

    def optimizer(self, original, old=False):
        names = list(PRIMITIVE_FIELDS)
        if len(original.param_groups) != len(names) or any(
            len(g["params"]) != 1 for g in original.param_groups
        ):
            raise ValueError("bridge requires native six-group Adan binding")
        groups = []
        for name, group in zip(names, original.param_groups):
            options = copy.deepcopy(
                {k: v for k, v in group.items() if k not in ("params", "step")}
            )
            groups.append(dict(options, params=[self.parameters[name]]))
        # Optimizer.load_state_dict can add PyTorch compatibility defaults such
        # as differentiable; these are not Adan constructor keyword arguments.
        optimizer = Adan(groups)
        optimizer.defaults = copy.deepcopy(original.defaults)
        if old:
            optimizer.load_state_dict(copy.deepcopy(original.state_dict()))
        return optimizer

    @torch.no_grad()
    def sync_from_model(self, model):
        for name, p in self.parameters.items():
            p.copy_(getattr(model, name)[self.rows])

    @torch.no_grad()
    def update(self, model, optimizer, enabled):
        """Exclude inactive groups entirely: Adan ages even groups with grad=None."""
        groups = optimizer.param_groups
        active = []
        for name, group in zip(PRIMITIVE_FIELDS, groups):
            p = self.parameters[name]
            grad = getattr(model, name).grad
            p.grad = (
                None if name not in enabled or grad is None else grad[self.rows].clone()
            )
            if p.grad is not None:
                active.append(group)
        try:
            optimizer.param_groups = active
            if active:
                optimizer.step()
        finally:
            optimizer.param_groups = groups
            optimizer.zero_grad(set_to_none=True)
        for name, p in self.parameters.items():
            getattr(model, name)[self.rows] = p


def zero_color_bridge(runtime, target, xy, leaves, counts, complexity):
    """Mutate only a staged branch. Caller must use persisted S0 on failure.

    No forward or external callback occurs between legacy append and zeroing.
    """
    from adasplat_training_state import canonical_hash, capture_rng, restore_rng
    from adasplat_vnext_primitives import StageCosineController

    if (
        runtime.parameter_banks
        and runtime.metadata["cohorts"][-1]["joint_updates"]
        < settings(runtime)["birth"]["ramp_steps"]
    ):
        raise ValueError("existing newborn cohort must mature before another birth")
    cfg = settings(runtime)
    birth = cfg["birth"]
    model = runtime.model
    if (
        model.alpha_cutoff != cfg["renderer"]["cutoff"]
        or model.p_min != cfg["representation"]["p_min"]
        or model.p_max != cfg["representation"]["p_max"]
    ):
        raise ValueError("bridge requires frozen Phase 1 runtime")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("zero-grad boundary required")
    if len(xy) != sum(counts) or len(leaves) != len(counts):
        raise ValueError("candidate allocation mismatch")
    if not len(xy):
        return runtime
    if (
        not torch.isfinite(xy).all()
        or (xy < 0).any()
        or (xy[:, 0] > model.width - 1).any()
        or (xy[:, 1] > model.height - 1).any()
    ):
        raise ValueError("candidate coordinates outside image")
    old_n = model.count
    old_tensors = {
        name: getattr(model, name).detach().clone() for name in PRIMITIVE_FIELDS
    }
    original = runtime.optimizers["old"]
    optimizer_before = canonical_hash(original.state_dict())
    schedule = copy.deepcopy(runtime.schedulers["actual"].state_dict())
    rng = capture_rng()
    try:
        # Append only the six native parameter tensors; never invoke legacy growth.
        for name in PRIMITIVE_FIELDS:
            old = getattr(model, name)
            added = torch.zeros(
                (len(xy),) + old.shape[1:], device=old.device, dtype=old.dtype
            )
            if name == "xys":
                added.copy_(xy)
            setattr(model, name, torch.nn.Parameter(torch.cat((old.detach(), added))))
        model.count += len(xy)
        scales, _ = context_leaf_birth_scales(
            leaves,
            counts,
            complexity,
            scale_min=birth["scale_min"],
            scale_max=birth["scale_max"],
            coefficient=birth["context_coefficient"],
            complexity_beta=birth["context_complexity_beta"],
            factor_min=birth["context_factor_min"],
            factor_max=birth["context_factor_max"],
        )
        with torch.no_grad():
            model.colors[old_n:] = 0
            model.p[old_n:] = birth["p"]
            model.rotation_angle[old_n:] = 0.0
            model.scale_raw[old_n:] = inverse_scale(scales)
            model.mass[old_n:] = birth["opacity"] * exact_l2_norm(
                model.p[old_n:], model.scales[old_n:]
            )
        for name, before in old_tensors.items():
            if not torch.equal(getattr(model, name)[:old_n], before):
                raise ValueError("legacy append modified old rows: " + name)
        if runtime.parameter_banks:
            # Preserve every historical cohort's own Adan age and moments.
            key = "cohort_" + str(len(runtime.metadata["cohorts"]) - 1)
            runtime.parameter_banks[key] = runtime.parameter_banks.pop("new")
            runtime.optimizers[key] = runtime.optimizers.pop("new")
            runtime.metadata["cohorts"][-1]["id"] = key
            old_ids = {name: name for name in runtime.optimizers if name != key}
            old_ids["new"] = key
            old_optimizer = runtime.optimizers["old"]
        else:
            old = CohortBank(model, torch.arange(old_n, device=model.xys.device))
            old_optimizer = old.optimizer(original, old=True)
            if canonical_hash(old_optimizer.state_dict()) != optimizer_before:
                raise ValueError("old Adan state changed in bank bridge")
            runtime.parameter_banks = {"old": old}
            runtime.optimizers = {"old": old_optimizer}
            runtime.metadata["cohorts"] = [dict(id="old", start=0, end=old_n)]
            old_ids = {"old": "old"}
        new = CohortBank(
            model, torch.arange(old_n, model.count, device=model.xys.device)
        )
        new_optimizer = new.optimizer(original)
        runtime.parameter_banks["new"] = new
        runtime.optimizers["new"] = new_optimizer
        scheduler = StageCosineController(
            old_optimizer, cfg["oracle_probe"]["old_updates"], 1.0
        )
        scheduler.load_state_dict(schedule)
        runtime.schedulers = {"actual": scheduler}
        model.optimizer = old_optimizer
        runtime.metadata["population"]["N"] = model.count
        runtime.metadata["cohorts"].append(
            dict(
                id="new", start=old_n, end=model.count, color_updates=0, joint_updates=0
            )
        )
        runtime.metadata["birth_old_bank_ids"] = old_ids
        runtime.metadata["clock"]["phase"] = "NEW_COLOR_WARMUP"
        runtime.metadata["bank_base_lrs"] = [
            g["lr"] for g in new_optimizer.param_groups
        ]
    finally:
        restore_rng(rng)
    return runtime


def bank_optimizer_step(runtime, *, warmup):
    """Apply one already-computed loss gradient; no policy or implicit growth."""
    cfg = settings(runtime)
    birth = cfg["birth"]
    model = runtime.model
    cohort = runtime.metadata["cohorts"][-1]
    if warmup:
        if cohort["color_updates"] >= birth["warmup_steps"]:
            raise ValueError("64-step warmup already complete")
        enabled = {"colors"}
        runtime.parameter_banks["new"].update(model, runtime.optimizers["new"], enabled)
        cohort["color_updates"] += 1
    else:
        if cohort["color_updates"] != birth["warmup_steps"]:
            raise ValueError("joint release requires 64 color updates")
        index = cohort["joint_updates"]
        factor = min(1.0, index / max(1, birth["ramp_steps"] - 1))
        enabled = {"colors"} if factor == 0 else set(PRIMITIVE_FIELDS)
        for name, group, base in zip(
            PRIMITIVE_FIELDS,
            runtime.optimizers["new"].param_groups,
            runtime.metadata["bank_base_lrs"],
        ):
            group["lr"] = base if name == "colors" else base * factor
        for key, bank in runtime.parameter_banks.items():
            if key != "new":
                bank.update(model, runtime.optimizers[key], set(PRIMITIVE_FIELDS))
        runtime.parameter_banks["new"].update(model, runtime.optimizers["new"], enabled)
        cohort["joint_updates"] += 1
        runtime.metadata["clock"]["logical_iteration"] += 1
        runtime.schedulers["actual"].step(
            runtime.metadata["clock"]["logical_iteration"]
        )
        runtime.metadata["clock"]["phase"] = "JOINT"
    model.zero_grad(set_to_none=True)
    # Warmup must not project/floor frozen rows. Native projection is pointwise.
    if not warmup:
        model.project_parameters_()
        for bank in runtime.parameter_banks.values():
            bank.sync_from_model(model)
