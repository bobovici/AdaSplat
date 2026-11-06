"""PRODUCTION VNEXT: pure geometry and LR schedule; no legacy decisions."""
import math
import torch

PRIMITIVE_FIELDS = ("xys", "scale_raw", "rotation_angle", "mass", "colors", "p")


def inverse_scale(scale):
    """Inverse of PMN's softplus(raw)+0.25 transform, numerically safely."""
    value = torch.as_tensor(scale).clamp_min(0.2501) - 0.25
    return torch.log(torch.expm1(value).clamp_min(1e-8))


@torch.no_grad()
def context_leaf_birth_scales(
    leaves,
    counts,
    complexity,
    coefficient=0.75,
    complexity_beta=2.0,
    factor_min=0.1,
    factor_max=1.25,
    scale_min=2.0,
    scale_max=16.0,
):
    """Return isotropic actual newborn scales from leaf spacing and complexity."""
    if complexity is None or complexity.ndim != 2:
        raise ValueError("context birth scale requires a [H,W] complexity map")
    if (
        coefficient <= 0
        or complexity_beta < 0
        or factor_min <= 0
        or factor_max < factor_min
    ):
        raise ValueError("invalid context birth-scale modulation")
    if scale_min <= 0.25 or scale_max < scale_min:
        raise ValueError("invalid context actual-scale bounds")
    populated = [(leaf, int(n)) for leaf, n in zip(leaves, counts) if int(n) > 0]
    if not populated:
        return torch.empty((0, 2), dtype=complexity.dtype, device=complexity.device), []
    means = torch.tensor(
        [
            float(complexity[leaf["y0"] : leaf["y1"], leaf["x0"] : leaf["x1"]].mean())
            for leaf, _ in populated
        ],
        dtype=complexity.dtype,
        device=complexity.device,
    )
    lo, hi = means.min(), means.max()
    normalized = (means - lo) / (hi - lo).clamp_min(torch.finfo(means.dtype).eps)
    factors = (
        (1.0 - normalized)
        .clamp_min(0)
        .pow(float(complexity_beta))
        .clamp(float(factor_min), float(factor_max))
    )
    chunks, rows = [], []
    for index, ((leaf, n), mean, norm, factor) in enumerate(
        zip(populated, means, normalized, factors)
    ):
        spacing = math.sqrt(float(leaf["area"]) / n)
        actual = min(
            float(scale_max),
            max(float(scale_min), float(coefficient) * spacing * float(factor)),
        )
        chunks.append(
            torch.full((n, 2), actual, dtype=complexity.dtype, device=complexity.device)
        )
        rows.append(
            {
                "leaf_index": index,
                "leaf_id": int(leaf.get("leaf_id", index)),
                "allocated_count": n,
                "leaf_area": int(leaf["area"]),
                "leaf_mean_complexity": float(mean),
                "normalized_complexity": float(norm),
                "complexity_factor": float(factor),
                "spacing": spacing,
                "actual_birth_scale": actual,
                "hit_min_clamp": bool(actual <= float(scale_min)),
                "hit_max_clamp": bool(actual >= float(scale_max)),
            }
        )
    return torch.cat(chunks, dim=0), rows


def cosine_factor(
    stage_iteration: int, stage_max_iterations: int, floor: float = 0.05
) -> float:
    t = min(1.0, max(0.0, float(stage_iteration) / max(int(stage_max_iterations), 1)))
    return float(floor) + (1.0 - float(floor)) * 0.5 * (1.0 + math.cos(math.pi * t))


class StageCosineController:
    """Cosine schedule preserving every optimizer group's relative base LR."""

    def __init__(self, optimizer, stage_max_iterations: int, floor: float = 0.05):
        self.optimizer = optimizer
        self.stage_max_iterations = int(stage_max_iterations)
        self.floor = float(floor)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.stage_start = 0
        self.last_iteration = None

    def state_dict(self):
        """Serialize the actual stage clock, not the model's unused StepLR."""
        return {
            "schema": "stage_cosine_v1",
            "stage_max_iterations": self.stage_max_iterations,
            "floor": self.floor,
            "stage_start": self.stage_start,
            "last_iteration": self.last_iteration,
            "base_lrs": list(self.base_lrs),
            "optimizer_lrs": [float(g["lr"]) for g in self.optimizer.param_groups],
            "group_parameter_counts": [
                len(g["params"]) for g in self.optimizer.param_groups
            ],
        }

    def load_state_dict(self, state):
        """Restore only after optimizer load; never restart or advance time."""
        if state.get("schema") != "stage_cosine_v1":
            raise ValueError(
                "missing actual stage-cosine schema; legacy clock cannot be inferred"
            )
        groups = self.optimizer.param_groups
        base = [float(x) for x in state["base_lrs"]]
        saved_lrs = [float(x) for x in state["optimizer_lrs"]]
        if (
            len(base) != len(groups)
            or len(saved_lrs) != len(groups)
            or state["group_parameter_counts"] != [len(g["params"]) for g in groups]
        ):
            raise ValueError("stage cosine optimizer group mismatch")
        if not all(math.isfinite(x) and x >= 0 for x in base + saved_lrs):
            raise ValueError("non-finite or negative stage cosine learning rate")
        if saved_lrs != [float(g["lr"]) for g in groups]:
            raise ValueError("restore optimizer learning rates before actual scheduler")
        horizon = int(state["stage_max_iterations"])
        floor = float(state["floor"])
        start = int(state["stage_start"])
        last = state["last_iteration"]
        if horizon <= 0 or not math.isfinite(floor) or not 0 <= floor <= 1:
            raise ValueError("invalid stage cosine horizon or floor")
        if last is not None:
            last = int(last)
        # Publish after validation so a rejected load cannot partially mutate us.
        self.stage_max_iterations, self.floor = horizon, floor
        self.stage_start, self.last_iteration = start, last
        self.base_lrs = base

    def restart(self, iteration: int, optimizer=None) -> None:
        if optimizer is not None:
            self.optimizer = optimizer
            if len(self.base_lrs) != len(optimizer.param_groups):
                self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.stage_start = int(iteration)
        self.last_iteration = int(iteration)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base

    def step(self, iteration: int) -> float:
        factor = cosine_factor(
            int(iteration) - self.stage_start, self.stage_max_iterations, self.floor
        )
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor
        self.last_iteration = int(iteration)
        return factor
