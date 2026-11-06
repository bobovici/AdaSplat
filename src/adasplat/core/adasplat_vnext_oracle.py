"""PRODUCTION VNEXT optional counterfactual oracle. Never a diagnosis default."""
import copy, math
import torch
from adasplat_vnext_growth import (
    propose_micro_growth,
    apply_micro_proposal,
    bank_optimizer_step,
)
from adasplat_vnext_config import settings
from adasplat_vnext_objective import *


@torch.no_grad()
def proposal_regions(runtime, proposal):
    """Expand frozen halos to contain conservative actual newborn support AABBs."""
    regions = {c["id"]: list(c["halo"]) for c in proposal["cores"]}
    if runtime.parameter_banks:
        m = runtime.model
        rows = runtime.parameter_banks["new"].rows
        xy = m.xys[rows].cpu()
        scale = m.scales[rows].cpu()
        p = m.p[rows].cpu()
        a = m.effective_opacity()[rows].cpu()
        theta = m.rotation_angle[rows].cpu()
        for i, point in enumerate(proposal["points"]):
            if float(a[i]) < m.alpha_cutoff:
                continue
            radius = (-2 * math.log(m.alpha_cutoff / float(a[i]))) ** (1 / float(p[i]))
            c = abs(math.cos(float(theta[i])))
            s = abs(math.sin(float(theta[i])))
            rx = radius * (c * float(scale[i, 0]) + s * float(scale[i, 1]))
            ry = radius * (s * float(scale[i, 0]) + c * float(scale[i, 1]))
            box = regions[point["core"]]
            box[:] = [
                max(0, min(box[0], math.floor(float(xy[i, 0]) - rx))),
                max(0, min(box[1], math.floor(float(xy[i, 1]) - ry))),
                min(m.width, max(box[2], math.ceil(float(xy[i, 0]) + rx) + 1)),
                min(m.height, max(box[3], math.ceil(float(xy[i, 1]) + ry) + 1)),
            ]
    return regions


def run_capacity_probe(
    store,
    target,
    remaining_budget,
    old_updates=None,
    allow_commit=True,
    progress=None,
    state_hook=None,
):
    """One S0/A/B transaction. Short probes cannot accept or commit continuation."""
    import time
    from adasplat_training_state import snapshot_active, canonical_hash

    cfg = settings(store.active)
    op = cfg["oracle_probe"]
    birth_cfg = cfg["birth"]
    ac = cfg["acceptance"]
    old_updates = op["old_updates"] if old_updates is None else old_updates
    complete = old_updates == op["old_updates"] and not cfg["runtime"]["smoke"]

    def phase(name):
        if state_hook:
            state_hook(name)

    if not 0 <= old_updates <= op["old_updates"]:
        raise ValueError("single-trial update budget outside frozen range")
    s0 = snapshot_active(store.active)
    d_ref = store.active.metadata["metric_runtime"].get("D_ref")
    if d_ref is None:
        raise ValueError("freeze D_ref before initializing the trial store")
    if (
        store.active.parameter_banks
        and store.active.metadata["cohorts"][-1]["joint_updates"]
        < birth_cfg["ramp_steps"]
    ):
        raise ValueError("oracle requires completed prior maturation")
    if store.active.schedulers["actual"].floor != 1.0:
        raise ValueError("old LR must follow frozen constant stage")
    start_clock = store.active.metadata["clock"]["logical_iteration"]
    with torch.no_grad():
        base_image = store.active.model().detach()
        base = control_metrics(base_image, target, cfg)
    noise_d = 0.0
    noise_ms = 0.0
    noise_image = 0.0
    noise_mse = 0.0
    for _ in range(op["noise_repeats"]):
        with torch.no_grad():
            image = store.active.model().detach()
            metrics = control_metrics(image, target, cfg)
        noise_d = max(noise_d, abs(metrics["D_ctl"] - base["D_ctl"]))
        noise_ms = max(noise_ms, abs(metrics["MS_SSIM"] - base["MS_SSIM"]))
        noise_image = max(noise_image, float((image - base_image).abs().max()))
        noise_mse = max(noise_mse, abs(metrics["MSE"] - base["MSE"]))
    epsilon = max(ac["epsilon_min"], ac["noise_multiplier"] * noise_d / d_ref)
    if (
        epsilon > ac["epsilon_max"]
        or noise_ms >= ac["noise_ms_tolerance"]
        or noise_image > birth_cfg["render_atol"]
        or noise_mse > birth_cfg["mse_atol"]
    ):
        raise ValueError("measurement noise gate failed")
    wall = time.perf_counter()
    phase("PROPOSE")
    proposal = propose_micro_growth(store.active, target, remaining_budget)
    proposal_seconds = time.perf_counter() - wall
    tid = store.begin(
        {
            "proposal": proposal,
            "D_ref": d_ref,
            "epsilon": epsilon,
            "old_updates": old_updates,
        }
    )
    store.record_cost(tid, proposal_wall_seconds=proposal_seconds)
    result = dict(
        trial=tid,
        S0_hash=s0.state_hash,
        K=proposal["K"],
        K_cap=proposal["K_cap"],
        D_ref=d_ref,
        epsilon=epsilon,
        noise=dict(D_ctl=noise_d, MS_SSIM=noise_ms, render=noise_image, MSE=noise_mse),
        evaluation={},
        formal_training_started=False,
        complete=complete,
    )
    final_states = {}
    final_images = {}
    lr_traces = {}
    event_pairs = []
    cost_branch = None

    def account():
        if event_pairs:
            torch.cuda.synchronize()
            seconds = sum(a.elapsed_time(b) for a, b in event_pairs) / 1000
            store.record_cost(
                tid,
                **{
                    cost_branch + "_gpu_seconds": seconds,
                    (
                        "warmup_updates"
                        if cost_branch == "warmup"
                        else cost_branch + "_old_updates"
                    ): len(event_pairs),
                }
            )
            event_pairs.clear()

    def timed_step(runtime, warmup=False):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        loss = unified_loss(runtime, target)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        if runtime.parameter_banks:
            bank_optimizer_step(runtime, warmup=warmup)
        else:
            runtime.optimizers["old"].step()
            runtime.model.zero_grad(set_to_none=True)
            runtime.model.project_parameters_()
            runtime.metadata["clock"]["logical_iteration"] += 1
            runtime.schedulers["actual"].step(
                runtime.metadata["clock"]["logical_iteration"]
            )
            runtime.metadata["clock"]["phase"] = "JOINT"
        b.record()
        event_pairs.append((a, b))

    def audit(runtime):
        snap = snapshot_active(runtime)
        return dict(
            state_hash=snap.state_hash,
            N=runtime.model.count,
            old_LR=[g["lr"] for g in runtime.optimizers["old"].param_groups],
            group_ages={
                name: [int(g.get("step", 0)) for g in opt.param_groups]
                for name, opt in runtime.optimizers.items()
            },
            clock=copy.deepcopy(runtime.metadata["clock"]),
        )

    def evaluate(runtime, label, branch):
        nonlocal wall
        account()
        with torch.no_grad():
            image = runtime.model().detach()
            m = control_metrics(image, target, cfg)
        regions = proposal_regions(runtime, proposal)
        m["local"] = {
            str(key): float(
                (
                    image[:, :, box[1] : box[3], box[0] : box[2]]
                    - target[:, :, box[1] : box[3], box[0] : box[2]]
                )
                .square()
                .mean()
            )
            for key, box in regions.items()
        }
        m["regions"] = regions
        m.update(audit(runtime))
        result["evaluation"][branch][label] = m
        store.record_cost(tid, **{branch + "_wall_seconds": time.perf_counter() - wall})
        wall = time.perf_counter()
        if progress:
            progress(branch, label, m)
        return image, m

    try:
        for branch in ("A", "B"):
            if branch == "B":
                phase("RESTORE_S0")
            runtime = store.restore_origin(tid)
            if snapshot_active(runtime).state_hash != s0.state_hash:
                raise ValueError("branch origin mismatch")
            torch.cuda.reset_peak_memory_stats()
            cost_branch = branch
            wall = time.perf_counter()
            result["evaluation"][branch] = {
                "S0": dict(copy.deepcopy(base), **audit(runtime))
            }
            result.setdefault("branch_contracts", {})[branch] = dict(
                S0=snapshot_active(runtime).state_hash,
                target=canonical_hash(target),
                external=canonical_hash(runtime.external),
                renderer=copy.deepcopy(runtime.metadata["renderer_runtime"]),
                metric=copy.deepcopy(runtime.metadata["metric_runtime"]),
                loss="MSE+.05*(1-SSIM7)",
                rng=canonical_hash(snapshot_active(runtime).payload["rng"]),
                named_rng=canonical_hash(
                    snapshot_active(runtime).payload["named_generators"]
                ),
                clock=copy.deepcopy(runtime.metadata["clock"]),
            )
            phase("BRANCH_" + branch)
            if branch == "B":
                apply_micro_proposal(runtime, target, proposal)
            runtime.metadata["proposal"] = copy.deepcopy(proposal)
            image, birth = evaluate(runtime, "after_birth", branch)
            if branch == "B":
                if (
                    float((image - base_image).abs().max()) > birth_cfg["render_atol"]
                    or abs(birth["MSE"] - base["MSE"]) > birth_cfg["mse_atol"]
                ):
                    raise ValueError("BIRTH_INVARIANCE")
                if (
                    base["MSE"] >= 1e-5
                    and abs(birth["MSE"] - base["MSE"]) / base["MSE"]
                    > birth_cfg["mse_relative"]
                ):
                    raise ValueError("BIRTH_RELATIVE_MSE")
            if branch == "B" and proposal["K"]:
                unified_loss(runtime, target).backward()
                norms = runtime.model.colors.grad[-proposal["K"] :].norm(dim=1)
                result["newborn_gradient"] = dict(
                    live_fraction=float(
                        (norms > birth_cfg["live_norm"]).float().mean()
                    ),
                    mean=float(norms.mean()),
                )
                runtime.model.zero_grad(set_to_none=True)
                if (
                    not torch.isfinite(norms).all()
                    or result["newborn_gradient"]["live_fraction"]
                    < birth_cfg["live_fraction"]
                    or result["newborn_gradient"]["mean"] <= birth_cfg["live_mean"]
                ):
                    raise ValueError("DEAD_PROPOSAL")
                cost_branch = "warmup"
                for _ in range(birth_cfg["warmup_steps"]):
                    timed_step(runtime, warmup=True)
                account()
                cost_branch = branch
                old_n = s0.payload["metadata"]["population"]["N"]
                for key, value in s0.payload["model"].items():
                    if not torch.equal(
                        runtime.model.state_dict()[key][:old_n].cpu(), value
                    ):
                        raise ValueError("OLD_WARMUP_MUTATION")
                for old_key, saved in s0.payload["optimizers"].items():
                    key = runtime.metadata["birth_old_bank_ids"][old_key]
                    if canonical_hash(
                        runtime.optimizers[key].state_dict()
                    ) != canonical_hash(saved["state_dict"]):
                        raise ValueError("OLD_OPTIMIZER_WARMUP_MUTATION")
                if canonical_hash(
                    runtime.schedulers["actual"].state_dict()
                ) != canonical_hash(s0.payload["schedulers"]["actual"]["state_dict"]):
                    raise ValueError("OLD_SCHEDULER_WARMUP_MUTATION")
            evaluate(runtime, "warmup64", branch)
            trace = []
            for step in range(1, old_updates + 1):
                trace.append([g["lr"] for g in runtime.optimizers["old"].param_groups])
                timed_step(runtime)
                if step in op["evaluation_steps"]:
                    evaluate(runtime, "old" + str(step), branch)
            image, end = evaluate(runtime, "endpoint", branch)
            if (
                runtime.metadata["clock"]["logical_iteration"]
                != start_clock + old_updates
            ):
                raise ValueError("OLD_UPDATE_COUNT")
            runtime.metadata["selected_accepted_topology"] = dict(
                id="candidate",
                N=runtime.model.count,
                logical_iteration=start_clock + old_updates,
                model_hash=canonical_hash(runtime.model.state_dict()),
            )
            final_states[branch] = snapshot_active(runtime)
            final_images[branch] = image.cpu()
            lr_traces[branch] = canonical_hash(trace)
            store.branch_end(tid, branch, final_states[branch], end)
            result.setdefault("peak_VRAM_bytes", {})[branch] = dict(
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
            )
        if lr_traces["A"] != lr_traces["B"]:
            raise ValueError("OLD_LR_AXIS_MISMATCH")
        if result["branch_contracts"]["A"] != result["branch_contracts"]["B"]:
            raise ValueError("TRIAL_INVALID_CONTRACT_MISMATCH")
        result["old_updates_per_branch"] = old_updates
        a = result["evaluation"]["A"]["endpoint"]
        b = result["evaluation"]["B"]["endpoint"]
        if not proposal["K"]:
            delta = float((final_images["A"] - final_images["B"]).abs().max())
            exact = {
                key: canonical_hash(final_states["A"].payload[key])
                == canonical_hash(final_states["B"].payload[key])
                for key in (
                    "rng",
                    "named_generators",
                    "schedulers",
                    "external",
                    "backend_flags",
                )
            }
            consistent = (
                delta <= ac["k0_render_tolerance"]
                and abs(a["MSE"] - b["MSE"]) <= ac["k0_mse_tolerance"]
                and abs(a["D_ctl"] - b["D_ctl"]) / d_ref <= epsilon
                and abs(a["MS_SSIM"] - b["MS_SSIM"]) <= ac["k0_ms_tolerance"]
                and all(exact.values())
            )
            result["K0_consistency"] = dict(
                passed=consistent,
                render_max_delta=delta,
                exact_components=exact,
                entire_training_endpoint_bitwise_equal=final_states["A"].state_hash
                == final_states["B"].state_hash,
            )
            if not consistent:
                raise ValueError("K_ZERO_TRACK_MISMATCH")
        phase("COMPARE")
        decision = capacity_decision(
            a, b, proposal["K"], d_ref, epsilon, complete, True, cfg
        )
        result.update(
            decision=decision,
            lr_axis_hash=lr_traces["A"],
            invariants_pass=True,
            branch_hashes={key: s.state_hash for key, s in final_states.items()},
        )
        # Same final halo for the A/B local comparison, including moved support.
        result["local_comparison"] = {}
        cpu_target = target.cpu()
        for key, box in b["regions"].items():
            x0, y0, x1, y1 = box
            result["local_comparison"][str(key)] = {
                branch: float(
                    (
                        final_images[branch][:, :, y0:y1, x0:x1]
                        - cpu_target[:, :, y0:y1, x0:x1]
                    )
                    .square()
                    .mean()
                )
                for branch in ("A", "B")
            }
        if decision["approved"] and allow_commit:
            store.decide(tid, "B", decision, lambda *args: bool(args[-1]["approved"]))
            phase("COMMIT_B")
            store.commit(tid, "B", s0.state_hash)
            result["outcome"] = "COMMIT_B"
        else:
            phase("REJECT")
            store.reject(
                tid, "TRIAL_INCOMPLETE" if not complete else "capacity/guard rejection"
            )
            if snapshot_active(store.active).state_hash != s0.state_hash:
                raise ValueError("REJECT_EXACTNESS")
            result["reject_S0_exact"] = True
            result["outcome"] = "REJECT_S0"
            nonworse = (
                a["loss"] <= base["loss"]
                and a["PSNR"] >= base["PSNR"]
                and a["MS_SSIM"] >= base["MS_SSIM"]
            )
            if complete and allow_commit and nonworse:
                store.decide(
                    tid,
                    "A",
                    {"nonworse": True},
                    lambda *args: bool(args[-1]["nonworse"]),
                )
                phase("COMMIT_A")
                store.commit(tid, "A", s0.state_hash)
                result["outcome"] = "REJECT_THEN_COMMIT_A"
        result["selected_active_hash"] = snapshot_active(store.active).state_hash
        return result
    except Exception as error:
        account()
        if store.poisoned:
            store.recover()
        else:
            store.reject(tid, str(error), [str(error)])
        raise


# Compatibility name for historical test callers; production calls the oracle API.
run_single_trial = run_capacity_probe
