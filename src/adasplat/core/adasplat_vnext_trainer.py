"""PRODUCTION VNEXT unified transactional trainer. No legacy trainer imports."""
import copy, json, random, time
from pathlib import Path
import numpy as np
import torch
from adasplat_vnext_config import config_hash
from adasplat_training_state import TrialStore, snapshot_active, canonical_hash
from adasplat_vnext_runtime import (
    make_v42_model,
    create_selected_legacy_stage,
    selected_external,
    v42_state_factory,
    validate_v42_state,
    file_sha256,
)
from adasplat_vnext_controller import Controller, DiagnosisInput, diagnose
from adasplat_vnext_growth import bank_optimizer_step
from adasplat_vnext_objective import unified_loss, control_metrics
from adasplat_vnext_oracle import run_capacity_probe
from primix_teacher_artifacts import atomic_json


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def target_from_config(cfg):
    device = cfg["runtime"]["device"]
    size = cfg["runtime"]["synthetic_size"]
    if size is not None:
        if not cfg["runtime"]["smoke"]:
            raise ValueError("synthetic target is smoke-only")
        if size < 192:
            raise ValueError("MS-SSIM protocol requires synthetic_size>=192")
        y, x = torch.meshgrid(
            torch.linspace(0, 1, size, device=device),
            torch.linspace(0, 1, size, device=device),
            indexing="ij",
        )
        return torch.stack((x, y, (x + y) / 2))[None] * 0.4
    from PIL import Image

    return (
        torch.from_numpy(
            np.array(Image.open(cfg["evaluation"]["target"]).convert("RGB")).copy()
        )
        .permute(2, 0, 1)[None]
        .to(device)
        .float()
        / 255
    )


def external_contract(cfg, profile, target):
    source = cfg["evaluation"]["target"] or __file__
    ext = selected_external(source, profile)
    ext.update(
        config_hash=config_hash(cfg),
        target_tensor_hash=canonical_hash(target),
        initial_checkpoint_sha256=file_sha256(cfg["fit"]["legacy_checkpoint"])
        if cfg["fit"]["legacy_checkpoint"]
        else None,
        entrypoint_hash=file_sha256(
            Path(__file__).with_name("train_adasplat_vnext.py")
        ),
    )
    return ext


def create_runtime(cfg, target, profile, external):
    seed_all(cfg["runtime"]["seed"])
    checkpoint = cfg["fit"]["legacy_checkpoint"]
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get('parameterization_version') != cfg['representation']['schema']:
            raise ValueError('checkpoint scale coordinates differ; explicit conversion required')
        if (payload["p_min"], payload["p_max"]) != (
            cfg["representation"]["p_min"],
            cfg["representation"]["p_max"],
        ):
            raise ValueError(
                "legacy checkpoint p range cannot be changed during restoration; use fresh init"
            )
    else:
        model = make_v42_model(
            target,
            cfg["fit"]["initial_count"],
            cfg["optimizer"]["rates"],
            alpha_cutoff=cfg["renderer"]["cutoff"],
            p_min=cfg["representation"]["p_min"],
            p_max=cfg["representation"]["p_max"],
            scale_mode=__import__('primix_v4_2_parameterization').scale_mode_for_schema(cfg['representation']['schema']),
        )
        from optimizer import Adan

        options = copy.deepcopy(cfg["optimizer"]["adan"])
        options["betas"] = tuple(options["betas"])
        model.optimizer = Adan(
            [
                {"params": g["params"], "lr": g["lr"]}
                for g in model.optimizer.param_groups
            ],
            **options
        )
        if cfg["fit"]["init"] != "quadtree":
            raise ValueError("fresh v1 initialization must be quadtree")
        from primix_complexity_quadtree import complexity_quadtree_initialization

        xy, _, _, _ = complexity_quadtree_initialization(
            target, model.count, seed=cfg["runtime"]["seed"], **cfg["fit"]["quadtree"]
        )
        with torch.no_grad():
            model.xys.copy_(xy)
            xs = xy[:, 0].round().long().clamp(0, model.width - 1)
            ys = xy[:, 1].round().long().clamp(0, model.height - 1)
            model.colors.copy_(target[0, :, ys, xs].T)
        payload = dict(
            state_dict=model.state_dict(),
            optimizer=model.optimizer.state_dict(),
            PMN_config=cfg["optimizer"]["rates"],
            parameterization_version=cfg["representation"]["schema"],
            alpha_cutoff=cfg["renderer"]["cutoff"],
            p_min=cfg["representation"]["p_min"],
            p_max=cfg["representation"]["p_max"],
        )
    runtime = create_selected_legacy_stage(
        payload,
        target,
        cfg["evaluation"]["target"] or __file__,
        profile,
        cfg["renderer"]["cutoff"],
        "VNext explicit new-stage constructor; no legacy control flow",
    )
    if not checkpoint:
        runtime.optimizers["old"].defaults.update(options)
    if checkpoint and not cfg["optimizer"]["legacy_saved_lr"]:
        names = ["xyz_lr", "scale_lr", "rotation_lr", "opacity_lr", "color_lr", "p_lr"]
        for group, name in zip(runtime.optimizers["old"].param_groups, names):
            group["lr"] = cfg["optimizer"]["rates"][name]
    scheduler = runtime.schedulers["actual"]
    scheduler.base_lrs = [g["lr"] for g in runtime.optimizers["old"].param_groups]
    scheduler.stage_max_iterations = cfg["oracle_probe"]["old_updates"]
    scheduler.restart(cfg["fit"]["logical_start"])
    runtime.metadata["clock"].update(
        logical_iteration=cfg["fit"]["logical_start"], phase="FIT"
    )
    runtime.metadata.update(
        vnext_config=copy.deepcopy(cfg),
        config_hash=config_hash(cfg),
        controller=Controller().data,
        fit_trajectory=[],
        diagnosis_inputs={},
        initializer="legacy" if checkpoint else "quadtree",
    )
    ev = cfg["evaluation"]
    runtime.metadata["metric_runtime"].update(
        win_size=ev["ms_window"],
        win_sigma=ev["ms_sigma"],
        weights=ev["ms_weights"],
        K=ev["ms_K"],
    )
    with torch.no_grad():
        runtime.metadata["metric_runtime"]["D_ref"] = max(
            1e-8, control_metrics(runtime.model(), target, cfg)["MSE"]
        )
    seed_all(cfg["runtime"]["seed"])
    runtime.named_generators = {
        "proposal": torch.Generator(device=target.device).manual_seed(
            cfg["proposal"]["seed"]
        )
    }
    runtime.external = external
    validate_v42_state(runtime)
    return runtime


class Trainer:
    def __init__(self, store, target, config):
        self.store, self.target, self.cfg = store, target, config
        self.out = Path(config["checkpoint"]["output"])
        self.controller = Controller(store.active.metadata["controller"])

    def publish(self, mutate, kind):
        """Normal FIT/metadata checkpoint, distinct from capacity acceptance."""
        tid = self.store.begin({"kind": kind})
        started = time.perf_counter()
        try:
            mutate(self.store.active)
            snap = snapshot_active(self.store.active)
            self.store.record_cost(
                tid, checkpoint_wall_seconds=time.perf_counter() - started
            )
            self.store.branch_end(tid, "B", snap, {"kind": kind})
            self.store.decide(tid, "B", {"checkpoint_valid": True}, lambda *args: True)
            self.store.commit(tid, "B", self.store._trial(tid)["s0"]["state_hash"])
        except BaseException:
            self.store.recover()
            raise
        atomic_json(
            self.out / "checkpoints/latest.json",
            json.loads((self.out / "active.json").read_text()),
        )

    def persist_controller(self):
        state = copy.deepcopy(self.controller.data)
        self.publish(
            lambda runtime: runtime.metadata.update(controller=state),
            "controller_checkpoint",
        )

    def transition(self, phase, reason):
        self.controller.transition(phase, reason)
        self.persist_controller()

    def oracle_phase(self, phase):
        self.controller.transition(phase, "oracle boundary")
        # Durable observer journal. Counterfactual S0 itself remains immutable.
        self.store._append(
            "CONTROL_PHASE", controller=copy.deepcopy(self.controller.data)
        )

    def recover_controller(self):
        # Only events after the currently selected checkpoint can describe interrupted work.
        selected = self.store._pointer()["artifact"]["state_hash"]
        selected_rows = [
            i
            for i, r in enumerate(self.store.rows)
            if r["event"] in ("INIT", "PREPARE")
            and r.get("pointer", {}).get("artifact", {}).get("state_hash") == selected
        ]
        tail = self.store.rows[max(selected_rows, default=-1) + 1 :]
        journal = [r for r in tail if r["event"] == "CONTROL_PHASE"]
        phase = journal[-1]["controller"]["phase"] if journal else self.controller.phase
        if phase in {
            "SNAPSHOT",
            "PROPOSE",
            "BRANCH_A",
            "RESTORE_S0",
            "BRANCH_B",
            "COMPARE",
            "REJECT",
            "COMMIT_A",
            "COMMIT_B",
        }:
            self.controller.data["probes"] += 1
            self.controller.data["last_probe_fit_update"] = self.controller.data[
                "fit_updates"
            ]
            self.controller.data["evidence"].append(
                {"outcome": "RECOVERED_ORACLE", "phase": phase}
            )
        self.controller.recovery(phase)
        self.persist_controller()

    def fit_chunk(self, steps):
        def fit(runtime):
            with torch.no_grad():
                before = runtime.model().detach()
                params = {
                    n: p.detach().clone() for n, p in runtime.model.named_parameters()
                }
            for _ in range(steps):
                loss = unified_loss(runtime, self.target)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite FIT loss")
                loss.backward()
                if runtime.parameter_banks:
                    bank_optimizer_step(runtime, warmup=False)
                else:
                    runtime.optimizers["old"].step()
                    runtime.model.zero_grad(set_to_none=True)
                    runtime.model.project_parameters_()
                    runtime.metadata["clock"]["logical_iteration"] += 1
                    runtime.schedulers["actual"].step(
                        runtime.metadata["clock"]["logical_iteration"]
                    )
            with torch.no_grad():
                image = runtime.model().detach()
                metrics = control_metrics(image, self.target, self.cfg)
            update = sum(
                float((p.detach() - params[n]).square().sum())
                for n, p in runtime.model.named_parameters()
            )
            count = sum(p.numel() for p in runtime.model.parameters())
            runtime.metadata["diagnosis_inputs"] = dict(
                update_rms=(update / count) ** 0.5,
                render_change=float((image - before).abs().max()),
                residual_summary={
                    "rgb_sse": float((self.target - image).square().sum())
                },
            )
            self.controller.data["fit_updates"] += steps
            runtime.metadata["controller"] = copy.deepcopy(self.controller.data)
            runtime.metadata["fit_trajectory"].append(
                dict(metrics, updates=self.controller.data["fit_updates"])
            )

        self.publish(fit, "FIT")
        atomic_json(
            self.out / "trajectories/fit.json",
            self.store.active.metadata["fit_trajectory"],
        )

    def run(self):
        while self.controller.phase != "FINAL":
            phase = self.controller.phase
            data = self.controller.data
            remaining = self.cfg["termination"]["max_fit_updates"] - data["fit_updates"]
            if phase == "FIT":
                if remaining <= 0:
                    self.transition("SELECT", "compute budget exhausted")
                    continue
                if self.cfg["runtime"]["mode"] == "probe":
                    self.transition("DIAGNOSE", "explicit probe-only request")
                    continue
                self.fit_chunk(min(remaining, self.cfg["fit"]["chunk_steps"]))
                self.transition("DIAGNOSE", "fixed topology chunk complete")
            elif phase == "DIAGNOSE":
                active = self.store.active
                state = data
                op = self.cfg["oracle_probe"]
                capacity = min(
                    self.cfg["termination"]["max_primitives"] - active.model.count,
                    self.cfg["proposal"]["remaining_budget"],
                )
                permit = (
                    op["enabled"]
                    and state["probes"] < op["max_probes"]
                    and (
                        not active.parameter_banks
                        or active.metadata["cohorts"][-1]["joint_updates"]
                        >= self.cfg["birth"]["ramp_steps"]
                    )
                )
                last = state["last_probe_fit_update"]
                permit = permit and (
                    last is None or state["fit_updates"] - last >= op["cooldown_steps"]
                )
                if self.cfg["runtime"]["mode"] == "probe":
                    choice = {
                        "action": "PROBE_CAPACITY" if permit else "TERMINATE",
                        "reason": "explicit oracle request"
                        if permit
                        else "probe budget exhausted",
                    }
                elif (
                    self.cfg["runtime"]["mode"] == "fixed_n"
                    or not self.cfg["diagnosis"]["enabled"]
                ):
                    choice = {
                        "action": "FIT_MORE" if remaining > 0 else "TERMINATE",
                        "reason": "fixed-N policy",
                    }
                else:
                    choice = diagnose(
                        DiagnosisInput(
                            tuple(active.metadata["fit_trajectory"]),
                            previous_evidence=tuple(state["evidence"]),
                            candidate_preview=active.metadata.get("proposal"),
                            remaining_updates=remaining,
                            remaining_capacity=capacity,
                            probe_permitted=permit,
                            **active.metadata["diagnosis_inputs"]
                        ),
                        self.cfg,
                    )
                self.controller.data["last_diagnosis"] = choice
                action = choice["action"]
                dest = {
                    "FIT_MORE": "FIT",
                    "UNRESOLVED": "FIT",
                    "PROBE_CAPACITY": "SNAPSHOT",
                    "CONSOLIDATE": "CONSOLIDATE",
                    "TERMINATE": "SELECT",
                }[action]
                self.transition(dest, choice["reason"])
            elif phase == "SNAPSHOT":
                try:
                    result = run_capacity_probe(
                        self.store,
                        self.target,
                        max(
                            0,
                            min(
                                self.cfg["proposal"]["remaining_budget"],
                                self.cfg["termination"]["max_primitives"]
                                - self.store.active.model.count,
                            ),
                        ),
                        state_hook=self.oracle_phase,
                    )
                except BaseException as error:
                    self.store.recover()
                    self.store._append(
                        "CONTROL_FAILURE",
                        phase=self.controller.phase,
                        error_type=type(error).__name__,
                        reason=str(error),
                    )
                    raise
                self.controller.data["probes"] += 1
                self.controller.data["last_probe_fit_update"] = data["fit_updates"]
                self.controller.data["evidence"].append(
                    {
                        "outcome": result["outcome"],
                        "decision": result["decision"],
                        "K": result["K"],
                    }
                )
                atomic_json(
                    self.out
                    / "metrics"
                    / ("probe_%03d.json" % self.controller.data["probes"]),
                    result,
                )
                self.publish(
                    lambda runtime: runtime.metadata.update(
                        fit_trajectory=[], diagnosis_inputs={}
                    ),
                    "post_oracle_diagnosis_reset",
                )
                self.transition("CONSOLIDATE", "oracle transaction completed")
            elif phase == "CONSOLIDATE":
                steps = min(remaining, self.cfg["termination"]["consolidation_steps"])
                if steps > 0:
                    self.fit_chunk(steps)
                # B banks continue normally in FIT, but v1 never merges unequal Adan ages.
                self.transition(
                    "SELECT"
                    if self.cfg["runtime"]["mode"] == "probe" or remaining <= 0
                    else "FIT",
                    "consolidation boundary complete",
                )
            elif phase == "SELECT":
                self.transition(
                    "REPLAY", "select latest explicitly committed TrainingState"
                )
            elif phase == "REPLAY":
                replay(
                    self.store, self.target, self.cfg, self.out / "selected/replay.json"
                )
                self.transition("FINAL", "read-only selected replay passed")
            else:
                raise ValueError(
                    "resume must recover transient oracle phases before run"
                )
        atomic_json(
            self.out / "selected/pointer.json",
            json.loads((self.out / "active.json").read_text()),
        )
        result = replay(
            self.store, self.target, self.cfg, self.out / "selected/final_replay.json"
        )
        atomic_json(
            self.out / "final_manifest.json",
            dict(
                status="FINAL",
                selected=result,
                config_hash=config_hash(self.cfg),
                controller=self.controller.data,
                formal_training_started=self.cfg["runtime"]["formal_training"],
            ),
        )
        return result


def replay(store, target, cfg, path):
    before = snapshot_active(store.active)
    with torch.no_grad():
        image = store.active.model().detach()
        metrics = control_metrics(image, target, cfg)
    if snapshot_active(store.active).state_hash != before.state_hash:
        raise ValueError("replay mutated active TrainingState")
    result = dict(
        active_hash=before.state_hash,
        component_hashes={k: canonical_hash(v) for k, v in before.payload.items()},
        render_hash=canonical_hash(image),
        metrics=metrics,
        readonly=True,
    )
    atomic_json(path, result)
    return result
