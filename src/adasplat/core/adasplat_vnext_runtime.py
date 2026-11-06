"""Explicit Phase 0 renderer selection. No legacy conversion or training loop."""
import hashlib
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_renderer_binary(path, expected_sha256):
    """Must run before renderer imports; never replace an already loaded binary."""
    path = Path(path).resolve(strict=True)
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError("renderer binary hash mismatch")
    loaded = sys.modules.get("gsplat.csrc")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != path:
            raise RuntimeError("different renderer already loaded; use a fresh process")
    else:
        if any(k == "gsplat" or k.startswith("gsplat.") for k in sys.modules):
            raise RuntimeError(
                "gsplat already imported; select binary in a fresh process"
            )
        spec = importlib.util.spec_from_file_location("gsplat.csrc", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gsplat.csrc"] = module
    backend = importlib.import_module("gsplat.cuda._backend")
    if backend._C is not sys.modules["gsplat.csrc"]:
        raise RuntimeError("renderer backend does not match selected binary")
    return {"binary_path": str(path), "binary_sha256": expected_sha256.lower()}


def make_v42_model(target, count, learning_rates, *, alpha_cutoff, p_min, p_max, scale_mode='effective_support', support_mode='alpha'):
    """Call the native constructor directly, bypassing legacy adapter defaults."""
    if not (math.isfinite(alpha_cutoff) and 0 < alpha_cutoff < 1):
        raise ValueError("invalid explicit cutoff")
    if not (
        math.isfinite(p_min) and math.isfinite(p_max) and 0.1 <= p_min < p_max <= 8
    ):
        raise ValueError("numerical p domain must be within [0.1,8]")
    if p_min < 1.05 and scale_mode not in ('continuous_l2_rms','continuous_l2_rms_free_scale'):
        raise ValueError('subunit p requires continuous L2 scale mode')
    from adasplat_vnext_model import V42PMN

    args = SimpleNamespace(**learning_rates)
    args.alpha_cutoff = float(alpha_cutoff)
    if support_mode not in ('alpha','kernel'):raise ValueError('invalid support mode')
    args.support_mode=support_mode
    if scale_mode not in ('effective_support','continuous_l2_rms','continuous_l2_rms_free_scale'):
        raise ValueError('unsupported explicit scale mode')
    model = V42PMN(target, count, args, scale_mode)
    model.p_min, model.p_max = float(p_min), float(p_max)
    validate_model_runtime(
        model, {"alpha_cutoff": alpha_cutoff, "p_min": p_min, "p_max": p_max, 'scale_mode':scale_mode}
    )
    return model


def validate_model_runtime(model, profile):
    for name in ("alpha_cutoff", "p_min", "p_max"):
        if getattr(model, name) != profile[name]:
            raise ValueError("actual renderer profile mismatch: " + name)
    if model.v42_mode != profile.get('scale_mode','effective_support'):
        raise ValueError("unsupported representation; no implicit conversion")


def load_v42_tensors(
    model, payload, *, legacy_actual_cutoff=None, override_reason=None
):
    """Strict tensor load, with explicit acknowledgment of historical cutoff drift."""
    from primix_v4_2_parameterization import scale_mode_for_schema
    if scale_mode_for_schema(payload.get("parameterization_version")) != model.v42_mode:
        raise ValueError(
            "unsupported checkpoint representation; no implicit conversion"
        )
    declared = payload.get("alpha_cutoff")
    actual = declared if legacy_actual_cutoff is None else legacy_actual_cutoff
    if actual != declared and not (
        isinstance(override_reason, str) and override_reason.strip()
    ):
        raise ValueError("legacy cutoff override requires an evidence reason")
    if actual != model.alpha_cutoff:
        raise ValueError("checkpoint actual cutoff does not match runtime")
    if payload.get("p_min") != model.p_min or payload.get("p_max") != model.p_max:
        raise ValueError("checkpoint p domain does not match runtime")
    state = payload["state_dict"]
    current = model.state_dict()
    if state.keys() != current.keys() or any(
        state[k].shape != current[k].shape or state[k].dtype != current[k].dtype
        for k in current
    ):
        raise ValueError("checkpoint tensor schema mismatch")
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("nonfinite checkpoint tensor")
    model.load_state_dict(state, strict=True)
    return {
        "declared_cutoff": declared,
        "actual_cutoff": actual,
        "override_reason": override_reason,
        "conversion_applied": False,
    }


def selected_external(target_path, binary):
    """Live fingerprints for independent restoration, not copied from artifact."""
    import inspect
    from pytorch_msssim import ms_ssim

    root = Path(__file__).resolve().parent
    return dict(
        target_sha256=file_sha256(target_path),
        extension_hash=binary["binary_sha256"],
        code_hashes={
            name: file_sha256(root / name)
            for name in (
                "adasplat_training_state.py",
                "adasplat_vnext_runtime.py",
                "run_adasplat_vnext_probe.py",
                "adasplat_vnext_controller.py",
                "adasplat_vnext_primitives.py",
                "adasplat_vnext_config.py",
                "adasplat_vnext_trainer.py",
                "adasplat_vnext_growth.py",
                "adasplat_vnext_oracle.py",
                "adasplat_vnext_objective.py",
                "optimizer.py",
                "primix_v4_2_parameterization.py",
                "adasplat_vnext_model.py",
                "gsplat/gsplat/rasterize_sum.py",
            )
        },
        metric_source_sha256=file_sha256(inspect.getfile(ms_ssim)),
        environment=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0),
        ),
    )


def v42_state_factory(target):
    from adasplat_training_state import ActiveRuntime, clone_tree
    from adasplat_vnext_primitives import StageCosineController

    def factory(data):
        cfg = data["metadata"]["renderer_runtime"]
        model = make_v42_model(
            target,
            data["metadata"]["population"]["N"],
            cfg["learning_rates"],
            alpha_cutoff=cfg["alpha_cutoff"],
            p_min=cfg["p_min"],
            p_max=cfg["p_max"],
            scale_mode=cfg.get('scale_mode','effective_support'),
        )
        scheduler = StageCosineController(
            model.optimizer,
            data["schedulers"]["actual"]["state_dict"]["stage_max_iterations"],
            data["schedulers"]["actual"]["state_dict"]["floor"],
        )
        generators = {
            key: torch.Generator(device=saved["device"])
            for key, saved in data["named_generators"].items()
        }
        if data.get("parameter_banks"):
            from adasplat_vnext_controller import CohortBank

            banks = {
                key: CohortBank(model, saved["rows"])
                for key, saved in data["parameter_banks"].items()
            }
            optimizers = {
                key: bank.optimizer(model.optimizer) for key, bank in banks.items()
            }
            model.optimizer = optimizers["old"]
            scheduler = StageCosineController(
                optimizers["old"],
                data["schedulers"]["actual"]["state_dict"]["stage_max_iterations"],
                data["schedulers"]["actual"]["state_dict"]["floor"],
            )
            return ActiveRuntime(
                model,
                optimizers,
                {"actual": scheduler},
                clone_tree(data["metadata"]),
                clone_tree(data["external"]),
                generators,
                banks,
            )
        return ActiveRuntime(
            model,
            {"old": model.optimizer},
            {"actual": scheduler},
            clone_tree(data["metadata"]),
            clone_tree(data["external"]),
            generators,
        )

    return factory


def validate_v42_state(runtime):
    cfg = runtime.metadata["renderer_runtime"]
    validate_model_runtime(runtime.model, cfg)
    if runtime.model.count != runtime.metadata["population"]["N"]:
        raise ValueError("population mismatch")
    if (
        runtime.model.block != cfg["block_size"]
        or cfg["alpha_cap"] != 1
        or cfg["output_clip"] != [0, 1]
    ):
        raise ValueError("renderer non-tensor contract mismatch")
    expected = dict(
        protocol="rgb_fp32_msssim11_v1",
        win_size=11,
        win_sigma=1.5,
        weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        K=[0.01, 0.03],
        data_range=1,
        dtype="float32",
        channels="RGB",
        tf32=False,
    )
    if "vnext_config" in runtime.metadata:
        from adasplat_vnext_config import config_hash

        config = runtime.metadata["vnext_config"]
        if config_hash(config) != runtime.metadata["config_hash"]:
            raise ValueError("config hash mismatch")
        ev = config["evaluation"]
        expected.update(
            win_size=ev["ms_window"],
            win_sigma=ev["ms_sigma"],
            weights=ev["ms_weights"],
            K=ev["ms_K"],
        )
    protocol = runtime.metadata["metric_runtime"]
    if {k: v for k, v in protocol.items() if k != "D_ref"} != expected:
        raise ValueError("selected metric protocol mismatch")
    if "D_ref" in protocol and not (
        math.isfinite(protocol["D_ref"]) and protocol["D_ref"] >= 1e-8
    ):
        raise ValueError("invalid frozen D_ref")


def create_selected_legacy_stage(
    payload, target, target_path, binary, actual_cutoff, reason
):
    """Create a declared new experiment stage with old tensors and Adan moments.

    The historical actual cosine is unavailable. Start a new constant-LR stage
    at logical 60000 using each saved optimizer LR; never infer old scheduler age.
    """
    import random
    import numpy as np
    from adasplat_training_state import ActiveRuntime, canonical_hash
    from adasplat_vnext_primitives import StageCosineController

    rates = {
        k: payload["PMN_config"][k]
        for k in ("xyz_lr", "scale_lr", "rotation_lr", "opacity_lr", "color_lr", "p_lr")
    }
    model = make_v42_model(
        target,
        len(payload["state_dict"]["p"]),
        rates,
        alpha_cutoff=actual_cutoff,
        p_min=payload["p_min"],
        p_max=payload["p_max"],
        scale_mode=__import__('primix_v4_2_parameterization').scale_mode_for_schema(payload['parameterization_version']),
    )
    loading = load_v42_tensors(
        model, payload, legacy_actual_cutoff=actual_cutoff, override_reason=reason
    )
    model.optimizer.load_state_dict(payload["optimizer"])
    if canonical_hash(model.optimizer.state_dict()) != canonical_hash(
        payload["optimizer"]
    ):
        raise ValueError("legacy Adan state changed on load")
    from adasplat_vnext_config import DEFAULTS

    scheduler = StageCosineController(
        model.optimizer,
        DEFAULTS["oracle_probe"]["old_updates"],
        DEFAULTS["optimizer"]["floor"],
    )
    scheduler.restart(60000)
    random.seed(20260906)
    np.random.seed(20260906)
    torch.manual_seed(20260906)
    generators = {"proposal": torch.Generator(device="cuda").manual_seed(20260906)}
    cfg = dict(
        learning_rates=rates,
        alpha_cutoff=actual_cutoff,
        p_min=payload["p_min"],
        p_max=payload["p_max"],
        block_size=model.block,
        alpha_cap=1,
        output_clip=[0, 1],
        scale_mode=model.v42_mode,
    )
    metadata = dict(
        clock=dict(logical_iteration=60000, phase="FIT", new_experiment_stage=True),
        cohorts=[dict(id="legacy_all", start=0, end=model.count, birth_iteration=0)],
        population=dict(N=model.count),
        residual=None,
        proposal=None,
        controller=dict(fit_windows=[], diagnosis=None),
        utility=dict(accepted=None),
        selected_accepted_topology=dict(
            id="legacy_baseline", N=model.count, logical_iteration=60000
        ),
        renderer_runtime=cfg,
        metric_runtime=dict(
            protocol="rgb_fp32_msssim11_v1",
            win_size=11,
            win_sigma=1.5,
            weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            K=[0.01, 0.03],
            data_range=1,
            dtype="float32",
            channels="RGB",
            tf32=False,
        ),
        legacy_loading=loading,
        legacy_cohorts_hash=canonical_hash(payload.get("growth_cohorts")),
    )
    return ActiveRuntime(
        model,
        {"old": model.optimizer},
        {"actual": scheduler},
        metadata,
        selected_external(target_path, binary),
        generators,
    )
