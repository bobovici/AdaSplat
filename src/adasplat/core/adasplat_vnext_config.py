"""PRODUCTION VNEXT v1 configuration. No implicit legacy CLI defaults."""
import copy, hashlib, json, math

DEFAULTS = {
    "schema": "adasplat_vnext_v1",
    "representation": {
        "schema": "v4_2_exact_l2_effective_support",
        "p_min": 1.05,
        "p_max": 3.0,
    },
    "renderer": {
        "binary": "analysis/vnext_implementation_plan/native_build/lib/gsplat/csrc.pyd",
        "sha256": "25e0a18a76ff95a99d5b884490a1d5c168215ee34a7c545bdc9e3fadb719429a",
        "cutoff": 1 / 255,
    },
    "optimizer": {
        "name": "Adan",
        "adan": {
            "betas": [0.98, 0.92, 0.99],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 0.0,
            "no_prox": False,
            "foreach": True,
            "fused": False,
        },
        "rates": {
            "xyz_lr": 0.0015,
            "scale_lr": 0.0015,
            "rotation_lr": 0.001,
            "opacity_lr": 0.001,
            "color_lr": 0.0015,
            "p_lr": 0.001,
        },
        "legacy_saved_lr": True,
        "floor": 1.0,
    },
    "loss": {"ssim_weight": 0.05, "ssim_window": 7,
             "ssim_gaussian": False, "gradient_weight": 0.0, "scale": 1.0,
             "structure_patch_size": 0, "structure_seed": 20261002},
    "fit": {
        "chunk_steps": 100,
        "initial_count": 1024,
        "init": "quadtree",
        "legacy_checkpoint": None,
        "logical_start": 0,
        "quadtree": {
            "min_depth": 3,
            "max_depth": 5,
            "split_factor": 1.5,
            "min_size": 32,
            "min_per_leaf": 1,
            "pixel_beta": 1.0,
        },
    },
    "diagnosis": {
        "enabled": True,
        "window": 3,
        "significant_relative_improvement": None,
        "plateau_relative_improvement": None,
        "max_update_rms": None,
        "max_render_change": None,
        "unresolved_action": "FIT_MORE",
    },
    "proposal": {"seed": 20260906, "remaining_budget": 50000},
    "birth": {
        "context_coefficient": 0.75,
        "context_complexity_beta": 2.0,
        "context_factor_min": 0.1,
        "context_factor_max": 1.25,
        "warmup_steps": 64,
        "ramp_steps": 256,
        "opacity": 0.05,
        "p": 2.0,
        "scale_min": 2.0,
        "scale_max": 8.0,
        "live_norm": 1e-12,
        "live_fraction": 0.9,
        "live_mean": 1e-10,
        "render_atol": 1e-6,
        "mse_atol": 1e-8,
        "mse_relative": 1e-4,
    },
    "micro_growth": {
        "fraction": 0.02,
        "max_k": 512,
        "core_size": 128,
        "core_cap": 64,
        "halo": 64,
        "min_distance": 2.0,
        "sampling_multiplier": 8,
    },
    "oracle_probe": {
        "enabled": False,
        "max_probes": 1,
        "cooldown_steps": 4096,
        "old_updates": 4096,
        "evaluation_steps": [256, 1024, 2048, 4096],
        "noise_repeats": 10,
    },
    "acceptance": {
        "lambda": 0.002,
        "delta": 0.002,
        "epsilon_min": 1e-6,
        "epsilon_max": 1e-4,
        "noise_multiplier": 10.0,
        "noise_ms_tolerance": 1e-6,
        "k0_render_tolerance": 1e-6,
        "k0_mse_tolerance": 1e-8,
        "k0_ms_tolerance": 1e-6,
        "ms_tolerance": 1e-4,
        "psnr_tolerance": 0.01,
        "tail_relative": 1e-4,
        "tail_absolute": 1e-8,
    },
    "termination": {
        "max_fit_updates": 1000,
        "max_primitives": 50000,
        "consolidation_steps": 0,
    },
    "checkpoint": {"output": "outputs/vnext/run", "every_chunks": 1},
    "runtime": {
        "seed": 20260906,
        "device": "cuda",
        "threads": 2,
        "smoke": False,
        "formal_training": False,
        "synthetic_size": None,
        "mode": "dynamic",
    },
    "evaluation": {
        "target": None,
        "tail_area": 0.1,
        "mse_weight": 0.8,
        "tail_weight": 0.2,
        "core_size": 128,
        "ms_window": 11,
        "ms_sigma": 1.5,
        "ms_weights": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        "ms_K": [0.01, 0.03],
    },
}


def defaults():
    return copy.deepcopy(DEFAULTS)


def validate(config):
    config = copy.deepcopy(config)
    # Existing recipes retain their original objective; migration is additive.
    config['loss'].setdefault('ssim_gaussian', False)
    config['loss'].setdefault('gradient_weight', 0.0)
    config['loss'].setdefault('scale', 1.0)
    config['loss'].setdefault('structure_patch_size', 0)
    config['loss'].setdefault('structure_seed', 20261002)
    def visit(value, template, path=""):
        if isinstance(template, dict):
            if not isinstance(value, dict) or value.keys() != template.keys():
                raise ValueError("config keys: " + path)
            for key in template:
                visit(value[key], template[key], path + "." + key)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite config: " + path)

    visit(config, DEFAULTS)
    loss = config['loss']
    if type(loss['structure_patch_size']) is not int or (loss['structure_patch_size']!=0 and loss['structure_patch_size']<loss['ssim_window']):
        raise ValueError('structure patch must be zero or at least SSIM window')
    if type(loss['structure_seed']) is not int or not 0<=loss['structure_seed']<2**32:
        raise ValueError('invalid structure sampling seed')
    if isinstance(loss['scale'],bool) or not isinstance(loss['scale'],(int,float)) or loss['scale']<=0:
        raise ValueError('loss.scale must be positive')
    if type(loss['ssim_gaussian']) is not bool:
        raise ValueError('loss.ssim_gaussian must be boolean')
    for key in ('ssim_weight', 'gradient_weight'):
        if isinstance(loss[key], bool) or not isinstance(loss[key], (float, int)) or loss[key] < 0:
            raise ValueError('loss weights must be nonnegative numbers')
    if type(loss['ssim_window']) is not int or loss['ssim_window'] < 3 or loss['ssim_window'] % 2 != 1:
        raise ValueError('SSIM window must be odd and >=3')
    if config["schema"] != DEFAULTS["schema"]:
        raise ValueError("unsupported config schema")
    if config["representation"]["schema"] not in ('v4_2_exact_l2_effective_support','v4_2_exact_l2_continuous_l2_rms','v4_2_exact_l2_continuous_l2_rms_free_scale'):
        raise ValueError("representation migration not implemented")
    for section, key in [
        ("fit", "chunk_steps"),
        ("oracle_probe", "old_updates"),
        ("birth", "warmup_steps"),
        ("birth", "ramp_steps"),
        ("micro_growth", "core_size"),
        ("micro_growth", "sampling_multiplier"),
        ("diagnosis", "window"),
    ]:
        if not isinstance(config[section][key], int) or config[section][key] < 1:
            raise ValueError("positive integer required: " + section + "." + key)
    if config["diagnosis"]["unresolved_action"] != "FIT_MORE":
        raise ValueError("unresolved evidence cannot trigger oracle")
    if config["optimizer"]["floor"] != 1 or config["optimizer"]["name"] != "Adan":
        raise ValueError("v1 uses constant old LR and Adan")
    if config["checkpoint"]["every_chunks"] != 1:
        raise ValueError("v1 checkpoints every transactional chunk")
    if config["runtime"]["mode"] not in ("dynamic", "fixed_n", "probe"):
        raise ValueError("invalid mode")
    if config["runtime"]["smoke"] and (
        config["termination"]["max_fit_updates"] > 10
        or config["oracle_probe"]["old_updates"] > 10
        or config["birth"]["warmup_steps"] > 10
    ):
        raise ValueError("smoke must be <=10 steps")
    if config["termination"]["max_primitives"] < config["fit"]["initial_count"]:
        raise ValueError("initial count exceeds capacity")
    if config["runtime"]["formal_training"] and (
        config["runtime"]["smoke"] or config["runtime"]["mode"] == "probe"
    ):
        raise ValueError("probe/smoke cannot be marked formal training")
    if config["runtime"]["device"] != "cuda":
        raise ValueError("native sparse renderer requires CUDA")
    if config["birth"]["p"] != 2.0:
        raise ValueError("v1 local projection preview requires p=2 newborns")
    pmin, pmax = config["representation"]["p_min"], config["representation"]["p_max"]
    if not 0.1 <= pmin < 2 <= pmax <= 8:
        raise ValueError("unsupported p domain for p=2 birth")
    if pmin < 1.05 and config['representation']['schema'] not in ('v4_2_exact_l2_continuous_l2_rms','v4_2_exact_l2_continuous_l2_rms_free_scale'):
        raise ValueError('subunit p requires continuous L2 scale mode')
    for section, key in [
        ("termination", "max_fit_updates"),
        ("termination", "consolidation_steps"),
        ("oracle_probe", "max_probes"),
        ("oracle_probe", "cooldown_steps"),
        ("proposal", "remaining_budget"),
        ("micro_growth", "max_k"),
    ]:
        if type(config[section][key]) is not int or config[section][key] < 0:
            raise ValueError("nonnegative integer required: " + section + "." + key)
    for key, value in config["diagnosis"].items():
        if (
            key
            in (
                "significant_relative_improvement",
                "plateau_relative_improvement",
                "max_update_rms",
                "max_render_change",
            )
            and value is not None
            and (not isinstance(value, (int, float)) or value < 0)
        ):
            raise ValueError("invalid diagnosis threshold")
    micro = config["micro_growth"]
    birth = config["birth"]
    ac = config["acceptance"]
    if (
        not 0 <= micro["fraction"] <= 1
        or micro["min_distance"] < 0
        or micro["core_cap"] < 1
    ):
        raise ValueError("invalid proposal budget")
    if (
        not 0 < config["renderer"]["cutoff"] < birth["opacity"] <= 1
        or not 0.25 < birth["scale_min"] <= birth["scale_max"]
    ):
        raise ValueError("invalid birth support")
    required_halo = math.ceil(
        birth["scale_max"]
        * math.sqrt(2 * math.log(birth["opacity"] / config["renderer"]["cutoff"]))
    )
    if micro["halo"] < required_halo:
        raise ValueError("halo does not cover newborn support")
    if (
        not 0 < birth["live_fraction"] <= 1
        or birth["live_norm"] < 0
        or birth["live_mean"] < 0
    ):
        raise ValueError("invalid activity gate")
    if not 0 < ac["epsilon_min"] <= ac["epsilon_max"] or any(
        v < 0 for v in ac.values()
    ):
        raise ValueError("invalid acceptance thresholds")
    if config["evaluation"]["tail_area"] <= 0 or config["evaluation"]["tail_area"] > 1:
        raise ValueError("invalid tail area")
    if (
        abs(
            config["evaluation"]["mse_weight"] + config["evaluation"]["tail_weight"] - 1
        )
        > 1e-12
    ):
        raise ValueError("objective weights must sum to one")
    return copy.deepcopy(config)


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return validate(json.load(f))


def config_hash(config):
    return hashlib.sha256(
        json.dumps(
            validate(config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def settings(runtime=None):
    return (
        runtime.metadata.get("vnext_config", DEFAULTS)
        if runtime is not None
        else DEFAULTS
    )
