"""Unique PRODUCTION VNEXT entrypoint. Historical runners are read-only evidence."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gsplat"))
from adasplat_vnext_config import load, config_hash, validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR")
    parser.add_argument(
        "--selected",
        type=Path,
        metavar="RUN_DIR",
        help="restore selected final state; implies replay-only",
    )
    parser.add_argument("--replay-only", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--probe-only", action="store_true")
    modes.add_argument("--fixed-n-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    existing = args.selected or args.resume
    if existing and args.config:
        parser.error("resume uses saved run_config; no config replacement")
    if existing and (args.probe_only or args.fixed_n_only):
        parser.error("resume cannot change saved experiment mode")
    if not existing and not args.config:
        parser.error("--config required for new run")
    cfg = load(existing / "run_config.json" if existing else args.config)
    if args.probe_only:
        cfg["runtime"]["mode"] = "probe"
        cfg["oracle_probe"]["enabled"] = True
    if args.fixed_n_only:
        cfg["runtime"]["mode"] = "fixed_n"
    validate(cfg)
    out = Path(cfg["checkpoint"]["output"])
    if existing and out.resolve() != existing.resolve():
        parser.error("run directory must match saved config")
    if args.dry_run:
        print(
            json.dumps(
                dict(config_hash=config_hash(cfg), config=cfg, training_started=False),
                indent=2,
            )
        )
        return
    if args.replay_only and not existing:
        parser.error("replay requires --resume or --selected")
    import torch
    from adasplat_vnext_runtime import (
        select_renderer_binary,
        v42_state_factory,
        validate_v42_state,
    )
    from adasplat_training_state import TrialStore
    from adasplat_vnext_trainer import (
        target_from_config,
        external_contract,
        create_runtime,
        Trainer,
        replay,
    )
    from primix_teacher_artifacts import atomic_json

    torch.set_num_threads(cfg["runtime"]["threads"])
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    profile = select_renderer_binary(
        cfg["renderer"]["binary"], cfg["renderer"]["sha256"]
    )
    target = target_from_config(cfg)
    external = external_contract(cfg, profile, target)
    if not existing:
        if out.exists():
            raise FileExistsError("new run directory required; use --resume")
        out.mkdir(parents=True)
        for directory in [
            "checkpoints",
            "selected",
            "metrics",
            "trajectories",
            "figures",
        ]:
            (out / directory).mkdir()
        atomic_json(out / "run_config.json", cfg)
        atomic_json(out / "runtime_manifest.json", external)
    elif json.loads((out / "runtime_manifest.json").read_text()) != external:
        raise ValueError(
            "live config/code/binary/target/environment fingerprint mismatch"
        )
    with TrialStore(
        out, v42_state_factory(target), external, validate_v42_state
    ) as store:
        if existing:
            store.recover()
        else:
            store.initialize(create_runtime(cfg, target, profile, external))
        if args.selected or args.replay_only:
            result = replay(
                store, target, cfg, out / "selected/independent_replay.json"
            )
            print(json.dumps(result, indent=2))
            return
        trainer = Trainer(store, target, cfg)
        if existing:
            trainer.recover_controller()
        result = trainer.run()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
