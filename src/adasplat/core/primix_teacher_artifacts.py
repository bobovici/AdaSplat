"""Compact, crash-safe artifact contract for Dynamic PMN teacher runs.

This module has no optimisation policy.  It only owns paths, serialisation,
coordinate labels, and validation-friendly metadata.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


SCHEMA = "primix_teacher_artifact_v1"


def layout(run_dir):
    root = Path(run_dir)
    return {"root": root, "checkpoints": root / "checkpoints", "labels": root / "labels",
            "trajectory": root / "trajectory", "recon": root / "recon",
            "statistics": root / "statistics"}


def ensure_layout(run_dir):
    paths = layout(run_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _fsync_file(handle):
    handle.flush()
    os.fsync(handle.fileno())


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        _fsync_file(handle)
    os.replace(temporary, path)


def atomic_torch_save(path, payload, rotate_latest=False):
    """Save a checkpoint atomically; rotate an already-good latest to prev."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        _fsync_file(handle)
    if rotate_latest and path.exists():
        previous = path.with_name("latest_prev.pt")
        os.replace(path, previous)
    os.replace(temporary, path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def primitive_state_from_state_dict(state_dict):
    """Canonical, renderer-neutral names while retaining source-state names."""
    required = {"xys": "xy", "scale_raw": "scale_raw", "rotation_angle": "rotation_angle",
                "p": "p", "mass": "mass", "colors": "color"}
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise KeyError("primitive state missing fields: %s" % missing)
    return {target: state_dict[source].detach().cpu().contiguous() for source, target in required.items()}


def coordinate_payload(primitive_state, width, height, metadata):
    xy = primitive_state["xy"].detach().float().cpu().numpy().astype(np.float32, copy=False)
    normalized = xy.copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)
    base = {"xy_pixel": xy, "width": np.int64(width), "height": np.int64(height), "N": np.int64(len(xy))}
    for key in ("image_id", "iteration", "PSNR", "MSE", "MS_SSIM", "coordinate_convention"):
        if key in metadata and metadata[key] is not None:
            base[key] = metadata[key]
    normalized_base = dict(base)
    normalized_base["xy_normalized"] = normalized
    normalized_base["coordinate_convention"] = "xy_normalized=x/width,y/height; pixel_xy=[0,width)x[0,height)"
    return base, normalized_base


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(str(temporary), **arrays)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_final_coordinates(run_dir, primitive_state, width, height, metadata):
    paths = ensure_layout(run_dir)
    pixel, normalized = coordinate_payload(primitive_state, width, height, metadata)
    pixel_path = paths["labels"] / "coordinates_final.npz"
    normalized_path = paths["labels"] / "coordinates_final_normalized.npz"
    atomic_npz(pixel_path, **pixel); atomic_npz(normalized_path, **normalized)
    return pixel_path, normalized_path


def stage_snapshot_path(run_dir, stage_id):
    return ensure_layout(run_dir)["trajectory"] / ".coordinate_stages" / ("stage_%03d.npz" % int(stage_id))


def save_stage_coordinates(run_dir, stage_id, primitive_state, width, height, metadata):
    path = stage_snapshot_path(run_dir, stage_id)
    payload, _ = coordinate_payload(primitive_state, width, height, metadata)
    payload.update({"stage_id": np.int64(stage_id), "phase": str(metadata.get("phase", "unknown")),
                    "growth_index": np.int64(metadata.get("growth_index", -1))})
    atomic_npz(path, **payload)
    return path


def pack_trajectory_coordinates(run_dir):
    paths = ensure_layout(run_dir)
    stage_dir = paths["trajectory"] / ".coordinate_stages"
    entries = []
    packed = {}
    for source in sorted(stage_dir.glob("stage_*.npz")) if stage_dir.exists() else []:
        with np.load(source, allow_pickle=False) as data:
            stage = int(data["stage_id"])
            prefix = "stage_%d" % stage
            packed[prefix + "_xy"] = data["xy_pixel"].astype(np.float32, copy=False)
            packed[prefix + "_N"] = data["N"]
            iteration = data["iteration"] if "iteration" in data else np.int64(-1)
            packed[prefix + "_iteration"] = iteration
            entry = {"stage_id": stage, "N": int(data["N"]),
                     "iteration": int(iteration),
                     "coordinate_label_path": "labels/trajectory_coordinates.npz"}
            for key in ("PSNR", "MSE", "MS_SSIM", "growth_index", "phase"):
                if key not in data:
                    continue
                value = data[key]
                if key == "phase":
                    entry[key] = str(value.item() if getattr(value, "ndim", 0) == 0 else value)
                elif key == "growth_index":
                    entry[key] = int(value)
                else:
                    entry[key] = float(value)
            entries.append(entry)
    target = paths["labels"] / "trajectory_coordinates.npz"
    if packed:
        atomic_npz(target, **packed)
    return target, entries


def primitive_count(payload):
    state = payload.get("primitive_state") or primitive_state_from_state_dict(payload["state_dict"])
    return int(state["xy"].shape[0])


def relative(path, root):
    return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
