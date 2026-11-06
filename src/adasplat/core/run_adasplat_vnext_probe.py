"""Independent read-only checkpoint replay. No optimizer updates or final projection."""
import argparse
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'gsplat'))

import numpy as np
import torch
from PIL import Image

from adasplat_training_state import canonical_hash, snapshot_active, TrialStore, capture_rng, restore_rng
from adasplat_vnext_runtime import (file_sha256, select_renderer_binary,
    make_v42_model, load_v42_tensors, selected_external, v42_state_factory,
    validate_v42_state, create_selected_legacy_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selected-store', type=Path)
    parser.add_argument('--create-selected', action='store_true')
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--actual-cutoff', type=float)
    parser.add_argument('--override-reason')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('probe output must be a new file')
    binary = select_renderer_binary(args.binary, args.binary_sha256)
    from pytorch_msssim import ms_ssim
    torch.set_num_threads(2)
    torch.manual_seed(20260906)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not args.selected_store and not args.checkpoint:
        parser.error('checkpoint or selected-store required')
    if args.create_selected and (not args.selected_store or not args.checkpoint):
        parser.error('create-selected requires checkpoint and selected-store')
    source = args.checkpoint if not args.selected_store else args.selected_store / 'active.json'
    target_hash = file_sha256(args.target)
    target = torch.from_numpy(np.array(Image.open(args.target).convert('RGB')).copy())
    target = target.permute(2,0,1).unsqueeze(0).float().cuda()/255.
    runtime = None
    if args.selected_store:
        if args.create_selected:
            payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            runtime=create_selected_legacy_stage(payload,target,args.target,binary,args.actual_cutoff,args.override_reason)
        with TrialStore(args.selected_store,v42_state_factory(target),selected_external(args.target,binary),
                        validate_v42_state) as store:
            if args.create_selected: store.initialize(runtime)
            runtime=store.recover()
        model=runtime.model
        loading=runtime.metadata['legacy_loading']
        active_before=snapshot_active(runtime).state_hash
    else:
        payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        rates = dict(xyz_lr=.0015, scale_lr=.0015, rotation_lr=.001,
                     opacity_lr=.001, color_lr=.0015, p_lr=.001)
        model = make_v42_model(target, len(payload['state_dict']['p']), rates,
                              alpha_cutoff=args.actual_cutoff,
                              p_min=payload['p_min'], p_max=payload['p_max'])
        loading = load_v42_tensors(model, payload, legacy_actual_cutoff=args.actual_cutoff,
                                  override_reason=args.override_reason)
    source_hash=file_sha256(source)
    before = canonical_hash(model.state_dict())
    if runtime is None and before != canonical_hash(payload['state_dict']):
        raise RuntimeError('checkpoint tensor content changed on load')
    weights = [.0448, .2856, .3001, .2363, .1333]
    with torch.no_grad():
        image = model()
        repeat = model()
        mse = float((image-target).square().mean())
        metrics = dict(MSE=mse, PSNR=-10*math.log10(max(mse,1e-20)),
            MS_SSIM=float(ms_ssim(image,target,data_range=1,size_average=True,
                win_size=11,win_sigma=1.5,weights=weights,K=(.01,.03))))
        repeat_delta = float((repeat-image).abs().max())
    if canonical_hash(model.state_dict()) != before or file_sha256(source) != source_hash:
        raise RuntimeError('read-only replay changed checkpoint/model')
    if file_sha256(args.target) != target_hash:
        raise RuntimeError('target changed during replay')
    if runtime is not None and snapshot_active(runtime).state_hash != active_before:
        raise RuntimeError('replay changed complete active TrainingState')
    result = dict(schema='adasplat_phase0_replay_v1', binary=binary,
        checkpoint_sha256=source_hash, target_sha256=target_hash,
        checkpoint_tensor_sha256=before, rendered_tensor_sha256=canonical_hash(image),
        loading=loading, metrics=metrics, repeat_render_max_delta=repeat_delta,
        readonly=True, formal_training_started=False,
        metric_protocol=dict(torch=torch.__version__, cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0), dtype='float32', image='RGB/255',
            tf32=False, data_range=1, win_size=11, win_sigma=1.5, weights=weights,
            K=[.01,.03], ms_ssim_source_sha256=file_sha256(inspect.getfile(ms_ssim))))
    if runtime is not None:
        import random
        saved_rng=capture_rng()
        named_states={k:g.get_state() for k,g in runtime.named_generators.items()}
        try:
            next_draws=dict(python=random.random(),numpy=np.random.rand(5),cpu=torch.rand(5),
                cuda=[torch.rand(5,device='cuda:'+str(i)) for i in range(torch.cuda.device_count())],
                named={k:torch.rand(5,generator=g,device=g.device) for k,g in runtime.named_generators.items()})
            result['next_rng_sha256']=canonical_hash(next_draws)
        finally:
            restore_rng(saved_rng)
            for k,g in runtime.named_generators.items(): g.set_state(named_states[k])
        result['selected_active_hash']=active_before
        result['active_after_replay_hash']=snapshot_active(runtime).state_hash
        result['selected_components']={k:canonical_hash(v) for k,v in snapshot_active(runtime).payload.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
