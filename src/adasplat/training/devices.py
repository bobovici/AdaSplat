"""Select visible GPUs without importing CUDA into the batch supervisor."""
import os
import subprocess
import sys


def resolve_gpus(value):
    if value != 'auto':
        return [int(v.strip()) for v in value.split(',')]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        tokens = [v.strip() for v in visible.split(',') if v.strip()]
        if not tokens or tokens == ['-1']:
            raise ValueError('No GPUs visible in CUDA_VISIBLE_DEVICES')
        # Public --gpus uses physical integer selectors. Preserve a restricted mask.
        if not all(v.isdigit() for v in tokens):
            raise ValueError('UUID/MIG masks require an explicit compatible GPU mapping; use integer CUDA_VISIBLE_DEVICES')
        return [int(v) for v in tokens]
    probe = subprocess.run([sys.executable,'-c',
        'import torch; print(torch.cuda.device_count())'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if probe.returncode:
        raise RuntimeError('Cannot detect CUDA GPUs. Install CUDA PyTorch or pass --gpus explicitly. '+probe.stderr)
    count = int(probe.stdout.strip().splitlines()[-1])
    if not count:
        raise ValueError('No CUDA GPUs found')
    return list(range(count))
