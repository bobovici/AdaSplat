"""Optional post-training codec stage. Training decisions remain unchanged."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import ROOT, resolve_plan


def check_codec_dependency():
    if importlib.util.find_spec('constriction') is None:
        raise RuntimeError('Encoding requested: install requirements-codec.txt before training')


def encode_completed_run(config, output, target, runner=subprocess.run):
    output = Path(output).resolve()
    status = output/'encoding.json'
    started = time.monotonic()
    paths = [output/'final.pt']
    if config['encoding']['milestones']:
        index = output/'milestones/index.json'
        if index.exists():
            entries=json.loads(index.read_text(encoding='utf-8'))['entries']
            paths += [output/e['checkpoint'] for e in entries if e['status']=='saved']
    jobs=[]
    temporary=None
    try:
        if config['preprocess']['smoke_crop']:
            from PIL import Image
            plan=resolve_plan(config,target)
            temporary=tempfile.TemporaryDirectory(prefix='adasplat-smoke-target-')
            cropped=Path(temporary.name)/'target.png'
            with Image.open(target) as image:
                image.convert('RGB').crop((0,0,plan['width'],plan['height'])).save(cropped)
            target=cropped
        for checkpoint in paths:
            relative=checkpoint.relative_to(output)
            stream=output/'encoded'/relative.with_suffix('.adagc')
            decoded=stream.with_suffix('.png')
            report=stream.with_suffix('.adagc.json')
            command=[sys.executable,str(ROOT/'codec.py'),'encode',
                '--checkpoint',str(checkpoint),'--target',str(target),'--output',str(stream),
                '--decoded-output',str(decoded),'--max-loss-db',str(config['encoding']['max_loss_db'])]
            # Inherit CUDA_VISIBLE_DEVICES from the single-image worker.
            stream.parent.mkdir(parents=True,exist_ok=True)
            log=stream.with_suffix('.console.log')
            with log.open('w',encoding='utf-8') as handle:
                result=runner(command,stdout=handle,stderr=subprocess.STDOUT,cwd=str(ROOT))
            if result.returncode:
                raise RuntimeError('Encoding failed; trained checkpoints retained. See '+str(log))
            data=json.loads(report.read_text(encoding='utf-8'))
            jobs.append(dict(checkpoint=relative.as_posix(),stream=str(stream.relative_to(output)),
                decoded=str(decoded.relative_to(output)),bytes=data['bytes'],bpp=data['bpp'],
                loss_db=data['loss_db'],decoded_RGB8_PSNR=data.get('decoded_RGB8_PSNR'),original_RGB8_PSNR=data.get('original_RGB8_PSNR'),report=str(report.relative_to(output))))
        record=dict(status='complete',seconds=time.monotonic()-started,artifacts=jobs)
    except Exception as error:
        record=dict(status='failed',seconds=time.monotonic()-started,artifacts=jobs,error=str(error),
            training_complete=True,training_outputs_retained=True)
        status.write_text(json.dumps(record,indent=2),encoding='utf-8')
        raise
    finally:
        if temporary is not None:temporary.cleanup()
    status.write_text(json.dumps(record,indent=2),encoding='utf-8')
    summary=output/'summary.json'
    if summary.exists():
        data=json.loads(summary.read_text(encoding='utf-8'));data['encoding']=record
        summary.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(dict(event='encoding_complete',**record)),flush=True)
    return record
