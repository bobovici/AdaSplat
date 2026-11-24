"""GPU process scheduling, retries and summaries. No training math here."""
import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .config import ROOT, resolve_plan
from .run_logging import RunLogger
from .comparison import format_comparison, paired_summary
sys.path.insert(0,str(ROOT/'tools'))
from atomic_files import atomic_json, atomic_output

EXTENSIONS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}


def enumerate_images(root,recursive=False,patterns=None):
    root = Path(root).resolve(strict=True)
    if patterns:
        found = []
        for pattern in patterns:
            found.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    else:
        found = root.rglob('*') if recursive else root.iterdir()
    images = {p.resolve() for p in found if p.is_file() and p.suffix.lower() in EXTENSIONS}
    # Symlinks resolving outside the selected directory are excluded.
    images = [p for p in images if root in p.parents]
    return sorted(images,key=lambda p:(p.relative_to(root).as_posix().casefold(),p.relative_to(root).as_posix()))


def image_id(image,root):
    relative = image.relative_to(root).as_posix()
    stem = re.sub(r'[^\w.-]+','_',image.stem)[:64]
    return 'image-'+stem+'-'+hashlib.sha256(relative.encode('utf-8')).hexdigest()[:10]


def run_batch(images,input_root,config,output,gpus,retries=1,popen=subprocess.Popen,jobs_per_gpu=1):
    if not isinstance(jobs_per_gpu,int) or isinstance(jobs_per_gpu,bool) or jobs_per_gpu<1:
        raise ValueError('jobs_per_gpu must be a positive integer')
    gpus = list(gpus)
    if not gpus or len(set(gpus))!=len(gpus) or any(not isinstance(g,int) or isinstance(g,bool) or g<0 for g in gpus):
        raise ValueError('gpus must contain unique nonnegative integer indices')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    with RunLogger(output) as logger:
        logger.record('batch_started',config=config,input_root=str(input_root),gpus=gpus,
            jobs_per_gpu=jobs_per_gpu,retries=retries,python=sys.version)
        code = _run_batch(images,input_root,config,output,gpus,retries,popen,jobs_per_gpu,logger)
        logger.record('batch_finished',exit_code=code)
        return code


def _run_batch(images,input_root,config,output,gpus,retries,popen,jobs_per_gpu,logger):
    if not isinstance(jobs_per_gpu,int) or isinstance(jobs_per_gpu,bool) or jobs_per_gpu<1:
        raise ValueError('jobs_per_gpu must be a positive integer')
    gpus = list(gpus)
    if not gpus or len(set(gpus))!=len(gpus) or any(not isinstance(g,int) or isinstance(g,bool) or g<0 for g in gpus):
        raise ValueError('gpus must contain unique nonnegative integer indices')
    # Fill one slot on each GPU before using a second slot on any GPU.
    slots = [(gpu,index) for index in range(jobs_per_gpu) for gpu in gpus]
    output = Path(output).resolve()
    input_root = Path(input_root).resolve()
    atomic_json(output/'config.json',config)
    jobs = []
    for image in images:
        image = Path(image).resolve()
        job = dict(id=image_id(image,input_root),input=str(image),status='pending',attempts=[],last_resume=None)
        try:
            job['plan'] = resolve_plan(config,image)
        except Exception as error:
            job.update(status='failed',error=str(error))
        jobs.append(job)
    pending = deque(job for job in jobs if job['status']=='pending')
    running = {}
    started = time.monotonic()
    last_progress_poll = 0.
    fields = ['input','status','gpu','gpu_slot','attempts','requested_cap','actual_final_N','terminal_N',
              'actual_final_density','primitives_per_megapixel','PSNR','SSIM','MS_SSIM',
              'delta_PSNR_vs_fixed_p2','delta_SSIM_vs_fixed_p2','delta_MS_SSIM_vs_fixed_p2',
              'delta_N_vs_fixed_p2','metric_profile',
              'terminal_step','selected_checkpoint_step','elapsed_seconds','encoding_status','encoded_bytes','encoded_bpp','decoded_RGB8_PSNR','codec_seconds','output','error']

    def persist(phase='training'):
        rows = []
        for job in jobs:
            summary = job.get('summary',{})
            metric = summary.get('metrics',{})
            delta = summary.get('delta_vs_fixed_p2') or {}
            enc = summary.get('encoding',{})
            artifact = (enc.get('artifacts') or [{}])[0]
            rows.append(dict(input=job['input'],status=job['status'],gpu=job.get('gpu'),
                gpu_slot=job.get('gpu_slot'),attempts=len(job['attempts']),requested_cap=job.get('plan',{}).get('requested_cap'),
                actual_final_N=summary.get('actual_final_N'),terminal_N=summary.get('terminal_N'),
                actual_final_density=summary.get('actual_final_density'),
                primitives_per_megapixel=summary.get('primitives_per_megapixel'),
                PSNR=metric.get('PSNR'),SSIM=metric.get('SSIM'),MS_SSIM=metric.get('MS_SSIM'),
                delta_PSNR_vs_fixed_p2=delta.get('PSNR'),delta_SSIM_vs_fixed_p2=delta.get('SSIM'),
                delta_MS_SSIM_vs_fixed_p2=delta.get('MS_SSIM'),delta_N_vs_fixed_p2=delta.get('N'),
                metric_profile=summary.get('metric_profile'),
                terminal_step=summary.get('terminal_step'),selected_checkpoint_step=summary.get('selected_checkpoint_step'),
                elapsed_seconds=summary.get('elapsed_seconds'),encoding_status=enc.get('status'),
                encoded_bytes=artifact.get('bytes'),encoded_bpp=artifact.get('bpp'),
                decoded_RGB8_PSNR=artifact.get('decoded_RGB8_PSNR'),codec_seconds=enc.get('seconds'),
                output=job.get('output'),error=job.get('error')))
        paired = paired_summary(jobs)
        atomic_json(output/'summary.json',dict(schema='adasplat_batch_summary_v1',jobs=rows,paired_vs_fixed_p2=paired,
            complete=not pending and not running,gpus=gpus,jobs_per_gpu=jobs_per_gpu,
            max_parallel_jobs=len(slots),elapsed_seconds=time.monotonic()-started))
        with atomic_output(output/'summary.csv') as temporary:
            with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        active = [dict(input=r['job']['input'],gpu=gpu,gpu_slot=index,pid=r['process'].pid,attempt=r['attempt'],
                       progress=str(r['directory']/'progress.json'),console=str(r['log_path']))
                  for (gpu,index),r in running.items()]
        atomic_json(output/'progress.json',dict(pid=os.getpid(),complete=not pending and not running,
            phase=phase,gpus=gpus,jobs_per_gpu=jobs_per_gpu,max_parallel_jobs=len(slots),
            pending=len(pending),running=active,succeeded=sum(j['status']=='complete' for j in jobs),
            failed=sum(j['status']=='failed' for j in jobs),total_images=len(jobs),
            elapsed_seconds=time.monotonic()-started))
        atomic_json(output/'jobs.json',jobs)
        atomic_json(output/'comparison.json',dict(total_images=len(jobs),paired_vs_fixed_p2=paired))
        if paired is not None:
            print(json.dumps(dict(event='paired_fixed_p2_comparison',**paired),ensure_ascii=False),flush=True)
        logger.record('batch_state',phase=phase,jobs=rows,running=active,pending=len(pending))

    def launch(job,slot):
        gpu,index = slot
        attempt = len(job['attempts'])+1
        folder = output/'images'/job['id']
        folder.mkdir(parents=True,exist_ok=True)
        directory = folder/('attempt_'+str(attempt))
        job_config = copy.deepcopy(config)
        job_config['input'] = job['input']
        job_config['output'] = str(directory)
        config_path = folder/('attempt_'+str(attempt)+'.config.json')
        atomic_json(config_path,job_config)
        log_path = folder/('attempt_'+str(attempt)+'.console.log')
        log = log_path.open('w',encoding='utf-8')
        command = [sys.executable,str(ROOT/'train.py'),'--config',str(config_path),'--gpu',str(gpu)]
        if job['last_resume']:
            command.extend(['--resume',job['last_resume']])
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        attempt_record = dict(attempt=attempt,gpu=gpu,gpu_slot=index,pid=None,output=str(directory),console=str(log_path))
        job['attempts'].append(attempt_record)
        try:
            process = popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),env=environment)
        except Exception as error:
            log.close()
            job.update(error='Worker startup failed: '+str(error),gpu=gpu,gpu_slot=index,output=str(directory))
            attempt_record['startup_error'] = str(error)
            if len(job['attempts'])<=retries:
                job['status'] = 'pending'
                pending.append(job)
            else:
                job['status'] = 'failed'
            return
        except BaseException:
            log.close()
            job['status'] = 'interrupted'
            raise
        job.update(status='running',gpu=gpu,gpu_slot=index,output=str(directory))
        attempt_record['pid'] = process.pid
        running[slot] = dict(job=job,process=process,log=log,log_path=log_path,directory=directory,
                            attempt=attempt,last_reported_step=None)
        logger.record('worker_launched',input=job['input'],gpu=gpu,gpu_slot=index,pid_child=process.pid,attempt=attempt,command=command,console=str(log_path))
        print(json.dumps(dict(event='launched',input=job['input'],gpu=gpu,gpu_slot=index,pid=process.pid,attempt=attempt),ensure_ascii=False),flush=True)

    persist()
    try:
        while pending or running:
            changed = False
            for slot in slots:
                if slot not in running and pending:
                    launch(pending.popleft(),slot)
                    changed = True
            if time.monotonic()-last_progress_poll>=5:
                last_progress_poll = time.monotonic()
                for (gpu,index),active in running.items():
                    try:
                        progress = json.loads((active['directory']/'progress.json').read_text(encoding='utf-8'))
                    except (OSError,ValueError):
                        continue
                    row = progress.get('last_evaluation') or {}
                    if row.get('delta_vs_fixed_p2') is not None and row.get('step')!=active['last_reported_step']:
                        active['last_reported_step'] = row['step']
                        print('[GPU {0} slot {1}] '.format(gpu,index)+format_comparison(active['job']['input'],
                            row['step'],row['N'],row['metrics'],row['delta_vs_fixed_p2']),flush=True)
            for slot in list(running):
                gpu,index = slot
                active = running[slot]
                code = active['process'].poll()
                if code is None:
                    continue
                active['log'].close()
                job = active['job']
                directory = active['directory']
                job['attempts'][-1]['exit_code'] = code
                summary = None
                if code==0 and (directory/'summary.json').is_file():
                    try:
                        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
                    except (ValueError,OSError):
                        pass
                if summary is not None and summary.get('complete'):
                    job.update(status='complete',summary=summary)
                    job.pop('error',None)
                    if summary.get('delta_vs_fixed_p2') is not None:
                        print('[FINAL] '+format_comparison(job['input'],summary['selected_checkpoint_step'],
                            summary['actual_final_N'],summary['metrics'],summary['delta_vs_fixed_p2']),flush=True)
                else:
                    failure = directory/'failure.json'
                    reason = 'Child failed or did not finish, exit code '+str(code)
                    if failure.is_file():
                        try:
                            reason = json.loads(failure.read_text(encoding='utf-8')).get('error',reason)
                        except (ValueError,OSError):
                            pass
                    checkpoint = directory/'resume.pt'
                    job['last_resume'] = str(checkpoint) if checkpoint.is_file() else None
                    job['error'] = reason
                    encoding_failed = (directory/'encoding.json').is_file() and json.loads((directory/'encoding.json').read_text(encoding='utf-8')).get('status')=='failed'
                    if encoding_failed:
                        job['error'] = json.loads((directory/'encoding.json').read_text(encoding='utf-8')).get('error','encoding failed')
                    if len(job['attempts'])<=retries and not encoding_failed:
                        job['status'] = 'pending'
                        pending.append(job)
                    else:
                        job['status'] = 'failed'
                logger.record('worker_finished',input=job['input'],gpu=gpu,gpu_slot=index,status=job['status'],exit_code=code,attempt=active['attempt'],error=job.get('error'))
                print(json.dumps(dict(event='finished',input=job['input'],gpu=gpu,gpu_slot=index,status=job['status'],exit_code=code),ensure_ascii=False),flush=True)
                del running[slot]
                changed = True
            if changed:
                persist()
            if pending or running:
                time.sleep(.2)
        persist('completed')
        return 1 if any(job['status']=='failed' for job in jobs) else 0
    except BaseException:
        for active in running.values():
            process = active['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            active['log'].close()
            active['job']['status'] = 'interrupted'
        running.clear()
        for job in pending:
            job['status'] = 'interrupted'
        pending.clear()
        persist('interrupted')
        raise
