"""One image trainer used by both public entry points.

The Lp representation, native renderer, CG and cohort optimizer are reused from
the accepted release. This module owns generic input, schedules and run state.
"""
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from .config import ROOT, recipe_fingerprint, resolve_plan, validate
from .control import FixedStageEarlyStop, RelativeBirthBudget
from .milestones import PrimitiveSnapshots
from .run_logging import RunLogger
from .comparison import comparison, format_comparison
from adasplat.paths import bootstrap
bootstrap()
sys.path.insert(0,str(ROOT/'tools'))

import numpy as np
import torch
from PIL import Image
from atomic_files import atomic_json, atomic_output
from adasplat_vnext_runtime import select_renderer_binary, file_sha256
from adasplat_rebuild_model import RebuildModel, RebuildOptions
from adasplat_rebuild_optimizer import CohortAdan, SensitivityTrust
from adasplat_rebuild_growth import propose_births
from adasplat_fixed_capacity import initialize_fixed_capacity, fit_linear_colors
from adasplat_full_image_objective import full_image_loss_gradient
from adasplat_vnext_objective import control_metrics
from train_rebuild import fit_new_colors, quality_gate
from train_ssim_rebuild import image_save, diagnostics


RESUME_SCHEMA = 'adasplat_trainer_resume_v1'
METRIC_CONFIG = {
    'evaluation': {'core_size':128,'tail_area':.1,'mse_weight':.8,'tail_weight':.2,
                   'ms_window':11,'ms_sigma':1.5,'ms_K':[.01,.03],
                   'ms_weights':[.0448,.2856,.3001,.2363,.1333]},
    'loss': {'ssim_weight':0.,'ssim_window':11,'ssim_gaussian':True,
             'gradient_weight':0.,'scale':32.}
}


def torch_save(value,path):
    with atomic_output(path) as temporary:
        torch.save(value,temporary)


@torch.no_grad()
def dark_metrics(image,target,threshold=.05,margin=.04,grid=4):
    """Whole image and relative grid cells; no image-name-specific coordinates."""
    h,w = target.shape[-2:]
    regions = [('whole',0,h,0,w)]
    for iy in range(grid):
        for ix in range(grid):
            regions.append(('cell_{0}_{1}'.format(iy,ix),iy*h//grid,(iy+1)*h//grid,
                            ix*w//grid,(ix+1)*w//grid))
    result = {}
    for name,y0,y1,x0,x1 in regions:
        pred = image[...,y0:y1,x0:x1]
        ref = target[...,y0:y1,x0:x1]
        mask = ref.mean(1)<threshold
        pixels = int(mask.sum())
        if not pixels:
            continue
        excess = pred.mean(1)-ref.mean(1)
        result[name] = dict(dark_pixels=pixels,dark_mse=float((pred-ref).square().mean(1)[mask].mean()),
                            bright_spot_pixels=int((mask&(excess>margin)).sum()),
                            bright_leak_energy=float(torch.relu(excess-margin).square()[mask].mean()))
    return result


class Trainer:
    def __init__(self,config,output,resume=None):
        self.cfg = copy.deepcopy(validate(config))
        self.plan = resolve_plan(self.cfg)
        self.out = Path(output).expanduser().resolve()
        self.resume_path = Path(resume).expanduser().resolve(strict=True) if resume else None
        self.adv = self.cfg['advanced']
        self.options = RebuildOptions(cap=self.plan['requested_cap'],**self.adv['options']).validate()
        self.fingerprint = recipe_fingerprint(self.cfg,self.plan)
        self.binary = ROOT/('native/csrc.pyd' if os.name == 'nt' else 'native/csrc.so')
        self.batch = RelativeBirthBudget(self.plan)
        self.early = FixedStageEarlyStop(self.cfg['early_stop'],self.plan['fixed_start_step'])
        self.snapshots = PrimitiveSnapshots(**self.cfg['snapshots'])
        self.best_payload = None
        self.prior = None
        self.budget = None
        self.elapsed_offset = 0.
        self.start_step = 0
        self.trust = None
        self.run_started = None
        self.state = None
        self.fixed_reference = None

    def elapsed(self):
        return self.elapsed_offset+time.monotonic()-self.run_started

    def _initialize(self):
        torch.set_num_threads(self.adv['threads'])
        if not torch.cuda.is_available():
            raise RuntimeError('A compatible CUDA GPU is required for training; --dry-run needs no GPU')
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(self.cfg['seed'])
        select_renderer_binary(self.binary,self.adv['renderer_sha256'])
        path = Path(self.plan['input'])
        with Image.open(path) as image:
            array = np.asarray(image.convert('RGB')).copy()
        if self.cfg['preprocess']['smoke_crop']:
            array = array[:self.plan['height'],:self.plan['width']].copy()
        self.target = torch.from_numpy(array).permute(2,0,1)[None].cuda().float()/255
        self.dark_mask = self.target.mean(1,keepdim=True)<self.adv['dark_threshold']
        self.target_hash = file_sha256(path)
        baseline_path = ROOT/'configs/fixed_p2_100_baseline.json'
        if baseline_path.is_file() and not self.cfg['preprocess']['smoke_crop']:
            references = json.loads(baseline_path.read_text(encoding='utf-8'))['references']
            self.fixed_reference = references.get(self.target_hash)
            if self.fixed_reference is None:
                self.log.record('fixed_p2_baseline_unavailable',input=str(path),target_sha256=self.target_hash)
        core = ROOT/'src/adasplat/core'
        dependencies = list(core.rglob('*.py'))+list(Path(__file__).parent.glob('*.py'))
        dependencies += [ROOT/'tools/train_rebuild.py',ROOT/'tools/train_ssim_rebuild.py',ROOT/'tools/atomic_files.py']
        if baseline_path.is_file():
            dependencies.append(baseline_path)
        self.implementation = {p.relative_to(ROOT).as_posix():file_sha256(p) for p in dependencies}
        saved = torch.load(self.resume_path,map_location='cpu',weights_only=False) if self.resume_path else None
        if saved is not None:
            if saved.get('schema') != RESUME_SCHEMA:
                raise ValueError('This API resumes its own resume.pt; use the preserved v1.0 trainer for legacy resume files')
            if saved['recipe_fingerprint'] != self.fingerprint:
                raise ValueError('Resume recipe mismatch, including total_steps and stage schedule')
            if saved['model']['target_sha256'] != self.target_hash or saved['model']['renderer']['sha256'] != self.adv['renderer_sha256']:
                raise ValueError('Resume input/renderer hash mismatch')
            if saved['implementation_hashes'] != self.implementation:
                raise ValueError('Resume training implementation changed')
        count = saved['model']['N'] if saved else self.plan['initial_N']
        self.model = RebuildModel(self.target,count,self.options)
        opt = self.adv['optimizer']
        self.optimizer = CohortAdan(self.model,rates=opt['rates'],betas=tuple(opt['betas']),eps=opt['eps'])
        if saved:
            self.model.load_state_dict(saved['model']['model'])
            self.optimizer.load_state_dict(saved['optimizer'])
            self.state = copy.deepcopy(saved['state'])
            self.start_step = saved['step']
            self.elapsed_offset = saved['state']['elapsed_seconds']
            self.batch.load_state_dict(saved['batch'])
            self.early.load_state_dict(saved['early_stop'])
            self.prior = saved['prior'].cuda() if saved['prior'] is not None else None
            self.budget = saved['mse_budget']
            self.best_payload = saved['best_model']
            self.snapshots.load_state_dict(saved['primitive_snapshots'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state(saved['cuda_rng'])
            self.state.update(complete=False,resumed_from=str(self.resume_path),output=str(self.out))
        else:
            initialization = initialize_fixed_capacity(self.model,self.target,seed=self.cfg['seed'])
            self.state = dict(schema='adasplat_training_result_v1',complete=False,output=str(self.out),
                              config=self.cfg,resolved_plan=self.plan,recipe_fingerprint=self.fingerprint,
                              implementation_hashes=self.implementation,initialization=initialization,
                              source_checkpoint=None,initial_N=count,requested_cap=self.plan['requested_cap'],
                              target_sha256=self.target_hash,renderer_sha256=self.adv['renderer_sha256'],
                              trajectory=[],trials=[],optimizer_updates=0,gradient_evaluations=0,
                              proposal_seconds=0.,trial_seconds=0.,native_forward_calls=0,native_backward_calls=0)
        self.state.update(pid=os.getpid(),gpu=torch.cuda.get_device_name(0),
                          fixed_p2_reference=self.fixed_reference,metric_profile='legacy_rgb32_gaussian11_ms5_v1',
                          cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        RebuildModel.forward_passes = self.state.get('native_forward_calls',0)
        RebuildModel.backward_passes = self.state.get('native_backward_calls',0)
        if self.options.sensitivity_trust:
            self.trust = SensitivityTrust(self.model)
            self.trust.refresh()

    def _observe_snapshot(self,step,metrics,payload=None):
        if self.snapshots.counts:
            payload = payload if payload is not None else self._payload(step)
            if self.snapshots.observe(payload,metrics,self.adv['ssim_weight']):
                self.snapshots.flush(self.out,torch_save,atomic_json)
            self.state['primitive_snapshots'] = self.snapshots.summary()
            self.log.record('primitive_snapshots',step=step,summary=self.snapshots.summary())

    def progress(self,phase,step,**extra):
        self.log.record('progress',phase=phase,step=step,N=self.model.count,elapsed_seconds=self.elapsed(),**extra)
        latest = self.state['trajectory'][-1] if self.state and self.state['trajectory'] else None
        atomic_json(self.out/'progress.json',dict(pid=os.getpid(),complete=False,phase=phase,
            stage='growth' if step<=self.plan['growth_end_step'] and self.cfg['growth']['enabled'] else 'fixed',
            step=step,total_steps=self.plan['total_steps'],N=self.model.count,cap=self.plan['requested_cap'],
            elapsed_seconds=self.elapsed(),optimizer_updates=self.state['optimizer_updates'],
            primitive_snapshots=self.snapshots.summary(),
            last_evaluation=latest,remaining_capacity=self.plan['requested_cap']-self.model.count,**extra))

    def _metrics(self,model,optimizer):
        with torch.no_grad():
            image = model()
            values = control_metrics(image,self.target,METRIC_CONFIG)
            dark = dark_metrics(image,self.target,self.adv['dark_threshold'],self.adv['dark_margin'],self.adv['dark_grid'])
            stats = diagnostics(model)
            stats.pop('opacity_gt1',None)
            minor = model.scales.amin(1)
            ratio = model.scales.amax(1)/minor
            peak = model.colors[:,:3]/model.norm()[:,None]
            stats.update(unit_peak_alpha=True,peak_rgb_l2_p99=float(torch.quantile(peak.norm(dim=1),.99)),
                         thin_high_peak_count=int(((minor<.5)&(ratio>8)&(peak.mean(1)>.8)).sum()))
            stats['cohorts'] = [dict(birth_step=int(birth),N=int((optimizer.birth_step==birth).sum()),
                              thin_count=int(((optimizer.birth_step==birth)&(minor<.5)&(ratio>8)).sum()))
                              for birth in optimizer.birth_step.unique()]
            row = dict(N=model.count,metrics=values,dark_regions=dark,statistics=stats,
                       trainable_parameters=sum(p.numel() for p in model.parameters()))
        return row,image

    def _payload(self,step):
        return dict(**self.model.metadata(),model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                    renderer=dict(binary=self.binary.relative_to(ROOT).as_posix(),sha256=self.adv['renderer_sha256'],cutoff=1/255,support_mode='kernel',tail='C1'),
                    target_sha256=self.target_hash,step=step,
                    cohort_birth_step=self.optimizer.birth_step.detach().cpu())

    def _save_resume(self,step,payload):
        torch_save(dict(schema=RESUME_SCHEMA,model=payload,step=step,state=self.state,
                        optimizer=self.optimizer.state_dict(),batch=self.batch.state_dict(),
                        early_stop=self.early.state_dict(),mse_budget=self.budget,
                        prior=self.prior,best_model=self.best_payload,
                        primitive_snapshots=self.snapshots.state_dict(),
                        recipe_fingerprint=self.fingerprint,implementation_hashes=self.implementation,
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),self.out/'resume.pt')

    def _evaluate(self,step,check_stop=True):
        row,image = self._metrics(self.model,self.optimizer)
        row.update(step=step,optimizer_updates=self.state['optimizer_updates'])
        row['delta_vs_fixed_p2'] = comparison(row['metrics'],row['N'],row['trainable_parameters'],self.fixed_reference)
        self.log.record('evaluation',elapsed_seconds=self.elapsed(),**row)
        if self.state['trajectory'] and self.state['trajectory'][-1]['step']==step:
            self.state['trajectory'][-1] = row
        else:
            self.state['trajectory'].append(row)
        self.budget = min(row['metrics']['MSE'],self.budget or float('inf'))*10**(.005/10)
        # Relaxed growth protection compares with the current fit; no initial anchor.
        payload = self._payload(step)
        self._observe_snapshot(step,row['metrics'],payload)
        torch_save(payload,self.out/'latest.pt')
        image_save(image,self.out/'latest.png')
        best = self.state.get('best')
        value = row['metrics']['MSE']+self.adv['ssim_weight']*(1-row['metrics']['SSIM'])
        reset = best is not None and self.adv['checkpoint_selection']=='same_count' and best['N']!=row['N']
        if best is None or reset or value < best['metrics']['MSE']+self.adv['ssim_weight']*(1-best['metrics']['SSIM']):
            self.state['best'] = copy.deepcopy(row)
            self.best_payload = payload
            torch_save(payload,self.out/'best.pt')
            image_save(image,self.out/'best.png')
        should_stop = self.early.observe(step,row['metrics']) if check_stop else False
        self.state.update(elapsed_seconds=self.elapsed(),native_forward_calls=RebuildModel.forward_passes,
                          native_backward_calls=RebuildModel.backward_passes,
                          cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                          early_stop_state=self.early.state_dict())
        self._save_resume(step,payload)
        atomic_json(self.out/'result.json',self.state)
        if row['delta_vs_fixed_p2'] is not None:
            print(format_comparison(self.plan['input'],step,row['N'],row['metrics'],row['delta_vs_fixed_p2']),flush=True)
        print(json.dumps(dict(event='evaluation',step=step,N=row['N'],PSNR=row['metrics']['PSNR'],
                              SSIM=row['metrics']['SSIM'],MS_SSIM=row['metrics']['MS_SSIM'],
                              delta_vs_fixed_p2=row['delta_vs_fixed_p2'],
                              stage='fixed' if step>=self.plan['fixed_start_step'] else 'growth',
                              early_stop_bad_evaluations=self.early.bad_evaluations)),flush=True)
        return should_stop

    def active_weight(self,step):
        return self.adv['ssim_weight']*min(1.,max(0.,(step-self.plan['mse_steps'])/max(1,self.plan['ssim_ramp_steps'])))

    def _update(self,model,optimizer,step,rows=None,geometry=True,trial_weight=None,local_step=0,tracker=None):
        optimizer.zero_grad()
        raw = model.render_unclipped()
        image = raw.clamp(0,1)
        regularizer = self.adv['regularizer_weight']*(torch.relu(raw-1).square().mean()+torch.relu(-raw).square().mean())
        regularizer = regularizer+self.adv['dark_leak_weight']*(torch.relu(raw-self.target-self.adv['dark_margin']).square()*self.dark_mask).mean()
        interval = 4 if trial_weight is not None else self.adv['ssim_interval']
        weight = trial_weight if trial_weight is not None else self.active_weight(step)
        tick = local_step if trial_weight is not None else step
        if weight and tick%interval==0:
            loss,gradient,components = full_image_loss_gradient(image,self.target,ssim_weight=weight*interval,
                mse_budget=self.budget,budget_weight=20.,tile=self.adv['ssim_tile'])
            penalty = regularizer*32
            torch.autograd.backward([image,penalty],[gradient,None])
        else:
            mse = (image-self.target).square().mean()
            loss = (mse+regularizer)*32
            loss.backward()
            components = dict(mse=float(mse.detach()))
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('Nonfinite training loss or gradient')
        before = tracker.snapshot() if tracker is not None and geometry else None
        factor = .15+.85*.5*(1+math.cos(math.pi*min(step,self.plan['total_steps'])/self.plan['total_steps']))
        optimizer.step(factor=factor,rows=rows,geometry=geometry)
        if before is not None:
            self.state['last_trust'] = tracker.constrain(before)
        model.project_parameters_()
        self.state['optimizer_updates'] += 1
        self.state['gradient_evaluations'] += 1
        self.log.record('optimizer_update',step=step,local_step=local_step,N=model.count,
            scope='main' if trial_weight is None else ('control' if model is self.model else 'candidate'),
            geometry=geometry,optimizer_updates=self.state['optimizer_updates'],
            elapsed_seconds=self.elapsed(),scaled_loss=float(loss.detach())+(float(regularizer.detach())*32 if weight and tick%interval==0 else 0),
            regularizer=float(regularizer.detach()),ssim_weight=weight,
            ssim_gradient_applied=bool(weight and tick%interval==0),components=components)
        return components

    def _cycle(self,step):
        count = self.batch.request(step,self.model.count)
        if count<=0:
            return
        trial_start = time.monotonic()
        self.progress('candidate_selection',step,requested_births=count)
        weight = self.active_weight(step)
        before,image = self._metrics(self.model,self.optimizer)
        self._observe_snapshot(step,before['metrics'])
        proposal,info,self.prior = propose_births(self.model,image,self.target,count,
            ssim_weight=weight,seed=self.cfg['seed']+step,prior=self.prior)
        del image
        self.state['proposal_seconds'] += time.monotonic()-trial_start
        if proposal is None:
            self.batch.update(step,False,0)
            self.state['trials'].append(dict(step=step,accepted=False,added=0,proposal=info,
                reason='no positive eligible proposal',before=before))
            self.log.record('birth_rejected',step=step,reason='no positive eligible proposal',proposal=info,before=before)
            self.state['trial_seconds'] += time.monotonic()-trial_start
            atomic_json(self.out/'result.json',self.state)
            return
        candidate,mapping,rows = proposal
        candidate_opt = self.optimizer.transport(candidate,mapping,birth_step=step)
        self.progress('new_colors_joint_solve',step,new_rows=len(rows))
        cg = fit_new_colors(candidate,self.target,rows,steps=self.adv['new_color_cg'])
        previous = None
        patience = 0
        history = []
        warm_done = 0
        for i in range(self.adv['warm_max']):
            self._update(candidate,candidate_opt,step,rows=rows,geometry=False,trial_weight=weight,local_step=i)
            warm_done = i+1
            if warm_done%16==0 or warm_done==self.adv['warm_max']:
                row,image = self._metrics(candidate,candidate_opt)
                value = row['metrics']['MSE']+weight*(1-row['metrics']['SSIM'])
                history.append(dict(updates=warm_done,metrics=row['metrics'],objective=value))
                del image
                patience = patience+1 if previous is not None and previous-value<max(1e-8,abs(previous)*1e-4) else 0
                previous = value
                self.progress('new_cohort_color_maturity',step,warm_updates=warm_done,new_rows=len(rows))
                if warm_done>=self.adv['warm_min'] and (not self.options.adaptive_maturity or patience>=2):
                    break
        candidate_trust = None
        if self.options.sensitivity_trust:
            candidate_trust = SensitivityTrust(candidate)
            candidate_trust.refresh()
        for i in range(self.adv['joint_steps']):
            self._update(candidate,candidate_opt,step,trial_weight=weight,local_step=i,tracker=candidate_trust)
            if (i+1)%16==0:
                self.progress('candidate_joint_fit',step,joint_updates=i+1)
        after,image = self._metrics(candidate,candidate_opt)
        del image
        matched = warm_done+self.adv['joint_steps']+cg['updates']+2 if self.adv['growth_reference']=='continued_fit' else 0
        for i in range(matched):
            self._update(self.model,self.optimizer,step,trial_weight=weight,local_step=i,tracker=self.trust)
            if (i+1)%16==0:
                self.progress('continued_fit_control',step,control_updates=i+1,control_total=matched)
        if matched:
            control,image = self._metrics(self.model,self.optimizer)
            self._observe_snapshot(step,control['metrics'])
            del image
        else:
            control = before
        added = candidate.count-self.model.count
        accepted,decision = quality_gate(control['metrics'],after['metrics'],control['dark_regions'],
            after['dark_regions'],added,weight,dark_policy='relaxed')
        record = dict(step=step,accepted=accepted,added=added,N_before=self.model.count,N_candidate=candidate.count,
                      proposal=info,before=before,control=control,candidate=after,decision=decision,
                      ssim_weight=weight,new_color_cg=cg,warm_history=history,warm_updates=warm_done,
                      joint_updates=self.adv['joint_steps'],control_gradient_updates=matched,
                      comparison_reference=self.adv['growth_reference'],old_moments_preserved=True,
                      new_clocks_start_at_zero=True,elapsed_seconds=time.monotonic()-trial_start,
                      new_geometry_updates_min=int(candidate_opt.state['xys']['step'][rows].min()))
        self.batch.update(step,accepted,added)
        if accepted:
            self.model,self.optimizer,self.trust = candidate,candidate_opt,candidate_trust
            self._observe_snapshot(step,after['metrics'])
        self.state['trials'].append(record)
        self.log.record('birth_decision',**record)
        self.state['trial_seconds'] += time.monotonic()-trial_start
        self.state['batch_controller'] = self.batch.state_dict()
        atomic_json(self.out/'result.json',self.state)
        torch.cuda.empty_cache()

    def _finalize(self,step,reason,complete):
        self.snapshots.flush(self.out,torch_save,atomic_json,force=True)
        self.state['primitive_snapshots'] = self.snapshots.summary()
        best = self.state['best']
        # Preserve the selected checkpoint if resuming into a fresh output directory.
        torch_save(self.best_payload,self.out/'best.pt')
        torch_save(self.best_payload,self.out/'final.pt')
        if not (self.out/'best.png').is_file():
            selected = RebuildModel(self.target,best['N'],self.options)
            selected.load_state_dict(self.best_payload['model'])
            with torch.no_grad():
                image_save(selected(),self.out/'best.png')
            del selected
        shutil.copy2(self.out/'best.png',self.out/'final.png')
        self.state.update(complete=complete,termination=reason,terminal_step=step,terminal_N=self.model.count,
                          actual_final_N=best['N'],actual_final_density=best['N']/self.plan['pixels'],
                          actual_final_primitives_per_megapixel=best['N']/self.plan['pixels']*1e6,
                          selected_checkpoint_step=best['step'],selected_metrics=best['metrics'],
                          elapsed_seconds=self.elapsed(),native_forward_calls=RebuildModel.forward_passes,
                          native_backward_calls=RebuildModel.backward_passes,capacity_is_upper_bound=True,
                          resume_path='resume.pt',final_checkpoint='final.pt',final_image='final.png')
        atomic_json(self.out/'result.json',self.state)
        atomic_json(self.out/'summary.json',dict(schema='adasplat_training_summary_v1',complete=complete,
            input=self.plan['input'],width=self.plan['width'],height=self.plan['height'],
            capacity_mode=self.plan['capacity_mode'],requested_cap=self.plan['requested_cap'],
            requested_density=self.plan['requested_density'],actual_final_N=best['N'],terminal_N=self.model.count,
            actual_final_density=self.state['actual_final_density'],
            primitives_per_megapixel=self.state['actual_final_primitives_per_megapixel'],
            terminal_step=step,selected_checkpoint_step=best['step'],termination=reason,
            metrics=best['metrics'],metric_domain='full-resolution float tensor, not quantized PNG',
            fixed_p2_reference=self.fixed_reference,metric_profile='legacy_rgb32_gaussian11_ms5_v1',
            delta_vs_fixed_p2=comparison(best['metrics'],best['N'],best['trainable_parameters'],self.fixed_reference),
            source_checkpoint=str(self.resume_path) if self.resume_path else None,
            elapsed_seconds=self.state['elapsed_seconds'],optimizer_updates=self.state['optimizer_updates'],
            primitive_snapshots=self.snapshots.summary(),
            final_checkpoint='final.pt',final_image='final.png'))
        atomic_json(self.out/'progress.json',dict(pid=os.getpid(),complete=complete,phase=reason,stage='fixed' if step>self.plan['growth_end_step'] else 'growth',
            step=step,total_steps=self.plan['total_steps'],N=self.model.count,actual_final_N=best['N'],
            cap=self.plan['requested_cap'],elapsed_seconds=self.state['elapsed_seconds'],
            primitive_snapshots=self.snapshots.summary(),
            optimizer_updates=self.state['optimizer_updates'],last_evaluation=self.state['trajectory'][-1]))
        return self.state

    def fit(self,until_step=None):
        if until_step is not None and (not isinstance(until_step,int) or until_step<1 or until_step>self.plan['total_steps']):
            raise ValueError('Invalid session step limit')
        self.out.mkdir(parents=True,exist_ok=False)
        with RunLogger(self.out) as self.log:
            self.log.record('run_started',config=self.cfg,resolved_plan=self.plan,
                resume=str(self.resume_path) if self.resume_path else None,
                python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
                console_log='console.log',event_log='events.jsonl')
            result = self._fit(until_step)
            self.log.record('run_finished',termination=result['termination'],complete=result['complete'],
                terminal_step=result['terminal_step'],terminal_N=result['terminal_N'],
                selected_metrics=result['selected_metrics'],primitive_snapshots=self.snapshots.summary())
            return result

    def _fit(self,until_step=None):
        """Train once. until_step supports bounded sessions/tests, preserving total schedule."""
        if until_step is not None and (not isinstance(until_step,int) or until_step<1 or until_step>self.plan['total_steps']):
            raise ValueError('Invalid session step limit')
        self.run_started = time.monotonic()
        atomic_json(self.out/'config.json',self.cfg)
        atomic_json(self.out/'resolved_plan.json',self.plan)
        try:
            self._initialize()
            if self.resume_path is None:
                self.progress('initial_color_solve',0)
                self.state['initial_color_solve'] = fit_linear_colors(self.model,self.target,steps=self.adv['initial_cg'],
                    progress=lambda done,residual:self.progress('initial_color_solve',0,cg_updates=done,cg_relative_residual=residual))
                self._evaluate(0)
            else:
                if self.start_step>self.plan['total_steps']:
                    raise ValueError('Resume step exceeds the total schedule')
                # Restore best files without duplicating the boundary evaluation or budget update.
                self.snapshots.flush(self.out,torch_save,atomic_json,force=True)
                torch_save(self.best_payload,self.out/'best.pt')
                # A completed or same-step resume still needs a valid recovery file
                # in its new output directory, even if no evaluation follows.
                self._save_resume(self.start_step,self._payload(self.start_step))
                self.progress('restored_checkpoint',self.start_step)
            stop = until_step or self.plan['total_steps']
            if stop<self.start_step:
                raise ValueError('Session limit precedes resume step')
            reason = 'session_limit' if stop<self.plan['total_steps'] else 'completed'
            last_step = self.start_step
            for step in range(self.start_step+1,stop+1):
                if self.trust is not None and step%128==0:
                    self.trust.refresh()
                if step in self.plan['growth_steps']:
                    self._cycle(step)
                components = self._update(self.model,self.optimizer,step,tracker=self.trust)
                last_step = step
                if step%25==0 or self.cfg['preprocess']['smoke_crop']:
                    self.progress('fit',step,objective_components=components)
                es = self.cfg['early_stop']
                fixed_tick = es['enabled'] and step>self.plan['fixed_start_step'] and (step-self.plan['fixed_start_step'])%es['eval_interval']==0
                should_evaluate = step%self.adv['evaluate_every']==0 or step in (self.plan['fixed_start_step'],stop) or fixed_tick
                if should_evaluate and self._evaluate(step):
                    reason = 'early_stopped'
                    break
            if not self.state['trajectory'] or self.state['trajectory'][-1]['step']!=last_step:
                self._evaluate(last_step,check_stop=False)
            return self._finalize(last_step,reason,reason!='session_limit')
        except BaseException as error:
            elapsed = self.elapsed()
            atomic_json(self.out/'failure.json',dict(pid=os.getpid(),error=str(error),
                        exception=type(error).__name__,traceback=traceback.format_exc(),elapsed_seconds=elapsed))
            atomic_json(self.out/'progress.json',dict(pid=os.getpid(),complete=False,phase='failed',
                        error=str(error),elapsed_seconds=elapsed,step=self.start_step if self.state is None else self.state['trajectory'][-1]['step'] if self.state['trajectory'] else 0))
            raise
