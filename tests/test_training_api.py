"""Budget/stage contracts and scheduler behavior independent of CUDA."""
import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from adasplat.training.config import DEFAULTS,HARD_CAP,load_config,resolve_plan,recipe_fingerprint
from adasplat.training.control import FixedStageEarlyStop,RelativeBirthBudget
from adasplat.training.batch import enumerate_images,image_id,run_batch
from adasplat.training.cli import single_main


def image(path,size=(400,200)):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',size,(80,100,120)).save(path)
    return path


def test_density_scales_with_actual_pixels_and_count_is_resolution_independent(tmp_path):
    a=image(tmp_path/'a.png');b=image(tmp_path/'b.png',(800,400))
    cfg=load_config(overrides={'capacity':{'mode':'density','density':.005}})
    assert resolve_plan(cfg,a)['requested_cap']==400
    assert resolve_plan(cfg,b)['requested_cap']==1600
    cfg=load_config(overrides={'capacity':{'mode':'count','count':400}})
    assert resolve_plan(cfg,a)['requested_cap']==resolve_plan(cfg,b)['requested_cap']==400


def test_default_budget_recovers_existing_cap_and_initial_count(tmp_path):
    p=image(tmp_path/'reference.png',(4928,3264))
    plan=resolve_plan(load_config(),p)
    assert plan['requested_cap']==114567 and plan['initial_N']==32768
    assert plan['growth_end_step']==32000 and len(plan['growth_steps'])==16
    assert plan['mse_steps']==4000 and plan['ssim_ramp_steps']==4000


def test_schema_rejects_unknown_keys_conflicting_modes_and_nonfinite_numbers():
    with pytest.raises(ValueError,match='Unknown'):
        load_config(overrides={'num':400})
    with pytest.raises(ValueError,match='mutually exclusive'):
        load_config(overrides={'capacity':{'mode':'density','density':.01,'count':400}})
    with pytest.raises(ValueError):load_config(overrides={'capacity':{'density':float('nan')}})
    with pytest.raises(ValueError):load_config(overrides={'total_steps':True})
    with pytest.raises(ValueError):load_config(overrides={'advanced':{'options':{'pixel_samples':2.0}}})
    with pytest.raises(ValueError):load_config(overrides={'advanced':{'options':{'split_merge':True}}})


def test_short_inputs_and_over_budget_requests_fail_before_cuda(tmp_path):
    p=image(tmp_path/'tiny.png',(160,200))
    with pytest.raises(ValueError,match='160'):resolve_plan(load_config(),p)
    p=image(tmp_path/'large_request.png')
    cfg=load_config(overrides={'capacity':{'density':10}})
    with pytest.raises(ValueError,match=str(HARD_CAP)):resolve_plan(cfg,p)


def test_config_paths_are_relative_to_config_file_and_cli_paths_override(tmp_path):
    p=image(tmp_path/'data/a.png')
    f=tmp_path/'recipe.json'
    f.write_text(json.dumps({'input':'data/a.png','capacity':{'mode':'count','count':400}}))
    cfg=load_config(f)
    assert cfg['input']==str(p.resolve()) and cfg['capacity']['density'] is None
    assert resolve_plan(cfg)['requested_cap']==400


def test_total_steps_changes_stage_lengths_without_increasing_opportunity_count(tmp_path):
    p=image(tmp_path/'a.png')
    for total in [20000,40000,60000]:
        cfg=load_config(overrides={'total_steps':total})
        plan=resolve_plan(cfg,p)
        assert plan['growth_end_step']==round(.8*total) and len(plan['growth_steps'])==16
        assert all(plan['mse_steps']<s<=plan['growth_end_step'] for s in plan['growth_steps'])


def test_cap_is_upper_bound_and_rejections_do_not_force_new_primitives(tmp_path):
    cfg=load_config(smoke=True)
    plan=resolve_plan(cfg,image(tmp_path/'a.png'))
    batch=RelativeBirthBudget(plan)
    count=plan['initial_N']
    for step in plan['growth_steps']:
        k=batch.request(step,count)
        assert 0<k<=plan['requested_cap']-count
        batch.update(step,False,k)
    assert count<plan['requested_cap']
    assert batch.request(plan['growth_end_step']+1,count)==0
    assert batch.request(plan['growth_steps'][-1],plan['requested_cap']-1)==1


def early_config(**kwargs):
    cfg=copy.deepcopy(DEFAULTS['early_stop'])
    cfg.update(min_fixed_steps=20,patience=3,**kwargs)
    return cfg


def test_no_early_stop_in_growth_or_before_minimum_fixed_duration():
    stop=FixedStageEarlyStop(early_config(),100)
    m=dict(PSNR=30.,SSIM=.8)
    for step in range(1,120):assert not stop.observe(step,m)
    assert not stop.observe(120,m)
    assert not stop.observe(121,m)
    assert stop.observe(122,m)


def test_ssim_improvement_keeps_psnr_plateau_training_alive():
    stop=FixedStageEarlyStop(early_config(),0)
    for step in range(0,200,20):
        assert not stop.observe(step,dict(PSNR=30.,SSIM=.8+step*.0001))


def test_small_cumulative_gains_and_resume_preserve_early_stop_history():
    stop=FixedStageEarlyStop(early_config(),0)
    for step,p in [(0,30.),(20,30.004),(21,30.008),(22,30.012)]:
        assert not stop.observe(step,dict(PSNR=p,SSIM=.8))
    assert stop.bad_evaluations==0
    restored=FixedStageEarlyStop(early_config(),0)
    restored.load_state_dict(stop.state_dict())
    for step in [23,24]:assert not restored.observe(step,dict(PSNR=30.012,SSIM=.8))
    assert restored.observe(25,dict(PSNR=30.012,SSIM=.8))


def test_dry_run_does_not_create_output_and_cli_modes_are_exclusive(tmp_path,capsys):
    p=image(tmp_path/'a.png');output=tmp_path/'output'
    assert single_main(['--input',str(p),'--num','400','--output',str(output),'--dry-run'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['resolved_plan']['requested_cap']==400 and not output.exists()
    with pytest.raises(SystemExit) as error:single_main(['--num','400','--density','.01'])
    assert error.value.code==2


def test_recipe_fingerprint_excludes_relocation_but_keeps_total_schedule(tmp_path):
    cfg=load_config(overrides={'input':str(image(tmp_path/'a.png'))})
    plan=resolve_plan(cfg);a=recipe_fingerprint(cfg,plan)
    changed=copy.deepcopy(cfg);changed['output']='elsewhere';changed['input']='renamed.png'
    assert recipe_fingerprint(changed,plan)==a
    changed['total_steps']+=1
    assert recipe_fingerprint(changed,plan)!=a


def test_enumeration_and_output_ids_do_not_collide_for_same_basename(tmp_path):
    a=image(tmp_path/'left/a.png');b=image(tmp_path/'right/a.png')
    (tmp_path/'notes.txt').write_text('not an image')
    assert enumerate_images(tmp_path,recursive=True)==[a.resolve(),b.resolve()]
    assert image_id(a.resolve(),tmp_path.resolve())!=image_id(b.resolve(),tmp_path.resolve())


def test_batch_retries_with_resume_and_one_worker_per_gpu(tmp_path):
    source=tmp_path/'inputs';a=image(source/'a.png');b=image(source/'b.png')
    cfg=load_config(smoke=True)
    active={};commands=[]
    class Process:
        next_pid=100
        def __init__(self,command,**kwargs):
            Process.next_pid+=1;self.pid=Process.next_pid
            self.command=command;commands.append(command)
            self.gpu=int(command[command.index('--gpu')+1]);assert self.gpu not in active
            active[self.gpu]=self
            self.calls=0
            config=json.loads(Path(command[command.index('--config')+1]).read_text())
            self.folder=Path(config['output']);self.folder.mkdir()
            self.fail=config['input']==str(a.resolve()) and self.folder.name=='attempt_1'
            if self.fail:
                (self.folder/'resume.pt').write_bytes(b'fixture checkpoint')
                (self.folder/'failure.json').write_text(json.dumps(dict(error='transient fixture failure')))
            else:
                (self.folder/'summary.json').write_text(json.dumps(dict(complete=True,actual_final_N=32,terminal_N=32,
                    actual_final_density=32/(224*192),metrics=dict(PSNR=30.,SSIM=.8,MS_SSIM=.9))))
        def poll(self):
            self.calls+=1
            if self.calls==1:return None
            active.pop(self.gpu,None)
            return 1 if self.fail else 0
    assert run_batch([a,b],source,cfg,tmp_path/'batch',[0,1],1,popen=Process)==0
    jobs=json.loads((tmp_path/'batch/jobs.json').read_text())
    assert len(jobs[0]['attempts'])==2 and all(j['status']=='complete' for j in jobs)
    assert any('--resume' in c for c in commands)
    rows=json.loads((tmp_path/'batch/summary.json').read_text())['jobs']
    assert len(rows)==2 and all(r['actual_final_N']==32 for r in rows)


def test_batch_invalid_image_is_reported_without_training_or_blocking_other_jobs(tmp_path):
    source=tmp_path/'inputs';p=image(source/'too_small.png',(16,16))
    def forbidden(*args,**kwargs):raise AssertionError('Invalid image must not launch a GPU child')
    assert run_batch([p],source,load_config(),tmp_path/'batch',[0],1,popen=forbidden)==1
    summary=json.loads((tmp_path/'batch/summary.json').read_text())
    assert summary['complete'] and summary['jobs'][0]['status']=='failed'


def test_worker_startup_errors_are_bounded_and_recorded(tmp_path):
    source=tmp_path/'inputs';p=image(source/'a.png');calls=[]
    def unavailable(*args,**kwargs):
        calls.append(args);raise OSError('fixture worker unavailable')
    assert run_batch([p],source,load_config(smoke=True),tmp_path/'batch',[0],1,popen=unavailable)==1
    jobs=json.loads((tmp_path/'batch/jobs.json').read_text())
    assert len(calls)==len(jobs[0]['attempts'])==2 and jobs[0]['status']=='failed'


def test_exported_schema_validates_both_modes_and_rejects_unknown_keys():
    jsonschema=pytest.importorskip('jsonschema')
    schema=json.loads((ROOT/'configs/training.schema.json').read_text())
    for cfg in [load_config(),load_config(overrides={'capacity':{'mode':'count','count':400}})]:
        jsonschema.validate(cfg,schema)
    cfg=load_config();cfg['unknown']=True
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(cfg,schema)
