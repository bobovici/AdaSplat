"""Public workflow regression tests: CPU-only; no training or original assets."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from adasplat.training.cli import single_main,batch_main
from adasplat.training.config import load_config,resolve_plan
from adasplat.training.pipeline import encode_completed_run
from adasplat.training.devices import resolve_gpus

def picture(tmp_path,size=(4928,3264)):
    p=tmp_path/'arbitrary-name.png';Image.new('RGB',size).save(p);return p

def test_public_defaults_and_quick_profile(tmp_path,capsys):
    image=picture(tmp_path)
    assert single_main(['--input',str(image),'--dry-run'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['resolved_plan']['initial_N']==30000
    assert data['resolved_plan']['requested_cap']==200000
    assert data['resolved_plan']['total_steps']==50000
    assert single_main(['--input',str(image),'--preset','quick','--dry-run'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['resolved_plan']['requested_cap']==20000
    assert data['resolved_plan']['total_steps']==2000

def test_density_override_units_and_cap(tmp_path,capsys):
    image=picture(tmp_path,(1000,1000))
    assert single_main(['--input',str(image),'--initial-density','2','--max-density','10','--dry-run'])==0
    plan=json.loads(capsys.readouterr().out)['resolved_plan']
    assert (plan['initial_N'],plan['requested_cap'])==(2000,10000)
    cfg=load_config(ROOT/'configs/standard.json')
    cfg['capacity']['density']=10
    assert resolve_plan(cfg,image)['requested_cap']==200000

def test_encoding_flags_and_overrides(tmp_path,capsys):
    image=picture(tmp_path,(224,192))
    assert single_main(['--input',str(image),'--encode-milestones','--codec-max-loss-db','.01','--dry-run'])==0
    enc=json.loads(capsys.readouterr().out)['config']['encoding']
    assert enc==dict(enabled=True,milestones=True,max_loss_db=.01)
    assert single_main(['--input',str(image),'--encode','--no-encode','--dry-run'])==0
    assert not json.loads(capsys.readouterr().out)['config']['encoding']['enabled']

def test_density_conflict_rejected(tmp_path):
    with pytest.raises(SystemExit): single_main(['--input',str(picture(tmp_path)),'--initial-density','1','--initial-ratio','.2','--dry-run'])

def test_auto_devices_respects_existing_mask(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3,1')
    assert resolve_gpus('auto')==[3,1]
    assert resolve_gpus('2,0')==[2,0]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    with pytest.raises(ValueError):resolve_gpus('auto')

def test_batch_defaults_one_job_per_visible_gpu(tmp_path,capsys,monkeypatch):
    picture(tmp_path,(224,192));monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    assert batch_main(['--input-dir',str(tmp_path),'--dry-run'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['gpus']==[2,3] and data['jobs_per_gpu']==1 and data['max_parallel_jobs']==2

def test_pipeline_real_command_propagation_and_reports(tmp_path,monkeypatch):
    cfg=load_config();cfg['encoding'].update(enabled=True,milestones=True)
    (tmp_path/'milestones').mkdir()
    (tmp_path/'milestones/index.json').write_text(json.dumps({'entries':[{'status':'saved','checkpoint':'milestones/n1.pt'}]}))
    commands=[];monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    def run(command,**kwargs):
        commands.append(command)
        assert '--gpu' not in command and os.environ['CUDA_VISIBLE_DEVICES']=='3'
        stream=Path(command[command.index('--output')+1])
        stream.with_suffix('.adagc.json').write_text(json.dumps(dict(bytes=100,bpp=.5,loss_db=.001)))
        return SimpleNamespace(returncode=0)
    report=encode_completed_run(cfg,tmp_path,'original.png',runner=run)
    assert len(commands)==len(report['artifacts'])==2
    assert report['status']=='complete'
    assert '--target' in commands[0] and '--decoded-output' in commands[0]

def test_pipeline_failure_retains_training_outputs(tmp_path):
    cfg=load_config();(tmp_path/'final.pt').write_bytes(b'test marker')
    with pytest.raises(RuntimeError):
        encode_completed_run(cfg,tmp_path,'original.png',runner=lambda *a,**k:SimpleNamespace(returncode=1))
    assert (tmp_path/'final.pt').read_bytes()==b'test marker'
    assert json.loads((tmp_path/'encoding.json').read_text())['training_complete']

def test_smoke_encoding_uses_the_same_crop_as_training(tmp_path):
    image=picture(tmp_path,(800,400));cfg=load_config(smoke=True)
    def run(command,**kwargs):
        target=Path(command[command.index('--target')+1])
        with Image.open(target) as cropped: assert cropped.size==(224,192)
        stream=Path(command[command.index('--output')+1])
        stream.with_suffix('.adagc.json').write_text(json.dumps(dict(bytes=100,bpp=.5,loss_db=.001)))
        return SimpleNamespace(returncode=0)
    assert encode_completed_run(cfg,tmp_path,image,runner=run)['status']=='complete'

@pytest.mark.parametrize('complete,enabled,expected',[(True,True,1),(False,True,0),(True,False,0)])
def test_single_entry_dispatches_codec_only_after_complete_fit(tmp_path,monkeypatch,complete,enabled,expected):
    import types
    import adasplat.training.pipeline as pipeline
    calls=[]
    class Trainer:
        def __init__(self,*args,**kwargs):pass
        def fit(self,**kwargs):return {'complete':complete}
    monkeypatch.setitem(sys.modules,'adasplat.training.trainer',types.SimpleNamespace(Trainer=Trainer))
    monkeypatch.setattr(pipeline,'check_codec_dependency',lambda:None)
    monkeypatch.setattr(pipeline,'encode_completed_run',lambda *args:calls.append(args))
    image=picture(tmp_path,(224,192))
    command=['--input',str(image),'--output',str(tmp_path/'out')]
    if enabled:command+=['--encode']
    assert single_main(command)==0
    assert len(calls)==expected
