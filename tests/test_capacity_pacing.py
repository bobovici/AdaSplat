"""Capacity recovery and continuous leakage checks for stalled growth."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from adasplat.paths import bootstrap
bootstrap()
from adasplat_rebuild_growth import DeadlineBatch
from train_rebuild import quality_gate,dark_brightness_penalty

def metric_pair():
    return (dict(MSE=.001,PSNR=30.,SSIM=.85,MS_SSIM=.96,D_tail=.003),
        dict(MSE=.0009,PSNR=30.45,SSIM=.851,MS_SSIM=.961,D_tail=.0029))

def test_budget_is_used_by_deadline_despite_biased_local_prediction():
    batch=DeadlineBatch();count=51968;a,b=metric_pair();sizes=[]
    for index,step in enumerate(range(22500,32001,500)):
        k=batch.request(step,count)
        if not k:break
        accepted=index%5!=4;sizes.append(k)
        batch.update(accepted,a,b,k,predicted=1.,ssim_weight=.035)
        if accepted:count+=k
        assert count<=114567
    assert count==114567 and max(sizes)>2048

def test_rejection_does_not_ratchet_batch_down_to_128():
    batch=DeadlineBatch();a,b=metric_pair();sizes=[]
    for step in range(22000,24001,500):
        k=batch.request(step,51968);sizes.append(k);batch.update(False,a,b,k,predicted=1.)
    assert min(sizes)>=2048 and sizes[-1]>=sizes[0]

def test_final_request_respects_exact_remaining_slots_and_deadline():
    batch=DeadlineBatch()
    assert batch.request(32000,114530)==37
    assert batch.request(32500,51968)==0
    assert batch.request(30000,114567)==0

def test_count_threshold_flicker_does_not_veto_lower_leakage_energy():
    a,b=metric_pair()
    before=dict(left=dict(dark_mse=.0001,bright_leak_energy=.00001,bright_spot_pixels=100))
    after=dict(left=dict(dark_mse=.00009,bright_leak_energy=.000009,bright_spot_pixels=104))
    assert not quality_gate(a,b,before,after,4096)[0]
    assert quality_gate(a,b,before,after,4096,dark_policy='continuous',dark_anchor=before)[0]

def test_real_leakage_regression_is_still_rejected():
    a,b=metric_pair();before=dict(left=dict(dark_mse=.0001,bright_leak_energy=.00001,bright_spot_pixels=100))
    after=dict(left=dict(dark_mse=.00009,bright_leak_energy=.00002,bright_spot_pixels=90))
    assert not quality_gate(a,b,before,after,4096,dark_policy='continuous',dark_anchor=before)[0]

def test_fixed_anchor_prevents_compounding_small_regressions():
    a,b=metric_pair();anchor=dict(left=dict(dark_mse=.0001,bright_leak_energy=.00001))
    before=dict(left=dict(dark_mse=.0001,bright_leak_energy=.00001019))
    after=dict(left=dict(dark_mse=.0001,bright_leak_energy=.00001028))
    accepted,decision=quality_gate(a,b,before,after,4096,dark_policy='continuous',dark_anchor=anchor)
    assert decision['guards']['dark_regions'] and not decision['guards']['dark_anchor'] and not accepted

def test_raw_dark_leakage_gradient_recovers_saturated_white_pixels():
    raw=torch.full((1,3,4,4),2.,requires_grad=True);target=torch.zeros_like(raw)
    ((raw.clamp(0,1)-target).square().mean()).backward()
    assert torch.equal(raw.grad,torch.zeros_like(raw))
    raw.grad=None;dark_brightness_penalty(raw,target).backward()
    assert bool((raw.grad>0).all())
    bright=torch.full_like(raw,.8);raw.grad=None;dark_brightness_penalty(raw,bright).backward()
    assert torch.equal(raw.grad,torch.zeros_like(raw))
