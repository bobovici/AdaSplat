import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adasplat.paths import bootstrap
bootstrap()
import torch
from pytorch_msssim import ssim
from adasplat_full_image_objective import full_ssim_value_gradient,full_image_loss_gradient
from adasplat_residual_growth import residual_birth_state

def test_full_ssim_matches_value_and_gradient_across_halos():
    torch.manual_seed(1);a=torch.rand(1,3,45,61,dtype=torch.float64,requires_grad=True);b=torch.rand_like(a)
    exact=ssim(a,b,data_range=1,win_size=11,win_sigma=1.5);g=torch.autograd.grad(exact,a)[0]
    value,vg=full_ssim_value_gradient(a,b,tile=17)
    assert torch.allclose(value,exact,atol=1e-12,rtol=1e-10)
    assert torch.allclose(vg,g,atol=1e-12,rtol=1e-9)

def test_full_objective_budget_gradient_matches_direct():
    a=torch.rand(1,3,24,29,requires_grad=True);b=torch.zeros_like(a)
    mse=(a-b).square().mean();loss=(mse+.04*(1-ssim(a,b,data_range=1,win_size=11))+10*torch.relu(mse-.01))*32
    expected=torch.autograd.grad(loss,a)[0]
    actual,g,_=full_image_loss_gradient(a,b,.04,.01,tile=8)
    assert torch.allclose(actual,loss,atol=1e-5)
    assert torch.allclose(g,expected,atol=1e-6,rtol=1e-5)

def test_zero_color_birth_preserves_old_tensors_and_unique_cells():
    state={k:torch.rand((5,3) if k=='colors' else (5,2) if k in ['xys','scale_raw'] else (5,)) for k in ['xys','scale_raw','rotation_angle','mass','colors','p']}
    target=torch.rand(1,3,32,40);merged,info=residual_birth_state(state,torch.zeros_like(target),target,20)
    for k in state:assert torch.equal(merged[k][:5],state[k])
    assert torch.count_nonzero(merged['colors'][5:])==0
    xy=merged['xys'][5:].long()//4
    assert len(torch.unique(xy,dim=0))==20
    assert info['count']==20
