"""Vectorized residual proposals and function-preserving tensor append."""
import torch
import torch.nn.functional as F
from primix_v4_2_parameterization import exact_l2_norm,inverse_softplus

FIELDS=('xys','scale_raw','rotation_angle','mass','colors','p')

@torch.no_grad()
def residual_birth_state(state,prediction,target,count,seed=20261002,cell=4):
    if count<=0:raise ValueError('positive birth count required')
    r=(target-prediction).square().mean(1,keepdim=True)
    lum=target.mean(1,keepdim=True)
    # Spatial cells bound crowding; contrast weighting also proposes smooth dark
    # errors. This score ranks births, it is not the training/evaluation metric.
    var=(F.avg_pool2d(lum.square(),11,1,5)-F.avg_pool2d(lum,11,1,5).square()).clamp_min(0)
    score=r*(1+2/(1+var/.01))
    h,w=target.shape[-2:];ph,pw=h//cell,w//cell
    if count>ph*pw:raise ValueError('birth budget exceeds unique spatial cells')
    pooled=F.avg_pool2d(score[...,:ph*cell,:pw*cell],cell,cell).flatten()
    generator=torch.Generator(device=target.device).manual_seed(seed)
    indices=torch.multinomial(pooled.clamp_min(1e-12),count,replacement=False,generator=generator)
    # Peak residual inside selected cells, with unique cell indices.
    cells=score[0,0,:ph*cell,:pw*cell].reshape(ph,cell,pw,cell).permute(0,2,1,3).reshape(ph*pw,cell*cell)
    offsets=cells[indices].argmax(1)
    x=(indices%pw)*cell+offsets%cell;y=(indices//pw)*cell+offsets//cell
    xys=torch.stack((x,y),-1).to(target.dtype)
    # Different scales offer both detail and broader residual correction.
    sizes=target.new_tensor([.8,1.5,3.,6.])
    choice=torch.randint(0,len(sizes),(count,),device=target.device,generator=generator)
    physical=sizes[choice,None].expand(-1,2).clone();p=target.new_full((count,),2.)
    born=dict(xys=xys,scale_raw=inverse_softplus(physical-.25),rotation_angle=target.new_zeros(count),mass=.15*exact_l2_norm(p,physical),colors=target.new_zeros(count,3),p=p)
    combined={k:torch.cat((state[k].detach(),born[k]),0) for k in FIELDS}
    return combined,dict(count=count,old_count=len(state['p']),scales=[.8,1.5,3.,6.],color_zero=True,p_birth=2.,proposal_seed=seed,proposal_cell=cell)
