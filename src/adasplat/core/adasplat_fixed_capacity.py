"""Fresh, constant-capacity image fitting with one RGB L2 coefficient per atom."""
import math
import torch
import torch.nn.functional as F
from primix_v4_2_parameterization import inverse_softplus,exact_l2_norm

@torch.no_grad()
def initialize_fixed_capacity(model,target,seed=20261002):
    """A coverage lattice plus unique detail cells, totaling exactly model.count."""
    n=model.count;h,w=target.shape[-2:];device=target.device
    uniform=round(.65*n);detail=n-uniform
    cols=math.ceil(math.sqrt(uniform*w/h));rows=math.ceil(uniform/cols)
    xx=(torch.arange(cols,device=device)+.5)*w/cols-.5
    yy=(torch.arange(rows,device=device)+.5)*h/rows-.5
    gy,gx=torch.meshgrid(yy,xx,indexing='ij')
    grid=torch.stack([gx.flatten(),gy.flatten()],1)
    indices=torch.linspace(0,len(grid)-1,uniform,device=device).round().long()
    base_xy=grid[indices]
    gray=target.mean(1,keepdim=True)
    dx=F.pad(gray[...,1:]-gray[...,:-1],(0,1,0,0))
    dy=F.pad(gray[...,1:,:]-gray[...,:-1,:],(0,0,0,1))
    texture=F.avg_pool2d(dx.square()+dy.square(),4,4,ceil_mode=True)[0,0].sqrt()
    weights=(texture+texture.mean()*.05+1e-8).flatten()
    generator=torch.Generator(device=device).manual_seed(seed)
    cells=torch.multinomial(weights,detail,replacement=False,generator=generator)
    cell_w=texture.shape[1]
    detail_xy=torch.stack([(cells%cell_w)*4+1.5,(cells//cell_w)*4+1.5],1).float()
    detail_xy[:,0].clamp_(0,w-1);detail_xy[:,1].clamp_(0,h-1)
    xy=torch.cat([base_xy,detail_xy]);assert len(xy)==n
    model.xys.copy_(xy);model.p.fill_(2.);model.mass.fill_(1.)
    nominal=torch.empty(n,2,device=device)
    nominal[:uniform]=.58*math.sqrt(h*w/uniform)
    nominal[uniform:,0]=4.;nominal[uniform:,1]=1.5
    model.scale_raw.copy_(inverse_softplus(nominal-.25))
    model.rotation_angle.zero_()
    x=detail_xy[:,0].round().long();y=detail_xy[:,1].round().long()
    model.rotation_angle[uniform:]=torch.atan2(dy[0,0,y,x],dx[0,0,y,x])+math.pi/2
    # Sum the actual native kernels to normalize overlap at each atom center.
    # No checkpoint or over-budget model participates in initialization.
    norm=model.norm() if hasattr(model,'norm') else exact_l2_norm(model.p,model.scales)
    model.colors.zero_();model.colors[:,:3].copy_(norm[:,None].expand(-1,3))
    coverage=model.render_unclipped()[0,0]
    ix=xy[:,0].round().long().clamp(0,w-1);iy=xy[:,1].round().long().clamp(0,h-1)
    local=target[0,:,iy,ix].T
    model.colors[:,:3].copy_(local*norm[:,None]/coverage[iy,ix,None].clamp_min(.1))
    model.mass.requires_grad_(False)
    return dict(N=n,uniform_count=uniform,detail_count=detail,seed=seed,coverage_min=float(coverage.min()),coverage_max=float(coverage.max()),initialization='fresh lattice plus sampled unique detail cells',amplitude='one signed L2 RGB coefficient, fixed unit mass')

def assert_fixed_capacity(model,n):
    if model.count!=n or any(len(v)!=n for v in model.state_dict().values()):
        raise ValueError('fixed primitive count violated')
    if model.mass.requires_grad or not torch.equal(model.mass,torch.ones_like(model.mass)):
        raise ValueError('unit mass invariant violated')

@torch.no_grad()
def clear_replaced_moments(optimizer,model,slots):
    for param in [model.xys,model.scale_raw,model.rotation_angle,model.colors,model.p]:
        for value in optimizer.state.get(param,{}).values():
            if torch.is_tensor(value) and value.shape==param.shape:value[slots]=0

def fit_linear_colors(model,target,steps=64,ridge=1e-4,progress=None):
    """Matrix-free RGB least squares in the exact native, unclipped basis.

    Geometry and count remain fixed. This is a color optimization stage of
    the same model, with no teacher, extra atoms, or dense basis matrix.
    """
    if model.support_mode!='kernel':raise ValueError('linear colors require kernel support')
    flags={name:param.requires_grad for name,param in model.named_parameters()}
    for name,param in model.named_parameters():param.requires_grad_(name=='colors')
    initial=model.colors.detach().clone();x=initial.clone()
    with torch.no_grad():before=float((model.render_unclipped()-target).square().mean())
    def normal(v):
        with torch.no_grad():model.colors.copy_(v)
        image=model.render_unclipped()
        result=torch.autograd.grad(image,model.colors,image.detach())[0]
        return result+ridge*v
    image=model.render_unclipped();rhs=torch.autograd.grad(image,model.colors,target)[0];del image
    def dot(a,b):return (a*b).reshape(-1,3).sum(0)
    def expand(v):return v.repeat(model.colors.shape[1]//3)[None]
    r=rhs-normal(x);direction=r.clone();rr=dot(r,r);rr0=rr.clone();done=0
    for iteration in range(steps):
        ad=normal(direction);den=dot(direction,ad)
        alpha=rr/den.clamp_min(1e-20)
        x=x+direction*expand(alpha);r=r-ad*expand(alpha);newrr=dot(r,r)
        if not torch.isfinite(x).all():break
        done=iteration+1
        if progress is not None and done%8==0:progress(done,float((newrr/rr0.clamp_min(1e-20)).max()))
        if bool((newrr<=rr0*1e-9).all()):rr=newrr;break
        direction=r+direction*expand(newrr/rr.clamp_min(1e-20));rr=newrr
    with torch.no_grad():
        model.colors.copy_(x if torch.isfinite(x).all() else initial)
        after=float((model.render_unclipped()-target).square().mean())
        accepted=math.isfinite(after) and after<=before
        if not accepted:model.colors.copy_(initial)
    for name,param in model.named_parameters():param.requires_grad_(flags[name])
    return dict(updates=done,accepted=accepted,raw_mse_before=before,raw_mse_after=after,ridge=ridge,relative_residual_squared=float((rr/rr0.clamp_min(1e-20)).max()))
