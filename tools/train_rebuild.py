"""Color fitting and population-acceptance helpers."""
import math
import torch

def fit_new_colors(model,target,rows,steps=32,ridge=1e-4):
    """Restricted matrix-free CG: unmodified old colors stay bitwise fixed."""
    flags={n:p.requires_grad for n,p in model.named_parameters()}
    for name,p in model.named_parameters():p.requires_grad_(name=='colors')
    initial=model.colors.detach().clone();x=initial[rows].clone()
    try:
        with torch.no_grad():
            before=float((model.render_unclipped()-target).square().mean())
            model.colors[rows]=0;background=model.render_unclipped().detach()
        def normal(v):
            with torch.no_grad():model.colors[rows]=v
            image=model.render_unclipped()-background
            return torch.autograd.grad(image,model.colors,image.detach())[0][rows]+ridge*v
        image=model.render_unclipped()
        rhs=torch.autograd.grad(image,model.colors,target-background)[0][rows];del image
        def dot(a,b):return (a*b).reshape(-1,3).sum(0)
        def expand(v):return v.repeat(model.colors.shape[1]//3)[None]
        r=rhs-normal(x);d=r.clone();rr=dot(r,r);rr0=rr.clone();done=0
        for _ in range(steps):
            ad=normal(d);alpha=rr/dot(d,ad).clamp_min(1e-20)
            x+=d*expand(alpha);r-=ad*expand(alpha);newrr=dot(r,r);done+=1
            if not torch.isfinite(x).all():break
            if bool((newrr<rr0*1e-9).all()):rr=newrr;break
            d=r+d*expand(newrr/rr.clamp_min(1e-20));rr=newrr
        with torch.no_grad():
            model.colors[rows]=x if torch.isfinite(x).all() else initial[rows]
            after=float((model.render_unclipped()-target).square().mean())
            accepted=math.isfinite(after) and after<=before
            if not accepted:model.colors[rows]=initial[rows]
            oldmask=torch.ones(model.count,dtype=torch.bool,device=rows.device);oldmask[rows]=False
            if not torch.equal(model.colors[oldmask],initial[oldmask]):raise ValueError('old colors changed in restricted solve')
        return dict(updates=done,accepted=accepted,raw_mse_before=before,raw_mse_after=after,ridge=ridge,old_colors_unchanged=True)
    finally:
        for name,p in model.named_parameters():p.requires_grad_(flags[name])

def dark_brightness_penalty(raw,target,mask=None,margin=.04):
    if mask is None:mask=target.mean(1,keepdim=True)<.05
    return (torch.relu(raw-target-margin).square()*mask).mean()

def quality_gate(a,b,dark_a,dark_b,k,weight=.035,dark_policy='strict',dark_anchor=None,dark_guard=None):
    gains=dict(mse=a['MSE']-b['MSE'],ssim=b['SSIM']-a['SSIM'],
               objective=a['MSE']-b['MSE']+weight*(b['SSIM']-a['SSIM']))
    guards=dict(psnr=b['PSNR']>=a['PSNR']-1e-5,ssim=b['SSIM']>=a['SSIM']-1e-6,
                ms_ssim=b.get('MS_SSIM',0)>=a.get('MS_SSIM',0)-1e-5,
                tail=b.get('D_tail',0)<=a.get('D_tail',0)*1.001+1e-9)
    if dark_policy=='strict':
        guards['dark_regions']=all(dark_b[r]['dark_mse']<=dark_a[r]['dark_mse']*1.01+1e-8 and
            dark_b[r]['bright_spot_pixels']<=dark_a[r]['bright_spot_pixels']+2 for r in dark_a)
    elif dark_policy=='continuous':
        if any('bright_leak_energy' not in dark_a[r] or 'bright_leak_energy' not in dark_b[r] for r in dark_a):
            raise ValueError('continuous dark checks require leakage energy metrics')
        guards['dark_regions']=all(dark_b[r]['dark_mse']<=dark_a[r]['dark_mse']*1.02+1e-8 and
            dark_b[r]['bright_leak_energy']<=dark_a[r]['bright_leak_energy']*1.01+1e-10 for r in dark_a)
        if dark_anchor is not None:
            guards['dark_anchor']=all(dark_b[r]['dark_mse']<=dark_anchor[r]['dark_mse']*1.02+1e-8 and
                dark_b[r]['bright_leak_energy']<=dark_anchor[r]['bright_leak_energy']*1.02+1e-10 for r in dark_anchor)
    elif dark_policy=='relaxed':
        limits = dark_guard or dict(min_pixels=256,mse_relative=.20,mse_absolute=1e-5,
                                   leak_relative=.25,leak_absolute=1e-6)
        # A tiny local sample must not veto growth for the whole image. The
        # whole-image check still includes every dark pixel, including these cells.
        checked = [r for r in dark_a if r=='whole' or dark_a[r]['dark_pixels']>=limits['min_pixels']]
        skipped = {r:dark_a[r]['dark_pixels'] for r in dark_a if r not in checked}
        failed = {}
        for region in checked:
            previous,current = dark_a[region],dark_b[region]
            mse_limit = previous['dark_mse']*(1+limits['mse_relative'])+limits['mse_absolute']
            leak_limit = previous['bright_leak_energy']*(1+limits['leak_relative'])+limits['leak_absolute']
            if current['dark_mse']>mse_limit or current['bright_leak_energy']>leak_limit:
                failed[region] = dict(dark_pixels=previous['dark_pixels'],mse_limit=mse_limit,
                    leak_limit=leak_limit,dark_mse=current['dark_mse'],bright_leak_energy=current['bright_leak_energy'])
        guards['dark_regions'] = not failed
        # Compare with the current pre-birth fit, not its obsolete initial errors.
        dark_detail = dict(policy=dark_policy,limits=limits,checked_regions=checked,
            skipped_small_cells=skipped,failed_regions=failed,initial_anchor_used=False)
    else:raise ValueError('unknown dark validation policy')
    guards['positive_gain']=gains['objective']>max(1e-9,k*1e-11)
    decision = dict(guards=guards,gains=gains)
    if dark_policy=='relaxed':decision['dark'] = dark_detail
    return all(guards.values()),decision

