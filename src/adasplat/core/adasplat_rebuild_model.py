"""Bounded-capacity C1 Lp model. Opt-in moments, quadrature and affine RGB."""
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from primix_v4_2_parameterization import q_continuous_l2_rms,exact_l2_norm,inverse_softplus
from adasplat_rebuild_moments import c1_q,c1_norm,c1_slope_factor,VERSION

@dataclass
class RebuildOptions:
    moment_correction: bool = False
    pixel_samples: int = 1  # 1, 2x2, or 4x4 midpoint quadrature
    affine_color: bool = False
    structure_birth: bool = True
    shape_competition: bool = True
    tangent_novelty: bool = False
    sensitivity_trust: bool = False
    split_merge: bool = False
    adaptive_maturity: bool = True
    cap: int = 114567
    def validate(self):
        if self.pixel_samples not in (1,2,4) or not (1<=self.cap<=200000):
            raise ValueError('invalid quadrature or primitive budget')
        return self

class RebuildModel(nn.Module):
    forward_passes=0
    backward_passes=0
    def __init__(self,target,count,options=None):
        super().__init__()
        self.options=(options or RebuildOptions()).validate()
        if not (1<=count<=self.options.cap):raise ValueError('invalid primitive count or budget exceeded')
        self.height,self.width=target.shape[-2:]
        self.count=count;self.alpha_cutoff=1/255;self.support_mode='kernel'
        self.p_min,self.p_max=.1,8.
        device=target.device
        self.xys=nn.Parameter(torch.zeros(count,2,device=device))
        self.scale_raw=nn.Parameter(torch.zeros(count,2,device=device))
        self.rotation_angle=nn.Parameter(torch.zeros(count,device=device))
        self.p=nn.Parameter(torch.full((count,),2.,device=device))
        self.colors=nn.Parameter(torch.zeros(count,9 if self.options.affine_color else 3,device=device))
        self.register_buffer('mass',torch.ones(count,device=device))

    @property
    def nominal_scales(self):return F.softplus(self.scale_raw)+.25
    def q(self,p):return c1_q(p) if self.options.moment_correction else q_continuous_l2_rms(p)
    @property
    def scales(self):return self.nominal_scales*self.q(self.p)[:,None]
    @property
    def rotations(self):return torch.stack([self.rotation_angle.cos(),self.rotation_angle.sin()],-1)
    def norm(self):return c1_norm(self.p,self.scales) if self.options.moment_correction else exact_l2_norm(self.p,self.scales)
    def effective_opacity(self):return 1/self.norm()
    def gaussian_equivalent_radii(self):return torch.ceil(3*self.scales.detach().amax(-1)).int()
    def render_unclipped(self):
        from gsplat.rasterize_sum import rasterize_lp
        scales=self.scales;norm=self.norm();angle=self.rotations
        samples=self.options.pixel_samples;image=None
        colors=self.colors
        if self.options.affine_color:
            # Normalize the two slope bases by their actual C1 RMS. Raw u/v
            # can be enormous at p<1 and makes the color solve ill-conditioned.
            slope=c1_slope_factor(self.p)[:,None]
            colors=torch.cat([colors[:,:3],colors[:,3:]*slope],1)
        for iy in range(samples):
            for ix in range(samples):
                offset=self.xys.new_tensor([(ix+.5)/samples-.5,(iy+.5)/samples-.5])
                value=rasterize_lp(self.xys-offset,torch.zeros_like(self.p),self.gaussian_equivalent_radii(),scales,angle,
                    colors,1/norm,self.p,self.height,self.width,background=self.xys.new_zeros(3),
                    alpha_cutoff=self.alpha_cutoff,p_min=.1,p_max=8.,support_mode='kernel',affine_color=self.options.affine_color)
                RebuildModel.forward_passes+=1
                image=value if image is None else image+value
        output=(image/(samples*samples)).permute(2,0,1)[None].contiguous()
        if output.requires_grad:
            def count_backward(gradient):
                RebuildModel.backward_passes+=samples*samples
                return gradient
            output.register_hook(count_backward)
        return output
    def forward(self):return self.render_unclipped().clamp(0,1)
    @torch.no_grad()
    def project_parameters_(self):
        self.xys[:,0].clamp_(0,self.width-1);self.xys[:,1].clamp_(0,self.height-1)
        self.p.clamp_(.1,8.);self.scale_raw.clamp_(min=-15.,max=256.)
        if self.count>self.options.cap or any(len(v)!=self.count for v in self.state_dict().values()):
            raise ValueError('primitive budget violated')
        if not all(torch.isfinite(v).all() for v in self.state_dict().values()):raise ValueError('nonfinite model')

    @torch.no_grad()
    def load_baseline(self,state,preserve_basis=True):
        source={k:v.to(self.xys.device) for k,v in state.items()}
        if len(source['p'])!=self.count:raise ValueError('count mismatch')
        if preserve_basis and self.options.moment_correction:
            physical=(F.softplus(source['scale_raw'])+.25)*q_continuous_l2_rms(source['p'])[:,None]
            nominal=physical/self.q(source['p'])[:,None]
            if bool((nominal<=.25).any()):
                raise ValueError('moment coordinate cannot preserve old basis below its nominal floor; use fresh initialization')
            source['scale_raw']=inverse_softplus(nominal-.25)
            source['colors']=source['colors']*(c1_norm(source['p'],physical)/exact_l2_norm(source['p'],physical))[:,None]
        if self.options.affine_color and source['colors'].shape[1]==3:
            source['colors']=F.pad(source['colors'],(0,6))
        self.load_state_dict(source,strict=True)

    def metadata(self):
        return dict(schema='adasplat_rebuild_v1',options=asdict(self.options),height=self.height,width=self.width,
                    N=self.count,p_range=[.1,8.],moment_table=VERSION if self.options.moment_correction else None,
                    trainable_parameters=sum(p.numel() for p in self.parameters()),fixed_mass=1.,
                    normalization='continuous_C1_kernel_L2_before_pixel_filter' if self.options.moment_correction else 'continuous_infinite_kernel_L2_before_pixel_filter',
                    sampled_basis='midpoint_pixel_area' if self.options.pixel_samples>1 else 'pixel_center')

def patch_basis(xy,scales,angle,p,grid,options):
    """Torch preview of exactly the native C1/normalization/quadrature basis.

    grid is P x 2 pixel centers. Return N x P x B (B=1 or 3).
    The affine coordinates share the renderer convention center minus pixel.
    """
    norm=c1_norm(p,scales) if options.moment_correction else exact_l2_norm(p,scales)
    c,s=angle.cos()[:,None],angle.sin()[:,None]
    result=None;count=options.pixel_samples
    for iy in range(count):
        for ix in range(count):
            offset=grid.new_tensor([(ix+.5)/count-.5,(iy+.5)/count-.5])
            pixel_grid=grid[None,:,:] if grid.ndim==2 else grid
            delta=xy[:,None,:]-pixel_grid-offset
            u=(c*delta[...,0]+s*delta[...,1])/scales[:,0,None]
            v=(-s*delta[...,0]+c*delta[...,1])/scales[:,1,None]
            # Clamp only before powers to implement the native symmetric zero
            # subgradient on the exact axis and avoid 0**p log(0) NaNs.
            au=u.abs();av=v.abs()
            up=torch.where(au==0,torch.zeros_like(au),au.clamp_min(1e-30).pow(p[:,None]))
            vp=torch.where(av==0,torch.zeros_like(av),av.clamp_min(1e-30).pow(p[:,None]))
            kernel=torch.exp(-.5*(up+vp));z=((kernel-1/255)/(3/255)).clamp(0,1)
            kernel=kernel*z.square()*(3-2*z)/norm[:,None]
            if options.affine_color:
                slope=c1_slope_factor(p)[:,None]
                value=kernel[...,None]*torch.stack([torch.ones_like(u),u*slope,v*slope],-1)
            else:value=kernel[...,None]
            result=value if result is None else result+value
    return result/(count*count)
