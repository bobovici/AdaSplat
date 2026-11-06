"""Opt-in V4.2 short-A/B parameterization helpers.

This module does not alter V4.1, CUDA, or renderer code.  It supplies smooth,
deterministic q(p), Exact-L2 amplitude mapping, and explicit legacy conversion.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

P_MIN,P_MAX=1.05,2.0
TAU=1.0/255.0
PARAMETERIZATION_VERSIONS={
 "fixed_rms":"v4_2_exact_l2_fixed_rms",
 "effective_support":"v4_2_exact_l2_effective_support",
 "continuous_l2_rms":"v4_2_exact_l2_continuous_l2_rms",
 "continuous_l2_rms_free_scale":"v4_2_exact_l2_continuous_l2_rms_free_scale",
}

def scale_mode_for_schema(schema):
    for mode, version in PARAMETERIZATION_VERSIONS.items():
        if schema == version and mode in ('effective_support','continuous_l2_rms','continuous_l2_rms_free_scale'):
            return mode
    raise ValueError('unsupported scale schema: '+str(schema))

def q_continuous_l2_rms(p):
    """One smooth formula: unit second moment of K_p squared on R^2.

    M(p)=2*Gamma(3/p)/Gamma(1/p), M(2)=1. No p=2 branch.
    Runtime enforces p>=0.1 for FP32; p=0 is outside the mathematical domain.
    """
    return torch.exp(.5*(math.log(.5)+torch.lgamma(1/p)-torch.lgamma(3/p)))
_Q_EFFECTIVE=(0.541783282646,0.553491548327,0.565025480103,0.576387428953,0.587576004771,0.598590891614,0.609430472278,0.620097440832,0.630591068253,0.640913559873,0.651064073626,0.661044280061,0.670855910513,0.680500707473,0.689979791682,0.699296386562,0.708449971945,0.717442943379,0.726278682444,0.73495844691,0.743485018377,0.751859828795,0.760085814947,0.768165062628,0.776099236296,0.783891735944,0.791543909447,0.799058019504,0.806437078665,0.813683155134,0.820798777168,0.827786045763,0.834646444081,0.841383540297,0.847998943777,0.854495526015,0.860875002684,0.867138539379,0.873290152312,0.879330877603,0.885263582405,0.891089295617,0.896810480952,0.902429556278,0.907948091567,0.913368641392,0.918691956473,0.92392084214,0.929056877063,0.934102074527,0.939058190882,0.943926714653,0.948709699568,0.953408064663,0.958024651007,0.962560182842,0.967016742708,0.971395550217,0.975698352367,0.979926650095,0.984081832302,0.988165348308,0.992178574176,0.996123150607,1.0)

def q_fixed_rms(p):
    return torch.exp(-(math.log(2.0)/p+0.5*(torch.lgamma(3.0/p)-torch.lgamma(1.0/p))))

def q_effective_support(p):
    """Audited interpolation on [1.05, 2], with a diagnostic p>2 branch.

    The table has no audited extrapolation beyond two and reaches exactly one
    at p=2.  Upper-bound sensitivity runs therefore use q=1 for p>2 instead of
    silently clipping the kernel p while pretending the compensation was
    extended.  Production runs remain bit-for-bit on the original branch.
    """
    pc=p.clamp(P_MIN,P_MAX);pos=(pc-P_MIN)/(P_MAX-P_MIN)*(len(_Q_EFFECTIVE)-1)
    lo=torch.floor(pos).long().clamp(0,len(_Q_EFFECTIVE)-1);hi=(lo+1).clamp(max=len(_Q_EFFECTIVE)-1)
    table=torch.as_tensor(_Q_EFFECTIVE,dtype=p.dtype,device=p.device);w=pos-lo.to(pos.dtype)
    interpolated=table[lo]*(1-w)+table[hi]*w
    return torch.where(p>P_MAX, torch.ones_like(interpolated), interpolated)

def q_for(mode,p):
    if mode in ('continuous_l2_rms','continuous_l2_rms_free_scale'):return q_continuous_l2_rms(p)
    if mode=="fixed_rms":return q_fixed_rms(p)
    if mode=="effective_support":return q_effective_support(p)
    raise ValueError("unknown V4.2 scale coordinate: %s"%mode)

def exact_l2_norm(p,physical_scales):
    # Log domain avoids an underflowed product of tiny scales at small p.
    return torch.exp(math.log(2.) + torch.lgamma(1.0+1.0/p) +
                     .5*torch.log(physical_scales).sum(dim=-1))

def inverse_softplus(y):
    y=y.clamp_min(1e-8);return y+torch.log(-torch.expm1(-y))

def legacy_to_v42(legacy_scale_raw,p,legacy_mass,legacy_gamma,mode):
    """Preserve initial physical scale and effective opacity exactly.

    Legacy: opacity_eff=m/Z^gamma. New: opacity_eff=a/||K||2.
    Therefore a=opacity_eff*||K||2 and nominal_scale=physical_scale/q(p).
    """
    physical=F.softplus(legacy_scale_raw)+.25
    z=(2**(1+1/p)*torch.exp(torch.lgamma(1+1/p))).square()*physical[:,0]*physical[:,1]
    opacity=legacy_mass/z.pow(float(legacy_gamma));q=q_for(mode,p)
    nominal=physical/q[:,None];raw=inverse_softplus((nominal-.25).clamp_min(1e-8))
    amplitude=opacity*exact_l2_norm(p,physical)
    return {"scale_raw":raw,"amplitude":amplitude,"physical_scales":physical,"effective_opacity":opacity}

def validate_checkpoint_metadata(payload,expected=None):
    version=payload.get("parameterization_version")
    if version is None:return {"legacy":True,"version":None}
    if expected is not None and version!=expected:raise ValueError("parameterization mismatch: expected %s, got %s"%(expected,version))
    return {"legacy":False,"version":version}
