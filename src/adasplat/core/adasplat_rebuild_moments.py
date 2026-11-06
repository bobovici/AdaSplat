"""C1-tail continuous moments, with differentiable C1 Hermite interpolation.

The table is generated deterministically in float64. It corrects continuous
integrals, not discrete sampled energy. No special branch at p=2.
"""
import math
import functools
import numpy as np
import torch
from primix_v4_2_parameterization import q_continuous_l2_rms, exact_l2_norm

VERSION = 'c1_gamma_moments_4097_128_v1'

def retained_energy(shape, tau=1/255):
    nodes, weights = np.polynomial.legendre.leggauss(128)
    t0, t1 = -2*math.log(4*tau), -2*math.log(tau)
    t = torch.as_tensor(t0+(nodes+1)*(t1-t0)/2, dtype=torch.float64)
    w = torch.as_tensor(weights*(t1-t0)/2, dtype=torch.float64)
    z = ((torch.exp(-t/2)-tau)/(3*tau)).clamp(0,1)
    taper = z.square()*(3-2*z)
    pdf = torch.exp((shape[...,None]-1)*t.log()-t-torch.lgamma(shape[...,None]))
    return torch.special.gammainc(shape, shape.new_full(shape.shape,t0)) + (pdf*w*taper.square()).sum(-1)

@functools.lru_cache(maxsize=8)
def cpu_table(tau):
    logp = torch.linspace(math.log(.1), math.log(8.), 4097, dtype=torch.float64)
    def logs(z):
        p = z.exp()
        e = retained_energy(2/p, tau).log()
        m = retained_energy(4/p, tau).log()
        return torch.stack([e, .5*(m-e)], -1)
    values = logs(logp)
    h = 1e-4
    slopes = (logs(logp-2*h)-8*logs(logp-h)+8*logs(logp+h)-logs(logp+2*h))/(12*h)
    p2 = torch.tensor(2., dtype=torch.float64)
    anchor = .5*(retained_energy(4/p2,tau).log()-retained_energy(2/p2,tau).log())
    return values, slopes, float(anchor)

@functools.lru_cache(maxsize=16)
def device_table(tau, device, dtype):
    values, slopes, anchor = cpu_table(tau)
    return values.to(device=device,dtype=dtype), slopes.to(device=device,dtype=dtype), anchor

def correction_logs(p, tau=1/255):
    values, slopes, anchor = device_table(float(tau),str(p.device),p.dtype)
    step = math.log(80)/4096
    pos = (p.clamp(.1,8).log()-math.log(.1))/step
    index = pos.floor().long().clamp(0,4095)
    z = (pos-index).clamp(0,1)[...,None]
    y0,y1 = values[index],values[index+1]
    d0,d1 = slopes[index]*step,slopes[index+1]*step
    value = (2*z**3-3*z**2+1)*y0+(z**3-2*z**2+z)*d0+(-2*z**3+3*z**2)*y1+(z**3-z**2)*d1
    return value[...,0], value[...,1], anchor

def c1_q(p, tau=1/255):
    _,log_rms,anchor = correction_logs(p,tau)
    return q_continuous_l2_rms(p)*torch.exp(anchor-log_rms)

def c1_norm(p, scales, tau=1/255):
    log_energy,_,_ = correction_logs(p,tau)
    return exact_l2_norm(p,scales)*torch.exp(.5*log_energy)

def c1_slope_factor(p,tau=1/255):
    """Scale u/v so constant and linear color bases have equal continuous L2 energy."""
    _,log_rms,_=correction_logs(p,tau)
    return math.sqrt(2)*q_continuous_l2_rms(p)*torch.exp(-log_rms)
