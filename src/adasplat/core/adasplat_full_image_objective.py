"""Exact valid-window Gaussian SSIM, streamed in output-space tiles."""
import torch
from pytorch_msssim import ssim

def full_ssim_value_gradient(image,target,tile=512,window=11):
    """Return the same value/VJP as full-image pytorch_msssim.ssim.

    Valid SSIM output centers are partitioned, input tiles include window-1
    extra pixels. Halo gradients accumulate; no output center is duplicated.
    Only one tile's autograd activations are retained at a time.
    """
    if image.shape!=target.shape or image.ndim!=4:raise ValueError('matching BCHW required')
    h,w=image.shape[-2:];vh,vw=h-window+1,w-window+1
    if min(vh,vw)<=0:raise ValueError('image smaller than SSIM window')
    gradient=torch.zeros_like(image);value=image.new_zeros(())
    for y in range(0,vh,tile):
        for x in range(0,vw,tile):
            ch,cw=min(tile,vh-y),min(tile,vw-x)
            a=image[...,y:y+ch+window-1,x:x+cw+window-1].detach().requires_grad_(True)
            b=target[...,y:y+ch+window-1,x:x+cw+window-1].detach()
            score=ssim(a,b,data_range=1,win_size=window,win_sigma=1.5,size_average=True)
            weight=ch*cw/(vh*vw)
            g=torch.autograd.grad(score,a)[0]
            gradient[...,y:y+ch+window-1,x:x+cw+window-1].add_(g,alpha=weight)
            value=value+score.detach()*weight
    return value,gradient

def full_image_loss_gradient(image,target,ssim_weight=.05,mse_budget=None,budget_weight=10.,scale=32.,tile=512):
    x=image.detach();difference=x-target.detach();mse=difference.square().mean()
    excess=(mse-mse_budget).clamp_min(0) if mse_budget is not None else mse.new_zeros(())
    multiplier=1.+(budget_weight if mse_budget is not None and float(mse)>mse_budget else 0.)
    gradient=difference.mul(2.*multiplier/x.numel());del difference
    score,ssim_gradient=full_ssim_value_gradient(x,target,tile=tile)
    gradient.add_(ssim_gradient,alpha=-ssim_weight).mul_(scale)
    loss=(mse+ssim_weight*(1-score)+budget_weight*excess)*scale
    return loss,gradient,dict(mse=float(mse),ssim=float(score),mse_multiplier=multiplier)
