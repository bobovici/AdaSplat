"""Full-image coarse SSIM and exhaustive deterministic fine-scale coverage."""
import random
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def coverage_boxes(height,width,patch,step,count=2,seed=20261002):
    boxes=[(y,x,min(y+patch,height),min(x+patch,width)) for y in range(0,height,patch) for x in range(0,width,patch)]
    # Tiny trailing strips are included by shifting their last window inward.
    boxes=[(min(y,max(0,height-patch)),min(x,max(0,width-patch)),min(y+patch,height),min(x+patch,width)) for y,x,_,_ in boxes]
    cycle=(step*count)//len(boxes);offset=(step*count)%len(boxes)
    order=list(range(len(boxes)));random.Random(seed+cycle).shuffle(order)
    result=[]
    for k in range(count):
        index=offset+k
        if index>=len(boxes):
            next_order=list(range(len(boxes)));random.Random(seed+cycle+1).shuffle(next_order)
            result.append(boxes[next_order[index-len(boxes)]])
        else:result.append(boxes[order[index]])
    return result

def multiscale_structure_loss(image,target,config,step=0):
    patch=config.get('structure_patch_size',512)
    boxes=coverage_boxes(*image.shape[-2:],patch,step,count=config.get('structure_patch_count',2),seed=config.get('structure_seed',20261002))
    fine=image.new_zeros(());gradient=image.new_zeros(())
    for y,x,y1,x1 in boxes:
        a=image[...,y:y1,x:x1].contiguous();b=target[...,y:y1,x:x1].contiguous()
        fine=fine+1-ssim(a,b,data_range=1,win_size=11,win_sigma=1.5)
        dx=(a[...,1:]-a[...,:-1])-(b[...,1:]-b[...,:-1])
        dy=(a[...,1:,:]-a[...,:-1,:])-(b[...,1:,:]-b[...,:-1,:])
        gradient=gradient+dx.square().mean()+dy.square().mean()
    fine=fine/len(boxes);gradient=gradient/len(boxes)
    global_loss=image.new_zeros(())
    for factor,weight in [(4,.2),(8,.05)]:
        a=F.avg_pool2d(image,factor,factor);b=F.avg_pool2d(target,factor,factor)
        global_loss=global_loss+weight*(1-ssim(a,b,data_range=1,win_size=11,win_sigma=1.5))
    return .75*fine+global_loss,gradient

def multiscale_reconstruction_loss(image,target,config,step=0):
    mse=(image-target).square().mean()
    structure,gradient=multiscale_structure_loss(image,target,config,step)
    ramp=min(1.,max(0.,step/config.get('ssim_ramp_steps',2000)))
    weight=config.get('ssim_start_weight',.01)+(config.get('ssim_weight',.05)-config.get('ssim_start_weight',.01))*ramp
    loss=mse+weight*structure+config.get('gradient_weight',.05)*gradient
    if config.get('mse_budget') is not None:
        loss=loss+config.get('mse_budget_weight',10)*torch.relu(mse-config['mse_budget'])
    return loss*config.get('scale',32.)
