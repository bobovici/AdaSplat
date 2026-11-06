"""Image output and primitive diagnostics."""
import numpy as np
import torch
from PIL import Image
from atomic_files import atomic_output

def image_save(im,path):
    a=(im[0].permute(1,2,0).detach().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
    with atomic_output(path) as temporary:Image.fromarray(a).save(temporary)

@torch.no_grad()
def diagnostics(model):
    p=model.p;sc=model.scales;minor=sc.amin(1);ratio=sc.amax(1)/minor
    return dict(p_min=float(p.min()),p_max=float(p.max()),p_lt1=int((p<1).sum()),thin_count=int(((minor<.5)&(ratio>8)).sum()),opacity_gt1=int((model.effective_opacity()>1).sum()))

