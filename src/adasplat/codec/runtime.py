"""Checkpoint export and inference for the original V6/V2 physical fields."""
from pathlib import Path
import hashlib,math,os
import numpy as np
import torch
import torch.nn.functional as F
from adasplat.paths import ROOT,bootstrap

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!='adasplat_rebuild_v1':
        raise ValueError('Use an inference checkpoint such as final.pt or a milestone, not resume.pt')
    options=payload['options']
    if options.get('moment_correction') or options.get('affine_color') or options.get('pixel_samples',1)!=1:
        raise ValueError('V6/V2 requires continuous L2 calibration, RGB-only coefficients and pixel_samples=1')
    n=int(payload['N']);state=payload['model']
    expected={'xys':(n,2),'scale_raw':(n,2),'rotation_angle':(n,),
              'p':(n,),'colors':(n,3),'mass':(n,)}
    if not 2<=n<=200000 or min(payload['width'],payload['height'])<=0:
        raise ValueError('Invalid population or image dimensions')
    for name,shape in expected.items():
        if tuple(state[name].shape)!=shape or not bool(torch.isfinite(state[name]).all()):
            raise ValueError('Invalid checkpoint field: '+name)
    if bool(((state['p']<.1)|(state['p']>8)).any()):raise ValueError('p outside [0.1,8]')
    return payload

@torch.no_grad()
def physical_fields(payload):
    # Identical FP32 physical formulas; CPU avoids GPU Jiterator workspace.
    bootstrap()
    from primix_v4_2_parameterization import q_continuous_l2_rms,exact_l2_norm
    s={k:v.detach().cpu().float() for k,v in payload['model'].items()}
    p=s['p'];scale=(F.softplus(s['scale_raw'])+.25)*q_continuous_l2_rms(p)[:,None]
    rgb=s['colors']*(s['mass']/exact_l2_norm(p,scale))[:,None]
    result={k:v.numpy().astype('<f4') for k,v in [('xy',s['xys']),('scale',scale),
        ('angle',s['rotation_angle']),('p',p),('rgb',rgb)]}
    if any(not np.isfinite(v).all() for v in result.values()) or np.any(result['scale']<=0):
        raise ValueError('Nonfinite or nonpositive physical fields')
    return result

def header_for(payload):
    return dict(N=payload['N'],width=payload['width'],height=payload['height'],
                cutoff=payload['renderer']['cutoff'],p_min=.1,p_max=8.,support_mode='kernel')

def select_renderer(renderer=None,checkpoint_sha=None,allow_change=False):
    import json
    bootstrap()
    from adasplat_vnext_runtime import select_renderer_binary
    if not torch.cuda.is_available():raise RuntimeError('Rendering requires CUDA; parameter decoding is CPU-only')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    binary=Path(renderer).expanduser().resolve() if renderer else ROOT/('native/csrc.pyd' if os.name=='nt' else 'native/csrc.so')
    if not binary.is_file():raise FileNotFoundError('Build the renderer first: python tools/build_renderer.py')
    digest=sha256(binary)
    if renderer is None:
        manifest=ROOT/'native/renderer.json'
        if not manifest.is_file():raise FileNotFoundError('Missing native/renderer.json; rebuild the renderer')
        if json.loads(manifest.read_text(encoding='utf-8'))['sha256']!=digest:
            raise ValueError('Renderer differs from its build manifest')
    if checkpoint_sha and checkpoint_sha!=digest and not allow_change:
        raise ValueError('Checkpoint uses another renderer build. Use --allow-renderer-change for an explicit cross-build replay; quality is re-evaluated')
    # Prime CUDA operations before the external extension is loaded on Windows.
    warm=torch.full((128,),.5,device='cuda')
    for fn in [torch.cos,torch.sin,torch.exp,torch.log,torch.lgamma,torch.ceil,F.softplus]:fn(warm)
    torch.cuda.synchronize();del warm
    select_renderer_binary(binary,digest)
    return dict(binary=str(binary),sha256=digest,checkpoint_renderer_sha256=checkpoint_sha,
                renderer_changed=bool(checkpoint_sha and checkpoint_sha!=digest))

@torch.no_grad()
def render_fields(header,fields,tile_rows=512):
    from gsplat.rasterize_sum import rasterize_lp
    if tile_rows<1:raise ValueError('tile_rows must be positive')
    n=int(header['N']);h=int(header['height']);w=int(header['width'])
    arrays={k:torch.as_tensor(v,device='cuda',dtype=torch.float32) for k,v in fields.items()}
    xy,scale,angle,p,rgb=[arrays[k] for k in ['xy','scale','angle','p','rgb']]
    if len(xy)!=n:raise ValueError('Population differs from stream header')
    rot=torch.stack([angle.cos(),angle.sin()],-1)
    radius=torch.ceil(3*scale.amax(-1)).to(torch.int32)
    result=np.empty((h,w,3),dtype=np.uint8)
    for y in range(0,h,tile_rows):
        height=min(tile_rows,h-y)
        output=rasterize_lp(xy-xy.new_tensor([0,y]),torch.zeros_like(p),radius,scale,rot,rgb,
            torch.ones_like(p),p,height,w,background=torch.zeros(3,device='cuda'),
            alpha_cutoff=header['cutoff'],p_min=header['p_min'],p_max=header['p_max'],support_mode='kernel')
        result[y:y+height]=output.clamp(0,1).mul(255).round().byte().cpu().numpy()
    return result

def rgb8(path):
    from PIL import Image
    with Image.open(path) as im:return np.array(im.convert('RGB'))

def psnr(prediction,target):
    if prediction.shape!=target.shape:raise ValueError('Image dimensions differ; no implicit resize or rotation')
    total=0
    for y in range(0,len(prediction),128):
        delta=prediction[y:y+128].astype(np.int32)-target[y:y+128].astype(np.int32)
        total+=int(np.square(delta).sum(dtype=np.int64))
    return -10*math.log10(max(total/(prediction.size*255**2),1e-20))

def decode_stream(path):
    from .algorithms import decode
    header,fields=decode(Path(path))
    domain=header.get('V2_scale_domain','linear')
    if domain=='log':fields['scale']=np.exp(fields['scale']).astype('<f4')
    elif domain!='linear':raise ValueError('Unsupported scale domain')
    if any(not np.isfinite(v).all() for v in fields.values()) or np.any(fields['scale']<=0):
        raise ValueError('Invalid decoded physical fields')
    return header,fields
