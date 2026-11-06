"""Preserved V6/V2 complete-stream encoder; portable imports only."""
import hashlib,lzma,math,zlib
from pathlib import Path
import numpy as np
import torch
from . import algorithms as ra
from . import algorithms as repack
from .algorithms import uniform_encode,write_stream
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(parameters,payload,path,source_hash,blocks=320,bits=16):
    n=payload['N'];height,width=payload['height'],payload['width'];factor=256
    fixed=np.rint(parameters['xy']*factor).astype(np.int64)
    order=np.lexsort((np.arange(n),fixed[:,1],fixed[:,0]))
    a={k:v[order] for k,v in parameters.items()}
    pos,xy=ra.fixed_position_encode(a['xy'],factor,width,height)
    keys,depth=ra.point_keys(xy,width,height);raht,laplace=ra.load_official()
    coeff=raht.haar3D(keys,a['rgb']@ra._rgb_basis().T,depth)['CT']
    # Original FP32 coefficients, not an intermediate FP16 RGB payload.
    codes,lo,step,lengths=ra.quantize_ac(coeff,bits,blocks)
    high=codes if bits==16 else codes>>8
    packed=laplace.laplace_encode_blocks(high,lengths,3,max_val=65535,auto_model=True)
    restored,rl,channels=laplace.laplace_decode_blocks(packed)
    assert channels==3 and rl==lengths and np.array_equal(restored,high)
    raw=dict(position_fixed_delta_u32=pos,rgb_raht_dc_f32=coeff[0].astype('<f4').tobytes(),
        rgb_raht_lo_f32=lo.tobytes(),rgb_raht_step_f32=step.tobytes(),rgb_raht_laplace=packed)
    if bits==24:
        low=codes&255
        packed_low=laplace.laplace_encode_blocks(low,lengths,3,max_val=255,auto_model=True)
        low_restored,ll,lc=laplace.laplace_decode_blocks(packed_low)
        assert ll==lengths and lc==3 and np.array_equal(low_restored,low)
        raw['rgb_raht_laplace_low8']=packed_low
    h=dict(schema='adasplat_mesongs_plus_raht_laplace_v1',N=n,width=width,height=height,
        p_bits=16,p_min=.1,p_max=8.,cutoff=payload['renderer']['cutoff'],support_mode='kernel',
        position_encoding='fixed_delta_zigzag_u32',position_fixed_factor=factor,position_byte_shuffled=False,
        primitive_order='spatial_xy',V2_scale_domain='log',scale_bits=16,rotation_bits=16,
        rgb_bits=bits,rgb_blocks=blocks,raht_depth=depth,rgb_transform='orthonormal_lcc',
        V2_RGB_lsb_bits=8 if bits==24 else 0,
        rgb_entropy_family_auto=True,rgb_mixed_precision=False,rgb_bits_by_channel_block=None,
        field_packing={},upstream_commit='0004153458b73387e0044b6f553f62116117388d',
        raht_sha256=sha(ra.RAHT_SOURCE),laplace_sha256=sha(ra.LAPLACE_SOURCE),
        final_artifact_sha256=source_hash,zero_training_updates=True,
        precision_configuration='V2 V6-logscale: XY256, logscale16, angle16, p16, RGB16/24 quality guard',
        RGB_source='original effective coefficients float32; no intermediate float16')
    for key,name in [('scale','scale_u16'),('angle','rotation_u16'),('p','p_sidecar')]:
        value=torch.from_numpy(a[key])
        if key=='scale':value=value.log()
        if key=='angle':value=torch.remainder(value,2*math.pi)
        if key=='p':code=((value-.1)/7.9*65535).round().clamp(0,65535).long()
        else:
            code,low,spacing=uniform_encode(value,16)
            if key=='scale':h.update(scale_lo=low.cpu().tolist(),scale_step=spacing.cpu().tolist())
            else:h.update(rotation_lo=float(low),rotation_step=float(spacing))
        raw[name]=repack.shuffle(code.cpu().numpy().astype(np.uint16).reshape(-1))
        h['field_packing'][name]='u16_byte_shuffle'
    sections=[];codecs={}
    for name,data in raw.items():
        if name.startswith('rgb_raht_laplace'):codec,coded='mesongs_plus_laplace',data
        else:
            choices=[('zlib9',zlib.compress(data,9)),('lzma6',lzma.compress(data,preset=6))]
            codec,coded=min(choices,key=lambda x:len(x[1]))
        sections.append((name,coded));codecs[name]=codec
    h.update(section_lossless_codecs=codecs,uncompressed_section_bytes={k:len(v) for k,v in raw.items()},
        uncompressed_section_sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()})
    path.parent.mkdir(parents=True,exist_ok=True);write_stream(path,h,sections)
    return h
