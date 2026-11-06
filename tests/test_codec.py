"""Real complete-stream round trips, corruption detection and fresh-process decode."""
from pathlib import Path
import json,os,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from adasplat.codec.encoder import encode
from adasplat.codec.runtime import decode_stream
from adasplat.codec.algorithms import read_stream,write_stream

@pytest.mark.parametrize('bits',[16,24])
def test_complete_stream_independent_decode_and_precision(tmp_path,bits):
    rng=np.random.RandomState(8);n=32
    raw=dict(xy=(rng.uniform(0,1,(n,2))*[223,191]).astype('<f4'),scale=rng.uniform(.2,8,(n,2)).astype('<f4'),
             angle=rng.uniform(0,6.28,n).astype('<f4'),p=rng.uniform(.1,8,n).astype('<f4'),
             rgb=rng.uniform(-.2,1.2,(n,3)).astype('<f4'))
    raw['xy'][:3]=[10.,20.]  # Collision ranks must survive independent reconstruction.
    payload=dict(N=n,width=224,height=192,renderer={'cutoff':1/255})
    stream=tmp_path/'model.adagc';encode(raw,payload,stream,'a'*64,blocks=4,bits=bits)
    h,decoded=decode_stream(stream)
    assert h['N']==n and h['final_artifact_sha256']=='a'*64
    fixed=np.rint(raw['xy']*256).astype(np.int64)
    order=np.lexsort((np.arange(n),fixed[:,1],fixed[:,0]))
    expected={k:v[order] for k,v in raw.items()}
    assert np.max(np.abs(expected['xy']-decoded['xy']))<=1/512+1e-5
    assert np.max(np.abs(expected['scale']/decoded['scale']-1))<1e-4
    assert np.max(np.abs(expected['p']-decoded['p']))<7e-5
    assert np.max(np.abs(expected['rgb']-decoded['rgb']))<(1e-4 if bits==16 else 1e-5)
    out=tmp_path/'fields.npz'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    p=subprocess.run([sys.executable,str(ROOT/'codec.py'),'decode','--input',str(stream),'--output',str(out)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',env=env)
    assert p.returncode==0,p.stderr
    with np.load(out,allow_pickle=False) as saved:
        for k,v in decoded.items():assert np.array_equal(saved[k],v)
    # Preserve entropy bytes, tamper only the expected integrity hash.
    header,sections=read_stream(stream);header['uncompressed_section_sha256']['scale_u16']='0'*64
    broken=tmp_path/'corrupt.adagc';write_stream(broken,header,list(sections.items()))
    with pytest.raises(ValueError,match='hash mismatch'):decode_stream(broken)

def test_codec_help_does_not_require_a_renderer():
    p=subprocess.run([sys.executable,str(ROOT/'codec.py'),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert p.returncode==0 and b'encode' in p.stdout and b'decode' in p.stdout
