"""Repository-local paths; generated run contracts pin resolved paths."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def bootstrap():
    core=ROOT/'src/adasplat/core';engine=ROOT/'src/adasplat/engine'
    for p in [core,core/'gsplat',engine]:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    return engine
def resolve_config(path):
    path=Path(path)
    if path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        c=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    else:c=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(c,dict):raise ValueError('configuration must be a mapping')
    def absolute(p):return str((ROOT/p).resolve()) if not Path(p).is_absolute() else p
    c['production_source']=str(ROOT/'src/adasplat/core')
    c['output']=absolute(c['output']);r=c['recipe'];r['checkpoint']['output']=c['output']
    r['loss'].setdefault('ssim_gaussian',False)
    r['loss'].setdefault('gradient_weight',0.0)
    r['loss'].setdefault('scale',1.0)
    r['loss'].setdefault('structure_patch_size',0)
    r['loss'].setdefault('structure_seed',20261002)
    r['renderer']['binary']=absolute(r['renderer']['binary'])
    if r['evaluation']['target']:r['evaluation']['target']=absolute(r['evaluation']['target'])
    if r['fit']['legacy_checkpoint']:r['fit']['legacy_checkpoint']=absolute(r['fit']['legacy_checkpoint'])
    return c
