"""Strict configuration and resolution-aware budgets; no CUDA imports here."""
import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'adasplat_training_v1'
HARD_CAP = 200000
REFERENCE_CAP = 114567
DEFAULTS = {
    'schema': SCHEMA, 'input': None, 'output': None, 'total_steps': 40000,
    'capacity': {'mode': 'density', 'density': REFERENCE_CAP/(4928*3264), 'count': None,
                 'initial_ratio': 32768/REFERENCE_CAP},
    'growth': {'enabled': True, 'fraction': .8, 'events': 16,
               'min_birth_ratio': .02, 'max_birth_ratio': .08},
    'early_stop': {'enabled': True, 'patience': 3, 'min_psnr_gain': .01,
                   'min_ssim_gain': .0005, 'eval_interval': 500, 'min_fixed_steps': 2000},
    'snapshots': {'counts': [], 'tolerance': .1},
    'seed': 20261002,
    'advanced': {
        'mse_fraction': .1, 'ssim_ramp_fraction': .1, 'ssim_weight': .035,
        'ssim_interval': 8, 'evaluate_every': 1000,
        'initial_cg': 64, 'new_color_cg': 16, 'warm_min': 64, 'warm_max': 192,
        'joint_steps': 256, 'threads': 2, 'ssim_tile': 1024,
        'dark_leak_weight': .5, 'dark_threshold': .05, 'dark_margin': .04,
        'dark_grid': 4, 'regularizer_weight': .01,
        'growth_reference': 'pre_birth', 'checkpoint_selection': 'within_cap',
        'optimizer': {'rates': {'xys': .03, 'scale_raw': .01, 'rotation_angle': .003,
                               'colors': .025, 'p': .003},
                      'betas': [.98, .92, .99], 'eps': 1e-8},
        'options': {'moment_correction': False, 'pixel_samples': 1,
                    'affine_color': False, 'structure_birth': False,
                    'shape_competition': True, 'tangent_novelty': False,
                    'sensitivity_trust': False, 'split_merge': False,
                    'adaptive_maturity': True},
        'renderer_sha256': '0000000000000000000000000000000000000000000000000000000000000000'
    },
    'preprocess': {'smoke_crop': False}
}


def merge_strict(base, updates, path=''):
    if not isinstance(updates, dict):
        raise ValueError('Expected an object at '+(path or 'root'))
    for key, value in updates.items():
        label = path+'.'+key if path else key
        if key not in base:
            raise ValueError('Unknown configuration key: '+label)
        if isinstance(base[key], dict):
            merge_strict(base[key], value, label)
        else:
            base[key] = copy.deepcopy(value)
    return base


def number(value, name, minimum=None, maximum=None, integer=False):
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid or not math.isfinite(value) or (integer and not isinstance(value, int)):
        raise ValueError(name+' must be a finite '+('integer' if integer else 'number'))
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(name+' outside the supported range')
    return value


def validate(cfg):
    if cfg['schema'] != SCHEMA:
        raise ValueError('Unsupported configuration schema')
    for key in ('input','output'):
        if cfg[key] is not None and not isinstance(cfg[key],str):
            raise ValueError(key+' must be a path string or null')
    number(cfg['total_steps'],'total_steps',2,integer=True)
    number(cfg['seed'],'seed',0,2**32-1,integer=True)
    cap = cfg['capacity']
    if cap['mode'] == 'density':
        number(cap['density'],'capacity.density',1e-12)
        if cap['count'] is not None:
            raise ValueError('density and count modes are mutually exclusive')
    elif cap['mode'] == 'count':
        number(cap['count'],'capacity.count',2,HARD_CAP,integer=True)
        if cap['density'] is not None:
            raise ValueError('density and count modes are mutually exclusive')
    else:
        raise ValueError('capacity.mode must be density or count')
    number(cap['initial_ratio'],'capacity.initial_ratio',1e-12,1)
    snapshots = cfg['snapshots']
    counts = snapshots['counts']
    if not isinstance(counts,list):
        raise ValueError('snapshots.counts must be an increasing integer list')
    for count in counts:
        number(count,'snapshots.counts',2,HARD_CAP,integer=True)
    if counts != sorted(set(counts)):
        raise ValueError('snapshots.counts must be unique and increasing')
    number(snapshots['tolerance'],'snapshots.tolerance',0,1)
    growth = cfg['growth']
    if not isinstance(growth['enabled'],bool):
        raise ValueError('growth.enabled must be boolean')
    number(growth['fraction'],'growth.fraction',1e-12,1-1e-12)
    number(growth['events'],'growth.events',1,integer=True)
    number(growth['min_birth_ratio'],'growth.min_birth_ratio',1e-12,1)
    number(growth['max_birth_ratio'],'growth.max_birth_ratio',growth['min_birth_ratio'],1)
    es = cfg['early_stop']
    if not isinstance(es['enabled'],bool):
        raise ValueError('early_stop.enabled must be boolean')
    for key, lower in [('patience',1),('eval_interval',1),('min_fixed_steps',0)]:
        number(es[key],'early_stop.'+key,lower,integer=True)
    for key in ('min_psnr_gain','min_ssim_gain'):
        number(es[key],'early_stop.'+key,1e-12)
    adv = cfg['advanced']
    for key in ('mse_fraction','ssim_ramp_fraction'):
        number(adv[key],'advanced.'+key,0,1)
    if adv['mse_fraction']+adv['ssim_ramp_fraction'] > 1:
        raise ValueError('MSE and SSIM ramp fractions exceed the total budget')
    for key in ('ssim_interval','evaluate_every','threads','ssim_tile','dark_grid'):
        number(adv[key],'advanced.'+key,1,integer=True)
    for key in ('initial_cg','new_color_cg','warm_min','warm_max','joint_steps'):
        number(adv[key],'advanced.'+key,1,integer=True)
    if adv['warm_max'] < adv['warm_min']:
        raise ValueError('warm_max must be >= warm_min')
    for key in ('ssim_weight','dark_leak_weight','regularizer_weight'):
        number(adv[key],'advanced.'+key,0)
    for key in ('dark_threshold','dark_margin'):
        number(adv[key],'advanced.'+key,0,1)
    if adv['growth_reference'] not in ('pre_birth','continued_fit'):
        raise ValueError('Invalid growth_reference')
    if adv['checkpoint_selection'] not in ('within_cap','same_count'):
        raise ValueError('Invalid checkpoint_selection')
    options = adv['options']
    if options['pixel_samples'] not in (1,2,4) or not isinstance(options['pixel_samples'],int) or isinstance(options['pixel_samples'],bool):
        raise ValueError('pixel_samples must be 1, 2 or 4')
    for key in options:
        if key != 'pixel_samples' and not isinstance(options[key],bool):
            raise ValueError('advanced.options.'+key+' must be boolean')
    # Split/merge would violate the promised structurally fixed final stage.
    if options['split_merge']:
        raise ValueError('split_merge is disabled in this two-stage API; use a separate experimental recipe')
    opt = adv['optimizer']
    for key, value in opt['rates'].items():
        number(value,'optimizer.rates.'+key,1e-12)
    if not isinstance(opt['betas'],list) or len(opt['betas']) != 3:
        raise ValueError('optimizer.betas requires three numbers')
    for value in opt['betas']:
        number(value,'optimizer.betas',0,1-1e-12)
    number(opt['eps'],'optimizer.eps',1e-20)
    if not isinstance(adv['renderer_sha256'],str) or len(adv['renderer_sha256']) != 64 or any(c not in '0123456789abcdef' for c in adv['renderer_sha256']):
        raise ValueError('renderer_sha256 must contain 64 lowercase hexadecimal characters')
    if not isinstance(cfg['preprocess']['smoke_crop'],bool):
        raise ValueError('smoke_crop must be boolean')
    return cfg


def load_config(path=None, overrides=None, smoke=False):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        path = Path(path).expanduser().resolve(strict=True)
        updates = json.loads(path.read_text(encoding='utf-8-sig'))
        # Specifying count selects that mode without inheriting default density.
        if updates.get('capacity',{}).get('mode') == 'count':
            cfg['capacity']['density'] = None
        merge_strict(cfg,updates)
        for key in ('input','output'):
            if cfg[key] is not None:
                if not isinstance(cfg[key],str):
                    raise ValueError(key+' must be a path string or null')
                value = Path(cfg[key]).expanduser()
                cfg[key] = str((path.parent/value).resolve()) if not value.is_absolute() else str(value)
    overrides = overrides or {}
    if overrides.get('capacity',{}).get('mode') == 'count':
        cfg['capacity']['density'] = None
    if overrides.get('capacity',{}).get('mode') == 'density':
        cfg['capacity']['count'] = None
    merge_strict(cfg,overrides)
    if smoke:
        cfg['total_steps'] = 12
        cfg['capacity'] = {'mode':'count','density':None,'count':64,'initial_ratio':.5}
        cfg['growth'].update(fraction=.75,events=2,min_birth_ratio=.125,max_birth_ratio=.5)
        cfg['early_stop']['enabled'] = False
        cfg['preprocess']['smoke_crop'] = True
        cfg['snapshots']['counts'] = []  # Production milestones do not apply to the 64-point smoke profile.
        cfg['advanced'].update(mse_fraction=0,ssim_ramp_fraction=.1,initial_cg=2,new_color_cg=2,
                               warm_min=2,warm_max=2,joint_steps=2,evaluate_every=4,ssim_tile=64)
    if cfg['advanced']['renderer_sha256'] == '0'*64:
        build = ROOT/'native/renderer.json'
        if build.is_file():
            cfg['advanced']['renderer_sha256'] = json.loads(build.read_text(encoding='utf-8'))['sha256']
    return validate(cfg)


def rounded(value):
    return int(math.floor(value+.5))


def resolve_plan(cfg, input_path=None):
    value = input_path or cfg['input']
    if not value:
        raise ValueError('Supply --input or input in the configuration')
    path = Path(value).expanduser().resolve(strict=True)
    with Image.open(path) as im:
        w,h = im.size
        source_mode = im.mode
    source_size = [w,h]
    if cfg['preprocess']['smoke_crop']:
        w,h = min(w,224),min(h,192)
    if min(w,h) <= 160:
        raise ValueError('Both image dimensions must exceed 160 for the fixed 5-level MS-SSIM definition; input is not resized implicitly')
    pixels = w*h
    cap = cfg['capacity']
    count = rounded(cap['density']*pixels) if cap['mode']=='density' else cap['count']
    if not 2 <= count <= HARD_CAP:
        raise ValueError('Resolved capacity must be 2..200000; lower density/count to respect the hard budget')
    initial = min(count,max(2,rounded(cap['initial_ratio']*count))) if cfg['growth']['enabled'] else count
    counts = cfg['snapshots']['counts']
    if counts and counts[-1] > count:
        raise ValueError('Snapshot targets exceed the requested cap; raise --num or lower --save-counts')
    if counts and initial > counts[0]:
        raise ValueError('Initial count exceeds the first snapshot target; lower --initial-ratio')
    detail = initial-round(.65*initial)
    if detail > math.ceil(h/4)*math.ceil(w/4):
        raise ValueError('Initial detail sampling exceeds unique 4x4 cells; lower initial_ratio/count or use density mode')
    total = cfg['total_steps']
    end = min(total-1,max(1,rounded(total*cfg['growth']['fraction']))) if cfg['growth']['enabled'] else 0
    mse = rounded(total*cfg['advanced']['mse_fraction'])
    ramp = rounded(total*cfg['advanced']['ssim_ramp_fraction'])
    events = []
    if cfg['growth']['enabled'] and initial < count:
        available = end-mse
        if available <= 0:
            raise ValueError('MSE-only stage leaves no growth opportunity before the fixed stage')
        k = min(cfg['growth']['events'],available)
        events = sorted(set(mse+max(1,rounded(i*available/k)) for i in range(1,k+1)))
    min_birth = min(count,max(1,math.ceil(cfg['growth']['min_birth_ratio']*count)))
    max_birth = min(count,max(min_birth,math.ceil(cfg['growth']['max_birth_ratio']*count)))
    quantum = min(256,max(1,min_birth//4))
    plan = dict(input=str(path),source_mode=source_mode,source_size=source_size,width=w,height=h,pixels=pixels,
                requested_cap=count,requested_density=cap['density'] if cap['mode']=='density' else None,
                capacity_mode=cap['mode'],resolved_density=count/pixels,primitives_per_megapixel=count/pixels*1e6,
                initial_N=initial,initial_ratio=initial/count,total_steps=total,
                growth_end_step=end,fixed_start_step=end,growth_steps=events,
                mse_steps=mse,ssim_ramp_steps=ramp,min_birth=min_birth,max_birth=max_birth,birth_quantum=quantum,
                fixed_stage_steps=total-end,hard_capacity_limit=HARD_CAP)
    return plan


def recipe_fingerprint(cfg,plan):
    values = copy.deepcopy(cfg)
    values.pop('input');values.pop('output')
    dims = {k:plan[k] for k in ('width','height','requested_cap','initial_N','growth_steps')}
    return hashlib.sha256(json.dumps(dict(config=values,plan=dims),sort_keys=True).encode('utf-8')).hexdigest()
