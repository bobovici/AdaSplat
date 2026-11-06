"""Exact coordinate conversion and fixed-cardinality primitive replacement."""
import torch
import torch.nn.functional as F
from primix_v4_2_parameterization import q_for,inverse_softplus,exact_l2_norm
FIELDS=('xys','scale_raw','rotation_angle','mass','colors','p')

@torch.no_grad()
def convert_effective_to_continuous_free(state):
    result={k:v.detach().clone() for k,v in state.items()}
    physical=(F.softplus(state['scale_raw'])+.25)*q_for('effective_support',state['p'])[:,None]
    nominal=physical/q_for('continuous_l2_rms_free_scale',state['p'])[:,None]
    result['scale_raw']=inverse_softplus(nominal)
    return result

@torch.no_grad()
def utility_order(model):
    # Infinite-domain RGB energy proxy used only to rank replacement proposals.
    # Native-render quality checks decide whether replacing these slots is safe.
    energy=model.mass.square()*model.colors.square().sum(1)
    return torch.argsort(energy)

@torch.no_grad()
def replace_slots(state,slots,born):
    if len(torch.unique(slots))!=len(slots):raise ValueError('duplicate slots')
    result={k:v.detach().clone() for k,v in state.items()}
    for k in FIELDS:
        if len(born[k])!=len(slots):raise ValueError('replacement size mismatch')
        result[k][slots]=born[k]
        if result[k].shape!=state[k].shape:raise ValueError('fixed budget violated')
    return result

@torch.no_grad()
def proposed_born_state(prediction,target,k,seed,scale_mode='continuous_l2_rms_free_scale'):
    from adasplat_residual_growth import residual_birth_state
    empty={k:prediction.new_empty((0,3) if k=='colors' else (0,2) if k in ['xys','scale_raw'] else (0,)) for k in FIELDS}
    born,info=residual_birth_state(empty,prediction,target,k,seed)
    # residual_birth_state creates the historical +.25 nominal coordinate.
    # p=2 makes q=1; convert newborn widths to the free scale coordinate.
    if scale_mode=='continuous_l2_rms_free_scale':
        physical=F.softplus(born['scale_raw'])+.25
        born['scale_raw']=inverse_softplus(physical)
    return born,info
