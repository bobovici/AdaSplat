"""Structure-aware births and paired split/merge, with local candidate pricing.

Local previews rank proposals; only whole-image A/B training can accept them.
No dense full-image Jacobian is formed. Primitive budget is asserted at build.
"""
import math
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from adasplat_rebuild_model import RebuildModel,patch_basis
from adasplat_full_image_objective import full_image_loss_gradient
from primix_v4_2_parameterization import inverse_softplus

def local_grid(xy,h,w,radius=16):
    x=int(xy[0].round());y=int(xy[1].round())
    xs=torch.arange(max(0,x-radius),min(w,x+radius+1),device=xy.device)
    ys=torch.arange(max(0,y-radius),min(h,y+radius+1),device=xy.device)
    gy,gx=torch.meshgrid(ys,xs,indexing='ij')
    return torch.stack([gx.flatten(),gy.flatten()],1).float(),gy.flatten(),gx.flatten()

def tangent_projection(model,grid,indices,direction):
    """Project a P x 3 direction away from nearby RGB/shape tangent columns."""
    if not len(indices):return direction,1.
    n=len(indices);xy=model.xys[indices].detach();sc=model.scales[indices].detach()
    angle=model.rotation_angle[indices].detach();p=model.p[indices].detach()
    colors=model.colors[indices].detach().reshape(n,-1,3)
    basis=patch_basis(xy,sc,angle,p,grid,model.options)
    columns=[]
    for i in range(n):
        for b in range(basis.shape[-1]):
            for channel in range(3):
                value=grid.new_zeros(len(grid),3);value[:,channel]=basis[i,:,b];columns.append(value.flatten())
    # Finite differences of the exact local native-shaped kernel. Physical
    # scale derivatives plus p span the same space as nominal-scale plus p.
    for name,index,h in [('xy',0,.002),('xy',1,.002),('sc',0,.001),('sc',1,.001),('angle',None,.001),('p',None,.0005)]:
        plus=dict(xy=xy.clone(),sc=sc.clone(),angle=angle.clone(),p=p.clone())
        minus={k:v.clone() for k,v in plus.items()}
        if index is None:plus[name]+=h;minus[name]-=h
        elif name=='sc':plus[name][:,index]*=math.exp(h);minus[name][:,index]*=math.exp(-h)
        else:plus[name][:,index]+=h;minus[name][:,index]-=h
        plus['p'].clamp_(.1,8);minus['p'].clamp_(.1,8)
        a=patch_basis(plus['xy'],plus['sc'],plus['angle'],plus['p'],grid,model.options)
        b=patch_basis(minus['xy'],minus['sc'],minus['angle'],minus['p'],grid,model.options)
        derivative=torch.einsum('npb,nbc->npc',(a-b)/(2*h),colors)
        columns.extend([d.flatten() for d in derivative])
    jacobian=torch.stack(columns,1)
    norms=jacobian.norm(dim=0);jacobian=jacobian[:,norms>1e-8]/norms[norms>1e-8]
    batched=direction.ndim==3
    vector=direction.reshape(-1,len(grid)*3).T if batched else direction.flatten()
    if jacobian.shape[1]==0:return direction,1.
    gram=jacobian.T@jacobian+torch.eye(jacobian.shape[1],device=grid.device)*1e-4
    residual=vector-jacobian@torch.linalg.solve(gram,jacobian.T@vector)
    if batched:
        fraction=residual.square().sum(0)/vector.square().sum(0).clamp_min(1e-20)
        return residual.T.reshape_as(direction),fraction
    fraction=float(residual.square().sum()/vector.square().sum().clamp_min(1e-20))
    return residual.reshape_as(direction),fraction

@torch.no_grad()
def propose_births(model,image,target,count,ssim_weight=.035,seed=20261002,prior=None):
    if model.count+count>model.options.cap:raise ValueError('growth exceeds primitive budget')
    difference=(image-target)[0].permute(1,2,0)
    if model.options.structure_birth:
        # SSIM's tiled VJP needs its own autograd graph despite proposal no_grad.
        with torch.enable_grad():_,gradient,_=full_image_loss_gradient(image,target,ssim_weight=ssim_weight,scale=1.,tile=1024)
        gradient=gradient[0].permute(1,2,0)
    else:gradient=2*difference/image.numel()
    energy=gradient.square().sum(-1)
    scoremap=F.avg_pool2d(energy[None,None],4,4,ceil_mode=True)[0,0]
    if prior is not None and prior.shape==scoremap.shape:
        scoremap=(scoremap*prior.clamp_min(1e-30)).sqrt()
    available=int((scoremap>0).sum());pool=min(available,max(count*2,count+16))
    if pool==0:return None,dict(reason='zero image gradient'),scoremap
    generator=torch.Generator(device=model.xys.device).manual_seed(seed)
    sampled=torch.multinomial(scoremap.flatten().clamp_min(1e-30),pool,replacement=False,generator=generator)
    sampled=sampled[torch.argsort(scoremap.flatten()[sampled],descending=True)]
    cw=scoremap.shape[1];h,w=target.shape[-2:]
    gray=target.mean(1)[0]
    dx=F.pad(gray[:,1:]-gray[:,:-1],(0,1));dy=F.pad(gray[1:]-gray[:-1],(0,0,0,1))
    tree=cKDTree(model.xys.detach().cpu().numpy()) if model.options.tangent_novelty else None
    selected=[];coefficients=[];records=[]
    # A position competes over p, scale, anisotropy and edge orientation.
    for cell in sampled:
        cy,cx=int(cell//cw)*4,int(cell%cw)*4
        patch=energy[cy:min(cy+4,h),cx:min(cx+4,w)];peak=int(patch.argmax());pw=patch.shape[1]
        x,y=cx+peak%pw,cy+peak//pw
        xy=model.xys.new_tensor([x,y]);grid,gy,gx=local_grid(xy,h,w)
        tangent=float(torch.atan2(dy[y,x],dx[y,x])+math.pi/2)
        shapes=[]
        ps=[.5,1.,2.,4.,8.] if model.options.shape_competition else [2.]
        for p in ps:
            for size in [.8,1.5,3.,6.]:
                shapes.append([size,size,0.,p])
                if model.options.shape_competition:
                    shapes.append([size*1.7,max(.3,size*.55),tangent,p])
                    shapes.append([size*1.7,max(.3,size*.55),tangent+math.pi/2,p])
        specs=xy.new_tensor(shapes);p=specs[:,3];nominal=specs[:,:2]
        sc=nominal*model.q(p)[:,None];angle=specs[:,2];xys=xy[None].expand(len(shapes),-1)
        basis=patch_basis(xys,sc,angle,p,grid,model.options)
        local_g=gradient[gy,gx].clone();residual=-difference[gy,gx].clone()
        # Sequentially subtract already proposed local responses. Correlation
        # tests use the same sampled kernel rather than distance alone.
        previous_basis=None
        if selected:
            previous=torch.stack(selected)
            nearby=(previous[:,:2]-xy).norm(dim=1)<48
            if bool(nearby.any()):
                rows=previous[nearby];pb=patch_basis(rows[:,:2],rows[:,2:4]*model.q(rows[:,5])[:,None],rows[:,4],rows[:,5],grid,model.options)
                co=torch.stack(coefficients)[nearby]
                effect=torch.einsum('npb,nbc->pc',pb,co)
                residual-=effect;local_g+=2*effect/image.numel();previous_basis=pb[...,0]
        rhs=-torch.einsum('npb,pc->nbc',basis,local_g)*image.numel()/2
        gram=torch.einsum('npb,npd->nbd',basis,basis)
        ridge=torch.eye(basis.shape[-1],device=xy.device)[None]*1e-5
        co=torch.linalg.solve(gram+ridge,rhs)
        effect=torch.einsum('npb,nbc->npc',basis,co)
        gains=-(effect*local_g[None]).sum((1,2))-effect.square().sum((1,2))/image.numel()
        if previous_basis is not None:
            candidate=basis[...,0]
            corr=(candidate@previous_basis.T)/(candidate.norm(dim=1)[:,None]*previous_basis.norm(dim=1)[None]).clamp_min(1e-20)
            gains[corr.abs().amax(1)>.96]=-float('inf')
        order=torch.argsort(gains,descending=True)[:4];choice=None;novelty=1.
        order=order[gains[order]>0]
        if len(order) and tree is not None:
            _,neighbors=tree.query(xy.cpu().numpy(),k=min(8,model.count),distance_upper_bound=24)
            neighbors=[int(v) for v in torch.as_tensor(neighbors).flatten() if int(v)<model.count]
            _,fractions=tangent_projection(model,grid,torch.tensor(neighbors,device=xy.device),effect[order])
            if isinstance(fractions,float):fractions=gains.new_full((len(order),),fractions)
            prices=gains[order]*fractions;prices[fractions<.05]=-float('inf')
            if float(prices.max())>0:
                j=int(prices.argmax());choice=int(order[j]);novelty=float(fractions[j])
        elif len(order):choice=int(order[0])
        if choice is None:continue
        spec=specs[choice];selected.append(torch.cat([xy,spec[:2],spec[2:]]));coefficients.append(co[choice])
        records.append(dict(x=x,y=y,nominal_scale=spec[:2].tolist(),angle=float(spec[2]),p=float(spec[3]),
                            predicted_objective_gain=float(gains[choice])*novelty,novelty_fraction=novelty))
        if len(selected)==count:break
    if not selected:return None,dict(reason='no positive novel local direction',candidate_cells=pool),scoremap
    rows=torch.stack(selected);k=len(rows)
    born=dict(xys=rows[:,:2],scale_raw=inverse_softplus(rows[:,2:4]-.25),rotation_angle=rows[:,4],p=rows[:,5],
              colors=model.colors.new_zeros(k,model.colors.shape[1]),mass=model.mass.new_ones(k))
    state={name:torch.cat([value.detach(),born[name]],0) for name,value in model.state_dict().items()}
    trial=RebuildModel(target,model.count+k,model.options);trial.load_state_dict(state)
    mapping=torch.cat([torch.arange(model.count,device=rows.device),torch.full((k,),-1,device=rows.device)])
    return (trial,mapping,torch.arange(model.count,model.count+k,device=rows.device)),dict(kind='growth',count=k,requested=count,
        candidate_cells=pool,records=records,predicted_objective_gain=sum(v['predicted_objective_gain'] for v in records)),scoremap

@torch.no_grad()
def propose_split_merge(model,image,target,max_pairs=16):
    """A 2-to-1 merge buys a slot for 1-to-2 splitting: N stays unchanged."""
    tree=cKDTree(model.xys.detach().cpu().numpy());_,neighbors=tree.query(tree.data,k=2)
    energy=(image-target).square().mean(1)[0]
    x=model.xys[:,0].round().long();y=model.xys[:,1].round().long()
    priorities=energy[y,x]*model.nominal_scales.amax(1).square()
    parents=torch.argsort(priorities,descending=True).tolist()
    state={k:v.detach().clone() for k,v in model.state_dict().items()};used=set();pairs=[]
    # Only nearby shape-similar pairs are considered; actual sampled correlation
    # and a local least-squares merge decide eligibility.
    nominal=model.nominal_scales;physical=model.scales
    distances=torch.as_tensor(tree.data-tree.data[neighbors[:,1]],device=model.xys.device).norm(dim=1)
    candidates=torch.argsort(distances/nominal.amin(1).clamp_min(.5))[:4096].tolist()
    for i in candidates:
        j=int(neighbors[i,1])
        if i in used or j in used:continue
        if float(distances[i])>2*float(nominal[[i,j]].amin()):continue
        xy=(model.xys[i]+model.xys[j])/2;grid,_,_=local_grid(xy,model.height,model.width)
        indices=torch.tensor([i,j],device=xy.device)
        basis=patch_basis(model.xys[indices],physical[indices],model.rotation_angle[indices],model.p[indices],grid,model.options)
        a,b=basis[...,0];corr=float((a@b)/(a.norm()*b.norm()).clamp_min(1e-20))
        if corr<.95:continue
        parent=next((v for v in parents if v not in used and v not in (i,j)),None)
        if parent is None:break
        merged=patch_basis(xy[None],physical[i:i+1],model.rotation_angle[i:i+1],model.p[i:i+1],grid,model.options)[0]
        old=torch.einsum('npb,nbc->pc',basis,model.colors[indices].reshape(2,-1,3))
        coef=torch.linalg.solve(merged.T@merged+torch.eye(merged.shape[1],device=xy.device)*1e-5,merged.T@old)
        relative=float((merged@coef-old).square().sum()/old.square().sum().clamp_min(1e-20))
        if relative>.005:continue
        state['xys'][i]=xy;state['colors'][i]=coef.flatten()
        for key in state:state[key][j]=model.state_dict()[key][parent]
        state['colors'][j]*=.5;state['colors'][parent]*=.5
        # Controlled symmetry breaking of a function-preserving duplicate.
        theta=model.rotation_angle[parent];shift=.25*physical[parent,0]
        offset=torch.stack([theta.cos(),theta.sin()])*shift.clamp(.05,2.)
        state['xys'][j]+=offset;state['xys'][parent]-=offset
        used.update((i,j,parent));pairs.append(dict(merge=[i,j],split_parent=parent,correlation=corr,local_merge_relative_error=relative))
        if len(pairs)==max_pairs:break
    if not pairs:return None,dict(kind='split_merge',reason='no low-loss merge available')
    trial=RebuildModel(target,model.count,model.options);trial.load_state_dict(state);trial.project_parameters_()
    rows=torch.tensor(sorted(used),device=model.xys.device);mapping=torch.arange(model.count,device=rows.device);mapping[rows]=-1
    return (trial,mapping,rows),dict(kind='split_merge',count=len(pairs),changed_rows=len(rows),pairs=pairs,
                                    net_added=0,N_before=model.count,N_after=trial.count)

class MarginalBatch:
    def __init__(self,k=512,min_k=128,max_k=2048):self.k=k;self.min_k=min_k;self.max_k=max_k;self.history=[]
    def update(self,accepted,before,after,k,predicted,ssim_weight=.035):
        gain=before['MSE']-after['MSE'];per1000=gain*1000/max(k,1)
        objective_gain=gain+ssim_weight*(after['SSIM']-before['SSIM'])
        ratio=objective_gain/max(predicted,1e-20)
        if not accepted:self.k=max(self.min_k,self.k//2)
        elif ratio>.5:self.k=min(self.max_k,self.k*2)
        self.history.append(dict(accepted=accepted,count=k,mse_gain_per_1000=per1000,
            ssim_gain_per_1000=(after['SSIM']-before['SSIM'])*1000/max(k,1),actual_to_predicted_objective_ratio=ratio,next_k=self.k))

class DeadlineBatch:
    """Plan capacity use by remaining opportunities, independent of local gain bias."""
    def __init__(self,cap=114567,deadline=32000,interval=500,min_k=2048,max_k=8192,quantum=256):
        if not (1<=min_k<=max_k<=cap) or interval<1 or quantum<1:raise ValueError('invalid growth schedule')
        self.policy='deadline_v1';self.cap=cap;self.deadline=deadline;self.interval=interval
        self.min_k=min_k;self.max_k=max_k;self.quantum=quantum;self.k=min_k;self.history=[]
    def request(self,step,count):
        remaining=self.cap-count
        if remaining<=0 or step>self.deadline:self.k=0;return 0
        opportunities=(self.deadline-step)//self.interval+1
        rounds=max(1,opportunities-2)  # reserve two attempts for late rejections
        recent=self.history[-8:]
        rate=max(.5,min(1.,(4+sum(r['accepted'] for r in recent))/(5+len(recent))))
        required=math.ceil(remaining/(rounds*rate))
        rounded=math.ceil(required/self.quantum)*self.quantum
        self.k=min(remaining,self.max_k,max(self.min_k,rounded))
        self.remaining_capacity=remaining;self.remaining_opportunities=opportunities
        self.planned_opportunities=rounds;self.estimated_acceptance_rate=rate
        return self.k
    def update(self,accepted,before,after,k,predicted,ssim_weight=.035):
        gain=before['MSE']-after['MSE'];objective=gain+ssim_weight*(after['SSIM']-before['SSIM'])
        self.history.append(dict(accepted=bool(accepted),count=k,mse_gain_per_1000=gain*1000/max(1,k),
            ssim_gain_per_1000=(after['SSIM']-before['SSIM'])*1000/max(1,k),
            actual_to_predicted_objective_ratio=objective/max(predicted,1e-20)))
