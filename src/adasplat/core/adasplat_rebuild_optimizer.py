"""Row-wise Adan clocks: growth preserves old moments and starts new rows at zero."""
import torch

FIELDS=('xys','scale_raw','rotation_angle','colors','p')
DEFAULT_RATES=dict(xys=.03,scale_raw=.01,rotation_angle=.003,colors=.025,p=.003)

class CohortAdan:
    def __init__(self,model,rates=None,betas=(.98,.92,.99),eps=1e-8):
        self.model=model;self.rates=dict(rates or DEFAULT_RATES)
        self.betas=betas;self.eps=eps;self.state={}
        self.birth_step=torch.zeros(model.count,device=model.xys.device,dtype=torch.long)
        for name in FIELDS:
            parameter=getattr(model,name)
            self.state[name]=dict(step=torch.zeros(len(parameter),device=parameter.device,dtype=torch.long),
                m=torch.zeros_like(parameter),d=torch.zeros_like(parameter),v=torch.zeros_like(parameter),previous=torch.zeros_like(parameter))
    def zero_grad(self):self.model.zero_grad(set_to_none=True)
    @torch.no_grad()
    def step(self,factor=1.,rows=None,geometry=True):
        b1,b2,b3=self.betas
        for name in FIELDS:
            if not geometry and name!='colors':continue
            param=getattr(self.model,name)
            if param.grad is None:continue
            state=self.state[name]
            index=torch.arange(len(param),device=param.device) if rows is None else rows
            grad=param.grad[index]
            clock=state['step'][index];previous=state['previous'][index]
            broadcast=(len(index),)+(1,)*(param.ndim-1)
            difference=grad-previous
            difference=torch.where((clock==0).reshape(broadcast),torch.zeros_like(difference),difference)
            m=state['m'][index]*b1+grad*(1-b1)
            d=state['d'][index]*b2+difference*(1-b2)
            v=state['v'][index]*b3+(grad+b2*difference).square()*(1-b3)
            clock=clock+1
            bc1=(1-b1**clock).reshape(broadcast);bc2=(1-b2**clock).reshape(broadcast)
            bc3=(1-b3**clock).sqrt().reshape(broadcast)
            update=(m/bc1+b2*d/bc2)/(v.sqrt()/bc3+self.eps)
            if name!='colors':
                # Initial atoms keep the reference schedule. New geometry
                # ramps over its own first 256 updates after color warmup.
                ramp=(clock.float()/256).clamp(.05,1.)
                ramp=torch.where(self.birth_step[index]>0,ramp,torch.ones_like(ramp))
                update*=ramp.reshape(broadcast)
            param[index]=param[index]-self.rates[name]*factor*update
            for key,value in [('m',m),('d',d),('v',v),('previous',grad),('step',clock)]:state[key][index]=value
    def state_dict(self):
        return dict(rates=self.rates,betas=self.betas,eps=self.eps,birth_step=self.birth_step.clone(),
                    state={n:{k:v.detach().clone() for k,v in s.items()} for n,s in self.state.items()})
    def load_state_dict(self,data):
        self.rates=dict(data['rates']);self.betas=tuple(data['betas']);self.eps=data['eps']
        if 'birth_step' in data:self.birth_step.copy_(data['birth_step'])
        for name,states in data['state'].items():
            for key,value in states.items():
                if value.shape!=self.state[name][key].shape:raise ValueError('optimizer shape mismatch')
                self.state[name][key].copy_(value)
    def transport(self,new_model,old_rows,birth_step=1):
        """old_rows[i] is the old row, or -1 for a newborn/replaced atom."""
        if len(old_rows)!=new_model.count:raise ValueError('row mapping mismatch')
        other=CohortAdan(new_model,self.rates,self.betas,self.eps)
        valid=old_rows>=0
        other.birth_step.fill_(birth_step);other.birth_step[valid]=self.birth_step[old_rows[valid]]
        for name in FIELDS:
            for key,value in self.state[name].items():other.state[name][key][valid]=value[old_rows[valid]]
        return other

class SensitivityTrust:
    """Refresh a sampled local Jacobian; bound actual optimizer shape updates.

    This is a local response estimate, not a full-image Fisher matrix. Every
    component and aggregate response is bounded without resetting Adan state.
    """
    def __init__(self,model,response_limit=.08):
        self.model=model;self.response_limit=response_limit;self.sensitivity=None
    @torch.no_grad()
    def refresh(self):
        from adasplat_rebuild_model import patch_basis
        from primix_v4_2_parameterization import inverse_softplus
        model=self.model;device=model.xys.device;n=model.count
        relative=model.xys.new_tensor([[0.,0.],[.3,.2],[-.3,.2],[.3,-.2],[-.3,-.2],[1.,.4],[-1.,.4],[.4,1.],[.4,-1.],[2.,.7],[-2.,.7],[.7,2.],[.7,-2.]])
        # C1 moments give the actual continuous RMS; nominal is a conservative
        # patch scale on the older uncorrected coordinate.
        grid=model.xys[:,None,:]+relative[None]*model.nominal_scales[:,None,:].clamp_min(.5)
        old={name:getattr(model,name).detach().clone() for name in ('xys','scale_raw','rotation_angle','p')}
        def response(values):
            p=values['p'];sc=(torch.nn.functional.softplus(values['scale_raw'])+.25)*model.q(p)[:,None]
            basis=patch_basis(values['xys'],sc,values['rotation_angle'],p,grid,model.options)
            return torch.einsum('npb,nbc->npc',basis,model.colors.reshape(n,-1,3))
        before=response(old);self.reference=before.square().mean((1,2)).sqrt().clamp_min(.001)
        sensitivities=[]
        for name,index,h in [('xys',0,.002),('xys',1,.002),('scale_raw',0,.002),('scale_raw',1,.002),('rotation_angle',None,.001),('p',None,.0005)]:
            plus={k:v.clone() for k,v in old.items()};minus={k:v.clone() for k,v in old.items()}
            if index is None:plus[name].add_(h);minus[name].sub_(h)
            else:plus[name][:,index].add_(h);minus[name][:,index].sub_(h)
            if name=='p':plus[name].clamp_(.1,8);minus[name].clamp_(.1,8)
            denom=plus[name]-minus[name] if index is None else plus[name][:,index]-minus[name][:,index]
            derivative=(response(plus)-response(minus))/denom.clamp_min(1e-8)[:,None,None]
            sensitivities.append(derivative.square().mean((1,2)).sqrt())
        self.sensitivity=torch.stack(sensitivities,1)
    @torch.no_grad()
    def snapshot(self):
        return {name:getattr(self.model,name).detach().clone() for name in ('xys','scale_raw','rotation_angle','p')}
    @torch.no_grad()
    def constrain(self,before):
        model=self.model
        delta=torch.cat([model.xys-before['xys'],model.scale_raw-before['scale_raw'],
                         (model.rotation_angle-before['rotation_angle'])[:,None],(model.p-before['p'])[:,None]],1)
        estimated=(delta.abs()*self.sensitivity).sum(1)
        ratio=(self.response_limit*self.reference/estimated.clamp_min(1e-12)).clamp(max=1)
        minor=model.nominal_scales.amin(1).clamp_min(.5)
        ratio=torch.minimum(ratio,(.15*minor/delta[:,:2].norm(dim=1).clamp_min(1e-12)).clamp(max=1))
        ratio=torch.minimum(ratio,(.02/delta[:,-1].abs().clamp_min(1e-12)).clamp(max=1))
        for name,value in before.items():
            param=getattr(model,name);weight=ratio if param.ndim==1 else ratio[:,None]
            param.copy_(value+(param-value)*weight)
        return dict(limited_rows=int((ratio<.999).sum()),mean_trust_factor=float(ratio.mean()))
