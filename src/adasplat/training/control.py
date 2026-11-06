"""Convergence is observed only after the predetermined growth deadline."""
import copy
import math


class FixedStageEarlyStop:
    def __init__(self, config, fixed_start):
        self.config = copy.deepcopy(config)
        self.fixed_start = fixed_start
        self.anchor_psnr = None
        self.anchor_ssim = None
        self.bad_evaluations = 0
        self.last_step = None

    def observe(self, step, metrics):
        if not self.config['enabled'] or step < self.fixed_start:
            return False
        if self.last_step is not None and step <= self.last_step:
            return False
        p,s = metrics['PSNR'],metrics['SSIM']
        if not math.isfinite(p) or not math.isfinite(s):
            raise ValueError('Nonfinite early-stop metrics')
        self.last_step = step
        if self.anchor_psnr is None:
            self.anchor_psnr,self.anchor_ssim = p,s
            return False
        # Small cumulative improvements can reach the threshold across evaluations.
        significant = p-self.anchor_psnr >= self.config['min_psnr_gain'] or s-self.anchor_ssim >= self.config['min_ssim_gain']
        if significant:
            self.anchor_psnr = max(p,self.anchor_psnr)
            self.anchor_ssim = max(s,self.anchor_ssim)
            self.bad_evaluations = 0
        elif step-self.fixed_start >= self.config['min_fixed_steps']:
            self.bad_evaluations += 1
        else:
            self.bad_evaluations = 0
        return step-self.fixed_start >= self.config['min_fixed_steps'] and self.bad_evaluations >= self.config['patience']

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        if state['config'] != self.config or state['fixed_start'] != self.fixed_start:
            raise ValueError('Early-stop recipe mismatch')
        self.__dict__.update(copy.deepcopy(state))


class RelativeBirthBudget:
    def __init__(self, plan):
        self.cap = plan['requested_cap']
        self.opportunities = list(plan['growth_steps'])
        self.min_birth = plan['min_birth']
        self.max_birth = plan['max_birth']
        self.quantum = plan['birth_quantum']
        self.history = []

    def request(self, step, count):
        if step not in self.opportunities or count >= self.cap:
            return 0
        opportunities = sum(s >= step for s in self.opportunities)
        recent = self.history[-8:]
        rate = max(.5,min(1.,(4+sum(r['accepted'] for r in recent))/(5+len(recent))))
        rounds = max(1,opportunities-2)
        wanted = math.ceil((self.cap-count)/(rounds*rate))
        wanted = math.ceil(wanted/self.quantum)*self.quantum
        return min(self.cap-count,self.max_birth,max(self.min_birth,wanted))

    def update(self, step, accepted, count):
        self.history.append(dict(step=step,accepted=bool(accepted),count=count))

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        for key in ('cap','opportunities','min_birth','max_birth','quantum'):
            if state[key] != getattr(self,key):
                raise ValueError('Birth budget recipe mismatch')
        self.history = copy.deepcopy(state['history'])
