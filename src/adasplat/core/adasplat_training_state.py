"""Phase 0 complete registered active-state snapshots.

Includes single-writer persisted trials; no topology policy or checkpoint conversion.
The caller supplies a fresh runtime factory; publication is a single assignment
of the returned runtime. Unregistered arbitrary Python objects are rejected.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
import random
import struct
import json
import os
from pathlib import Path
import uuid

import numpy as np
import torch


SCHEMA = 'adasplat_training_state_v1'
REQUIRED_METADATA = frozenset(('clock', 'cohorts', 'population', 'residual',
    'proposal', 'controller', 'utility', 'selected_accepted_topology',
    'renderer_runtime', 'metric_runtime'))


def clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError('unregistered state value: %s' % type(value).__name__)


def canonical_hash(value):
    """Typed, framed content hash independent of torch.save/pickle bytes."""
    digest = hashlib.sha256()
    def frame(tag, payload=b''):
        digest.update(tag.encode('ascii') + b':' + str(len(payload)).encode('ascii') + b':' + payload)
    def visit(v):
        if torch.is_tensor(v):
            if v.layout != torch.strided:
                raise TypeError('only dense tensor state is registered')
            t = v.detach().cpu().contiguous()
            frame('tensor', str(t.dtype).encode())
            visit(tuple(t.shape))
            frame('data', t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise TypeError('object arrays are not registered state')
            frame('ndarray', v.dtype.str.encode()); visit(tuple(v.shape))
            frame('data', np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            if not all(isinstance(k, (str, int)) for k in v):
                raise TypeError('state keys must be strings or integers')
            frame('dict', str(len(v)).encode())
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            frame('list' if isinstance(v, list) else 'tuple', str(len(v)).encode())
            for x in v: visit(x)
        elif v is None: frame('none')
        elif isinstance(v, bool): frame('bool', b'1' if v else b'0')
        elif isinstance(v, int): frame('int', str(v).encode())
        elif isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError('non-finite scalar state')
            frame('float', struct.pack('!d', v))
        elif isinstance(v, str): frame('str', v.encode('utf-8'))
        elif isinstance(v, bytes): frame('bytes', v)
        else: raise TypeError('unregistered state value: %s' % type(v).__name__)
    visit(value)
    return digest.hexdigest()


def capture_rng():
    return clone_tree({'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda_all']:
        if not torch.cuda.is_available() or len(state['torch_cuda_all']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device inventory mismatch')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda_all']])


def backend_flags():
    return {'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled()}


def restore_backend_flags(flags):
    torch.backends.cudnn.allow_tf32 = flags['cudnn_tf32']
    torch.backends.cuda.matmul.allow_tf32 = flags['matmul_tf32']
    torch.backends.cudnn.benchmark = flags['cudnn_benchmark']
    torch.backends.cudnn.deterministic = flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic'], warn_only=flags['deterministic_warn_only'])


@dataclass
class ActiveRuntime:
    model: object
    optimizers: dict
    schedulers: dict
    metadata: dict
    external: dict
    named_generators: dict = field(default_factory=dict)
    parameter_banks: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ActiveSnapshot:
    payload: dict
    state_hash: str


def _class_id(obj):
    return type(obj).__module__ + '.' + type(obj).__qualname__


def _bindings(runtime, optimizer):
    names = {id(p): name for name, p in runtime.model.named_parameters()}
    for key,bank in runtime.parameter_banks.items():
        names.update({id(p):'@bank/'+key+'/'+name for name,p in bank.parameters.items()})
    try:
        return [[names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    except KeyError:
        raise ValueError('optimizer owns an unregistered model parameter')


def snapshot_active(runtime):
    """Capture at a zero-grad boundary; CPU tensors never alias live state."""
    missing = REQUIRED_METADATA - runtime.metadata.keys()
    if missing:
        raise ValueError('missing active metadata: %s' % sorted(missing))
    if any(p.grad is not None for p in runtime.model.parameters()):
        raise ValueError('snapshot requires all model gradients to be None')
    if torch.cuda.is_available(): torch.cuda.synchronize()
    optimizers = {key: {'class_id': _class_id(opt), 'bindings': _bindings(runtime, opt),
        'defaults': opt.defaults,
        'state_dict': opt.state_dict()} for key, opt in runtime.optimizers.items()}
    opt_ids = {id(opt): key for key, opt in runtime.optimizers.items()}
    schedulers = {}
    for key, scheduler in runtime.schedulers.items():
        if id(scheduler.optimizer) not in opt_ids:
            raise ValueError('scheduler optimizer is not registered')
        schedulers[key] = {'class_id': _class_id(scheduler),
            'optimizer_id': opt_ids[id(scheduler.optimizer)], 'state_dict': scheduler.state_dict()}
    payload = clone_tree({'schema': SCHEMA, 'model_class': _class_id(runtime.model),
        'model': runtime.model.state_dict(),
        'module_training_modes': {n: m.training for n, m in runtime.model.named_modules()},
        'parameter_requires_grad': {n: p.requires_grad for n, p in runtime.model.named_parameters()},
        'optimizers': optimizers, 'schedulers': schedulers,
        'metadata': runtime.metadata, 'external': runtime.external, 'rng': capture_rng(),
        'backend_flags': backend_flags(),
        'named_generators': {k: {'device': str(g.device), 'state': g.get_state()}
                             for k, g in runtime.named_generators.items()}})
    if runtime.parameter_banks:
        banks={}
        rows=[]
        for key,bank in runtime.parameter_banks.items():
            rows.extend(bank.rows.detach().cpu().tolist())
            for name,p in bank.parameters.items():
                if p.grad is not None or not torch.equal(p,getattr(runtime.model,name)[bank.rows]):
                    raise ValueError('bank/model mismatch or uncleared gradient')
            banks[key]={'rows':bank.rows,'parameters':bank.parameters,
                        'requires_grad':{n:p.requires_grad for n,p in bank.parameters.items()}}
        if sorted(rows)!=list(range(runtime.model.count)):
            raise ValueError('bank rows must cover model exactly once')
        payload['parameter_banks']=clone_tree(banks)
    return ActiveSnapshot(payload, canonical_hash(payload))


def restore_active(snapshot, factory, expected_external, runtime_validator=None):
    """Stage a fresh runtime, validate it, restore RNG last, then return it.

    The caller publishes only on success: active = restore_active(...).
    Factory must create new objects; it must not mutate an existing runtime.
    This phase-0 function does not write a commit ledger or advance a trial.
    """
    data = snapshot.payload
    if data.get('schema') != SCHEMA or canonical_hash(data) != snapshot.state_hash:
        raise ValueError('snapshot schema/hash mismatch')
    if canonical_hash(data['external']) != canonical_hash(expected_external):
        raise ValueError('target/code/config/environment fingerprint mismatch')
    entry_rng, entry_flags = capture_rng(), backend_flags()
    try:
        candidate = factory(clone_tree(data))
        if _class_id(candidate.model) != data['model_class']:
            raise ValueError('model class mismatch')
        if set(candidate.optimizers) != set(data['optimizers']) or set(candidate.schedulers) != set(data['schedulers']):
            raise ValueError('optimizer/scheduler registry mismatch')
        candidate.model.load_state_dict(data['model'], strict=True)
        if set(candidate.parameter_banks)!=set(data.get('parameter_banks',{})):
            raise ValueError('parameter bank registry mismatch')
        for key,saved_bank in data.get('parameter_banks',{}).items():
            bank=candidate.parameter_banks[key]
            if not torch.equal(bank.rows.cpu(),saved_bank['rows']) or set(bank.parameters)!=set(saved_bank['parameters']):
                raise ValueError('parameter bank binding mismatch')
            with torch.no_grad():
                for name,p in bank.parameters.items():
                    saved=saved_bank['parameters'][name]
                    if p.shape!=saved.shape or p.dtype!=saved.dtype: raise ValueError('bank tensor schema mismatch')
                    p.copy_(saved); p.requires_grad_(saved_bank['requires_grad'][name])
        parameters = dict(candidate.model.named_parameters())
        if set(parameters) != set(data['parameter_requires_grad']):
            raise ValueError('parameter trainability registry mismatch')
        for name, enabled in data['parameter_requires_grad'].items():
            parameters[name].requires_grad_(enabled)
        modules = dict(candidate.model.named_modules())
        if set(modules) != set(data['module_training_modes']):
            raise ValueError('module mode registry mismatch')
        for name, mode in data['module_training_modes'].items(): modules[name].training = mode
        for key, opt in candidate.optimizers.items():
            saved = data['optimizers'][key]
            if _class_id(opt) != saved['class_id'] or _bindings(candidate, opt) != saved['bindings']:
                raise ValueError('optimizer class or parameter binding mismatch')
            opt.load_state_dict(clone_tree(saved['state_dict']))
            opt.defaults=clone_tree(saved['defaults'])
        for key, scheduler in candidate.schedulers.items():
            saved = data['schedulers'][key]
            if (_class_id(scheduler) != saved['class_id'] or
                    scheduler.optimizer is not candidate.optimizers[saved['optimizer_id']]):
                raise ValueError('actual scheduler binding mismatch')
            scheduler.load_state_dict(clone_tree(saved['state_dict']))
        candidate.metadata, candidate.external = clone_tree(data['metadata']), clone_tree(data['external'])
        if set(candidate.named_generators) != set(data['named_generators']):
            raise ValueError('named RNG registry mismatch')
        for key, generator in candidate.named_generators.items():
            saved = data['named_generators'][key]
            if str(generator.device) != saved['device']: raise ValueError('named RNG device mismatch')
            generator.set_state(saved['state'].cpu())
        if runtime_validator is not None: runtime_validator(candidate)
        restore_backend_flags(data['backend_flags'])
        restore_rng(data['rng'])
        if snapshot_active(candidate).state_hash != snapshot.state_hash:
            raise ValueError('restored active state differs from snapshot')
        return candidate
    except Exception:
        restore_backend_flags(entry_flags)
        restore_rng(entry_rng)
        raise


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_artifact(directory, reference):
    """Validate both serialized bytes and logical content before restoration."""
    name = reference['file']
    if Path(name).name != name or name != reference['state_hash'] + '.pt':
        raise ValueError('invalid artifact path')
    path = Path(directory) / 'artifacts' / name
    if _file_hash(path) != reference['file_sha256']:
        raise ValueError('artifact file hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    snapshot = ActiveSnapshot(saved['payload'], saved['state_hash'])
    if snapshot.state_hash != reference['state_hash'] or canonical_hash(snapshot.payload) != snapshot.state_hash:
        raise ValueError('artifact logical hash mismatch')
    return snapshot


class TrialStore:
    """Single-writer persisted transactions; the pointer is the commit authority.

    Hold this context for all writes. After any I/O/restore exception, recover()
    is mandatory before accessing active. A failpoint may terminate the process
    at each durable boundary. Decision policy is supplied explicitly by callers;
    this storage layer implements no growth or capacity-gain policy.
    """
    def __init__(self, directory, factory, expected_external, validator=None, failpoint=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'artifacts').mkdir(exist_ok=True)
        self.factory, self.external, self.validator = factory, expected_external, validator
        self.failpoint = failpoint or (lambda name: None)
        self._active, self.poisoned = None, True
        self.lock = (self.root / 'writer.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0)
                if not self.lock.read(1): self.lock.write(b'0'); self.lock.flush()
                self.lock.seek(0); msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lock.close()
            raise RuntimeError('trial store already has a writer')
        try: self.rows = self._read_ledger()
        except Exception:
            self.close(); raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        if not self.lock.closed:
            if os.name == 'nt':
                import msvcrt
                self.lock.seek(0); msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            self.lock.close()

    @property
    def active(self):
        if self.poisoned or self.lock.closed: raise RuntimeError('recover required before active access')
        return self._active

    def _read_ledger(self):
        path = self.root / 'ledger.jsonl'
        data = path.read_bytes() if path.exists() else b''
        # A process can die in append(). Preserve all bytes, seal the incomplete
        # final line, and attest it in the next hash-chained recovery row.
        rows, pending = [], None
        parts = data.splitlines(keepends=True)
        for index, raw in enumerate(parts):
            complete = raw.endswith(b'\n')
            try:
                row = json.loads(raw)
                claimed = row.pop('row_hash')
                valid = (canonical_hash(row) == claimed and row['sequence'] == len(rows)
                    and row['previous'] == (rows[-1]['row_hash'] if rows else None))
                if not valid: raise ValueError('ledger hash chain mismatch')
                if pending is not None:
                    if row['event'] != 'TORN_TAIL' or row['raw_sha256'] != pending:
                        raise ValueError('unattested ledger fragment')
                    pending = None
                row['row_hash'] = claimed; rows.append(row)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if pending is not None: raise ValueError('multiple corrupt ledger records')
                pending = hashlib.sha256(raw.rstrip(b'\n')).hexdigest()
                if index != len(parts)-1 and complete:
                    continue  # Only allowed if the next valid row attests this fragment.
                if not complete:
                    with path.open('ab') as f: f.write(b'\n'); f.flush(); os.fsync(f.fileno())
            if not complete and pending is None:
                with path.open('ab') as f: f.write(b'\n'); f.flush(); os.fsync(f.fileno())
        self.rows = rows
        if pending is not None:
            self._append('TORN_TAIL', raw_sha256=pending)
        return self.rows

    def _append(self, event, **fields):
        row = dict(schema='adasplat_trial_ledger_v1', sequence=len(self.rows),
                   previous=self.rows[-1]['row_hash'] if self.rows else None,
                   timestamp=datetime.now(timezone.utc).isoformat(),
                   event=event, **fields)
        # Hash the representation actually persisted by JSON: object keys are
        # strings and tuples become arrays. Never hash Python-only distinctions.
        def unique_object(pairs):
            result={}
            for key,value in pairs:
                if key in result: raise ValueError('colliding JSON object keys')
                result[key]=value
            return result
        row=json.loads(json.dumps(row,allow_nan=False),object_pairs_hook=unique_object)
        row['row_hash'] = canonical_hash(row)
        encoded = json.dumps(row, sort_keys=True, allow_nan=False).encode('utf-8') + b'\n'
        with (self.root / 'ledger.jsonl').open('ab') as f:
            f.write(encoded); f.flush(); os.fsync(f.fileno())
        self.rows.append(row)
        return row

    def _save(self, snapshot):
        from primix_teacher_artifacts import atomic_torch_save
        if canonical_hash(snapshot.payload) != snapshot.state_hash: raise ValueError('corrupt snapshot')
        path = self.root / 'artifacts' / (snapshot.state_hash + '.pt')
        if not path.exists():
            atomic_torch_save(path, dict(payload=snapshot.payload, state_hash=snapshot.state_hash))
        ref = dict(file=path.name, state_hash=snapshot.state_hash, file_sha256=_file_hash(path))
        read_artifact(self.root, ref)
        return ref

    def _pointer(self):
        pointer = json.loads((self.root / 'active.json').read_text(encoding='utf-8'))
        matches = [r for r in self.rows if r['event'] in ('INIT', 'PREPARE') and r['pointer'] == pointer]
        if len(matches) != 1: raise ValueError('pointer lacks unique persisted preparation')
        committed = [r for r in self.rows if r['event'] == 'COMMIT']
        if committed:
            latest = next(r['pointer'] for r in self.rows if r['event'] == 'PREPARE'
                          and r['pointer']['transaction'] == committed[-1]['transaction'])
            if pointer != latest and pointer['parent'] != latest['artifact']['state_hash']:
                raise ValueError('pointer rolls back a committed transaction')
        read_artifact(self.root, pointer['artifact'])
        return pointer

    def initialize(self, runtime):
        from primix_teacher_artifacts import atomic_json
        if self.rows or (self.root / 'active.json').exists(): raise ValueError('store already initialized')
        if canonical_hash(runtime.external) != canonical_hash(self.external): raise ValueError('initial external mismatch')
        if self.validator is not None: self.validator(runtime)
        ref = self._save(snapshot_active(runtime))
        pointer = dict(schema='adasplat_pointer_v1', transaction=uuid.uuid4().hex,
                       trial=None, branch='S0', artifact=ref, parent=None)
        self._append('INIT', pointer=pointer)
        self.failpoint('after_init')
        atomic_json(self.root / 'active.json', pointer)
        self._active, self.poisoned = runtime, False
        return runtime

    def recover(self):
        """Published PREPARE wins; unpublished PREPARE never becomes a commit."""
        from primix_teacher_artifacts import atomic_json
        self.poisoned = True
        self.rows = self._read_ledger()
        if not (self.root / 'active.json').exists():
            initial = [r for r in self.rows if r['event'] == 'INIT']
            if len(initial) != 1: raise ValueError('missing initial pointer')
            read_artifact(self.root, initial[0]['pointer']['artifact'])
            atomic_json(self.root / 'active.json', initial[0]['pointer'])
        pointer = self._pointer()
        restored = restore_active(read_artifact(self.root, pointer['artifact']),
                                  self.factory, self.external, self.validator)
        for row in list(self.rows):
            if row['event'] == 'PREPARE':
                tx = row['pointer']['transaction']
                if any(r.get('transaction') == tx and r['event'] in ('COMMIT','ABORT_PREPARED') for r in self.rows): continue
                event = 'COMMIT' if row['pointer'] == pointer else 'ABORT_PREPARED'
                self._append(event, transaction=tx, trial=row['trial'], recovery=True)
        if pointer['branch'] == 'A' and not any(r['event'] == 'CONTINUATION_COMMIT_A' and
            r.get('transaction') == pointer['transaction'] for r in self.rows):
            self._append('CONTINUATION_COMMIT_A', trial=pointer['trial'],
                         transaction=pointer['transaction'], recovery=True)
        for row in list(self.rows):
            if row['event'] == 'BEGIN':
                tid = row['trial']
                terminal = any(r.get('trial') == tid and r['event'] in ('RESTORE_CONFIRMED','COMMIT') for r in self.rows)
                if not terminal:
                    if pointer['artifact'] != row['s0']: raise ValueError('interrupted trial has unexpected active pointer')
                    if not any(r.get('trial') == tid and r['event'] == 'REJECT' for r in self.rows):
                        self._append('REJECT', trial=tid, reason='process interrupted', failed_invariants=[],
                                     unrecorded_cost='unknown; not zero', recovery=True)
                    self._append('RESTORE_CONFIRMED', trial=tid, state_hash=pointer['artifact']['state_hash'], recovery=True)
        self._active, self.poisoned = restored, False
        return restored

    def begin(self, details=None):
        s0 = snapshot_active(self.active)
        if s0.state_hash != self._pointer()['artifact']['state_hash']: raise ValueError('active is not persisted S0')
        if any(r['event'] == 'BEGIN' and not any(t.get('trial') == r['trial'] and
            t['event'] in ('RESTORE_CONFIRMED','COMMIT') for t in self.rows) for r in self.rows):
            raise ValueError('trial already open')
        self.poisoned = True
        ref = self._save(s0); self.failpoint('after_s0_artifact')
        tid = uuid.uuid4().hex
        self._append('BEGIN', trial=tid, s0=ref, details=details or {})
        self.failpoint('after_begin'); self.poisoned = False
        return tid

    def _trial(self, tid):
        matches = [r for r in self.rows if r.get('trial') == tid and r['event'] == 'BEGIN']
        if len(matches) != 1: raise ValueError('unknown trial')
        if self._pointer()['artifact'] != matches[0]['s0']: raise ValueError('stale trial origin')
        return matches[0]

    def record_cost(self, tid, **increments):
        self.active; self._trial(tid)
        if not increments or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in increments.values()):
            raise ValueError('cost increments must be finite and nonnegative')
        self.poisoned = True
        self._append('COST', trial=tid, increments=increments)
        self.poisoned = False

    def restore_origin(self,tid):
        """Explicit counterfactual branch restore; not a reject or commit."""
        self.active; origin=self._trial(tid)
        self.poisoned=True
        restored=restore_active(read_artifact(self.root,origin['s0']),self.factory,self.external,self.validator)
        self._append('BRANCH_RESTORE',trial=tid,state_hash=snapshot_active(restored).state_hash)
        self._active,self.poisoned=restored,False
        return restored

    def branch_end(self, tid, branch, candidate, metrics):
        self.active; origin = self._trial(tid)
        if branch not in ('A','B'): raise ValueError('branch must be A or B')
        if any(r.get('trial') == tid and r.get('branch') == branch and r['event'] == 'BRANCH_END' for r in self.rows):
            raise ValueError('branch already ended')
        if canonical_hash(candidate.payload['external']) != canonical_hash(self.external): raise ValueError('candidate external mismatch')
        self.poisoned = True
        ref = self._save(candidate)
        self._append('BRANCH_END', trial=tid, branch=branch, s0=origin['s0'], artifact=ref, metrics=metrics)
        self.poisoned = False

    def decide(self, tid, branch, evidence, policy):
        """Persist an explicit decision bound to the immutable completed branch.

        policy receives S0, candidate, branch metrics and supplied evidence.
        This is a transaction gate, not the unimplemented Phase 1 comparator.
        """
        self.active; origin = self._trial(tid)
        rows = [r for r in self.rows if r.get('trial') == tid and r.get('branch') == branch and r['event'] == 'BRANCH_END']
        if len(rows) != 1: raise ValueError('completed branch required')
        branch_row = rows[0]
        entry_rng, entry_flags = capture_rng(), backend_flags()
        try:
            approved = policy(read_artifact(self.root, origin['s0']),
                read_artifact(self.root, branch_row['artifact']), clone_tree(branch_row['metrics']), clone_tree(evidence))
        finally:
            restore_rng(entry_rng); restore_backend_flags(entry_flags)
        if not isinstance(approved, bool): raise ValueError('decision policy must return bool')
        self.poisoned = True
        self._append('DECISION', trial=tid, branch=branch, artifact=branch_row['artifact'], approved=approved, evidence=evidence)
        self.poisoned = False
        return approved

    def reject(self, tid, reason, failed_invariants=()):
        self.active; origin = self._trial(tid)
        if any(r.get('trial') == tid and r['event'] in ('REJECT','RESTORE_CONFIRMED') for r in self.rows):
            raise ValueError('trial already rejected; no implicit continuation')
        self.poisoned = True
        self._append('REJECT', trial=tid, reason=reason, failed_invariants=list(failed_invariants))
        self.failpoint('after_reject')
        restored = restore_active(read_artifact(self.root, origin['s0']), self.factory, self.external, self.validator)
        self.failpoint('after_restore')
        self._append('RESTORE_CONFIRMED', trial=tid, state_hash=snapshot_active(restored).state_hash)
        self._active, self.poisoned = restored, False
        return restored

    def commit(self, tid, branch, expected_s0_hash):
        from primix_teacher_artifacts import atomic_json
        self.active; origin = self._trial(tid)
        if origin['s0']['state_hash'] != expected_s0_hash: raise ValueError('commit S0 mismatch')
        rejected = any(r.get('trial') == tid and r['event'] == 'REJECT' for r in self.rows)
        if rejected and branch != 'A': raise ValueError('rejected B cannot commit')
        if branch == 'A' and not any(r.get('trial') == tid and r['event'] == 'RESTORE_CONFIRMED' for r in self.rows):
            raise ValueError('A continuation requires completed rejection and exact S0 restore')
        if branch == 'A' and snapshot_active(self.active).state_hash != expected_s0_hash:
            raise ValueError('A continuation active state is no longer exact S0')
        decisions = [r for r in self.rows if r.get('trial') == tid and r.get('branch') == branch and r['event'] == 'DECISION']
        if not decisions or not decisions[-1]['approved']: raise ValueError('approved explicit decision required')
        decision = decisions[-1]
        self.poisoned = True
        restored = restore_active(read_artifact(self.root, decision['artifact']), self.factory, self.external, self.validator)
        self.failpoint('after_candidate_restore')
        pointer = dict(schema='adasplat_pointer_v1', transaction=uuid.uuid4().hex, trial=tid,
                       branch=branch, artifact=decision['artifact'], parent=origin['s0']['state_hash'])
        self._append('PREPARE', trial=tid, pointer=pointer, decision_hash=decision['row_hash'])
        self.failpoint('after_prepare')
        atomic_json(self.root / 'active.json', pointer)
        self.failpoint('after_pointer')
        self._append('COMMIT', trial=tid, transaction=pointer['transaction'], recovery=False)
        self.failpoint('after_commit')
        if branch == 'A': self._append('CONTINUATION_COMMIT_A', trial=tid, transaction=pointer['transaction'])
        self._active, self.poisoned = restored, False
        return restored
