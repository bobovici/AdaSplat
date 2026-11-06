"""Closest real primitive-count snapshots, independent of growth decisions."""
import copy
import math
from pathlib import Path


class PrimitiveSnapshots:
    def __init__(self, counts, tolerance=.1):
        self.counts = list(counts)
        self.tolerance = tolerance
        self.entries = {}
        self.last_count = None

    def observe(self, payload, metrics, weight):
        """Called only for the active, accepted model; never rejected candidates."""
        if not self.counts:
            return False
        count, step = int(payload['N']), int(payload['step'])
        if self.last_count is not None and count < self.last_count:
            raise ValueError('Primitive snapshot observations must use nondecreasing active counts')
        self.last_count = count
        score = float(metrics['MSE']) + weight*(1-float(metrics['SSIM']))
        if not math.isfinite(score):
            raise ValueError('Nonfinite primitive snapshot score')
        changed = False
        for target in self.counts:
            entry = self.entries.get(target)
            distance = abs(count-target)
            rank = (distance, score, -step)
            if entry is None or rank < tuple(entry['rank']):
                old_file = entry.get('file') if entry else None
                crossed = entry.get('crossed',False) if entry else False
                entry = dict(target_N=target, actual_N=count, step=step, rank=rank,
                    metrics=copy.deepcopy(metrics), payload=payload, crossed=crossed, file=old_file,
                    dirty=True)
                self.entries[target] = entry
            if count >= target and not entry['crossed']:
                entry['crossed'] = True
                entry['dirty'] = True
            # Within tolerance may qualify before the count reaches the target exactly.
            entry['eligible'] = entry['crossed'] or entry['rank'][0] <= target*self.tolerance
            changed = changed or (entry['eligible'] and entry['dirty'])
        return changed

    def summary(self):
        rows = []
        for target in self.counts:
            entry = self.entries.get(target)
            available = bool(entry and entry.get('eligible') and entry.get('file'))
            rows.append(dict(target_N=target,status='saved' if available else 'pending',
                actual_N=entry['actual_N'] if entry else None,
                step=entry['step'] if entry else None,
                relative_count_error=entry['rank'][0]/target if entry else None,
                target_crossed=bool(entry and entry['crossed']),
                within_tolerance=bool(entry and entry['rank'][0]<=target*self.tolerance),
                metrics=copy.deepcopy(entry['metrics']) if available else None,
                checkpoint='milestones/'+entry['file'] if available else None))
        saved = sum(row['status']=='saved' for row in rows)
        return dict(schema='adasplat_primitive_snapshots_v1',requested_counts=self.counts,
            saved=saved,total=len(rows),complete=saved==len(rows),tolerance=self.tolerance,
            distinct_saved_counts=len({r['actual_N'] for r in rows if r['status']=='saved'}),
            missing_counts=[r['target_N'] for r in rows if r['status']!='saved'],entries=rows)

    def flush(self, output, save_payload, save_json, force=False):
        if not self.counts:
            return
        root = Path(output)/'milestones'
        root.mkdir(parents=True,exist_ok=True)
        obsolete = []
        for target in self.counts:
            entry = self.entries.get(target)
            if not entry or not entry['eligible'] or not (force or entry['dirty']):
                continue
            name = 'n{0:06d}_actual{1:06d}.pt'.format(target,entry['actual_N'])
            previous = entry.get('file')
            # Store the genuine model N and add the requested target as separate metadata.
            payload = dict(entry['payload'])
            payload['primitive_snapshot'] = dict(target_N=target,actual_N=entry['actual_N'],
                step=entry['step'],metrics=entry['metrics'],relative_count_error=entry['rank'][0]/target)
            save_payload(payload,root/name)
            entry['file'],entry['dirty'] = name,False
            if previous and previous != name:
                obsolete.append(previous)
        save_json(root/'index.json',self.summary())
        # Delete superseded managed files only after all replacements and the index are saved.
        for name in obsolete:
            old = root/name
            if old.parent != root or old.name != name or not name.startswith('n') or not name.endswith('.pt'):
                raise ValueError('Invalid managed snapshot filename')
            if old.exists():
                old.unlink()

    def state_dict(self):
        # deepcopy preserves shared payload references between nearby targets.
        return copy.deepcopy(dict(counts=self.counts,tolerance=self.tolerance,
                                  entries=self.entries,last_count=self.last_count))

    def load_state_dict(self, state):
        if state['counts'] != self.counts or state['tolerance'] != self.tolerance:
            raise ValueError('Primitive snapshot recipe mismatch')
        self.entries = copy.deepcopy(state['entries'])
        self.last_count = state['last_count']
