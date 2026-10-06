"""Record observed scheduling overlap; do not infer a speedup without a control."""
from datetime import datetime
import json
from pathlib import Path

import psutil

from fullcv_context_v97 import ROOT, STUDY
from record_overlap_fullcv_v97 import coherent_json_reads


def main():
    now = datetime.now().astimezone()
    with coherent_json_reads():
        events = [json.loads(line) for line in (STUDY/'overlap_events.jsonl').read_text(encoding='utf-8').splitlines()]
        status = json.loads((STUDY/'controller_status.json').read_text(encoding='utf-8'))
        authorization = json.loads((STUDY/'authorization.json').read_text(encoding='utf-8'))
        active = set()
        seconds = {'0': 0., '1': 0., '2': 0.}
        previous = datetime.fromisoformat(events[0]['at'])
        for event in events:
            at = datetime.fromisoformat(event['at'])
            seconds[str(len(active))] = seconds.get(str(len(active)), 0.) + (at-previous).total_seconds()
            if event['kind'] in ('takeover', 'resume'):
                active.update(event.get('adopted_labels', []))
            elif event['kind'] == 'launch':
                active.add(event['label'])
            elif event['kind'] == 'complete':
                active.discard(event['label'])
            previous = at
        seconds[str(len(active))] = seconds.get(str(len(active)), 0.) + (now-previous).total_seconds()
        live = []
        for job in status.get('active_jobs', []):
            try:
                process = psutil.Process(job['pid'])
                live.append({'label': job['label'], 'pid': job['pid'],
                             'creation_time_matches': abs(process.create_time()-job['creation_time']) < .01})
            except psutil.NoSuchProcess:
                live.append({'label': job['label'], 'pid': job['pid'], 'missing': True})
        items = []
        for fold in range(1, 5):
            base = ROOT/'outputs/pure_dl_fullcv_v97'/f'fold_{fold}'
            root = ROOT/'outputs/pure_dl_fullcv_v97/root' if fold == 1 else base/'root'
            reports = list(base.glob('*/report.json')) + [root/f'fold_{fold}_report.json']
            for path in reports:
                if not path.exists():
                    continue
                report = json.loads(path.read_text(encoding='utf-8'))
                items.append({'fold': fold, 'report': str(path.relative_to(ROOT)),
                              'seconds': report.get('seconds'),
                              'peak_allocated_MiB': report.get('peak_allocated_MiB', report.get('peak_gpu_allocated_mib'))})
    snapshot = dict(recorded_at=now.isoformat(), final=status.get('status') == 'complete',
                    authorized_batch_elapsed_seconds=(now-datetime.fromisoformat(authorization['authorized_at'])).total_seconds(),
                    overlap_controller_elapsed_seconds=(now-datetime.fromisoformat(events[0]['at'])).total_seconds(),
                    observed_scheduler_task_count_seconds=seconds, currently_live_processes=live,
                    completed_item_timings=items,
                    scope='Scheduling intervals include loading, warmup, training, validation, checkpoint IO, and controller observation lag. Item timers cover their own recorded scope; do not sum them as batch wall time.',
                    limitation='Task overlap is not simultaneous GPU compute. Folds and stopping epochs differ; there is no equivalent-work serial control, so speedup is unproven.')
    (STUDY/'OVERLAP_TIMING.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(json.dumps({key: snapshot[key] for key in ('recorded_at', 'final', 'authorized_batch_elapsed_seconds',
                                                     'observed_scheduler_task_count_seconds', 'currently_live_processes')}))


if __name__ == '__main__':
    main()
