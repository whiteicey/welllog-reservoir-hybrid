"""Delete completed redundant recovery/probe caches; retain all selected sources.

Planning hashes are committed before unlink. No active output is eligible.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import argparse
import psutil
from fullcv_context_v97 import ROOT, STUDY
from fullcv_models_v97 import STAGES, fold_dir, root_dir, root_path, best_path


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def active_outputs():
    result=set()
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            args=proc.info.get('cmdline') or []
            if 'python' not in (proc.info.get('name') or '').lower():continue
            if any(Path(a).name=='train_stage_fullcv_v97.py' for a in args):
                f=int(args[args.index('--fold')+1]);stage=args[args.index('--stage')+1];result.add(stage_dir(f,stage).resolve())
            elif any(Path(a).name=='train_root_fullcv_v97.py' for a in args):result.add(Path(args[args.index('--out')+1]).resolve())
        except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError):continue
    return result


from fullcv_models_v97 import stage_dir
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    active=active_outputs();candidates=[];preserved=[]
    for f in (1,2,3,4):
        rdir=root_dir(f)
        if (rdir/f'fold_{f}_report.json').exists() and rdir.resolve() not in active:
            report=json.loads((rdir/f'fold_{f}_report.json').read_text())
            assert report['fold']==f and report['best_checkpoint_predictions_reproduced']
            required=[root_path(f),rdir/f'fold_{f}_conflict_weights.npz',rdir/f'fold_{f}_validation_history.json',rdir/f'fold_{f}_pred.npy']
            assert all(p.exists() for p in required)
            preserved.extend(required)
            candidates.extend(p for p in [rdir/f'fold_{f}_last_state.pt'] if p.exists())
            candidates.extend((rdir/f'fold_{f}_dynamics').glob('observation_*.npz'))
        for stage,*_ in STAGES:
            out=stage_dir(f,stage)
            if not (out/'report.json').exists() or out.resolve() in active:continue
            report=json.loads((out/'report.json').read_text())
            assert report['fold']==f and report['stage']==stage and report['best_predictions_reproduced_exactly']
            required=[best_path(f,stage),out/'channel_pred.npy',out/'validation_history.json',out/'config.json']
            assert all(p.exists() for p in required)
            preserved.extend(required)
            candidates.extend(p for p in [out/'last_state.pt'] if p.exists())
    workspace=(ROOT/'outputs/pure_dl_fullcv_v97').resolve()
    entries=[]
    for path in sorted(set(candidates)):
        path=path.resolve();assert path.is_relative_to(workspace) and not any(path.is_relative_to(p) for p in active)
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=file_sha(path)))
    if not entries:print('NO_ELIGIBLE_REDUNDANT_STATES');return
    stamp=datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%f')
    plan=STUDY/f'cleanup_plan_{stamp}.json'
    value=dict(planned_at=datetime.now().astimezone().isoformat(),scope='completed fullCV stages/root recovery and redundant probes only',
        preserved=[str(p.relative_to(ROOT)) for p in preserved],active_outputs=[str(p) for p in active],entries=entries,total_bytes=sum(e['bytes'] for e in entries),applied=False)
    plan.write_text(json.dumps(value,indent=2),encoding='utf-8')
    if a.apply:
        for e in entries:
            path=ROOT/e['path'];now=active_outputs()
            assert not any(path.resolve().is_relative_to(p) for p in now)
            assert path.stat().st_size==e['bytes'] and file_sha(path)==e['sha256']
            path.unlink()
        assert all(p.exists() for p in preserved) and all(not (ROOT/e['path']).exists() for e in entries)
        value.update(applied=True,finished_at=datetime.now().astimezone().isoformat());(STUDY/f'cleanup_done_{stamp}.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        note='\n\n## 完成阶段的冗余缓存清理\n仅已完成阶段的last恢复状态与已完成来源探测缓存，实际删除'+str(len(entries))+'文件/'+str(value['total_bytes'])+'字节。计划先保存路径/大小/SHA，删除前重读活动进程；全部选定模型、通道预测、训练配置/历史及后续Q依赖保留，不停止训练，不迁移C盘。证据：'+plan.name+'及对应done清单。完整OOF尚未完成。\n'
        for p in [STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as stream:stream.write(note)
    print('CLEANUP',json.dumps(dict(plan=str(plan),files=len(entries),bytes=value['total_bytes'],applied=a.apply)))


if __name__=='__main__':main()
