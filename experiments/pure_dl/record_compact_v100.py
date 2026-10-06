"""Record real compact stage results; assemble only actual selected predictions."""
from datetime import datetime
from pathlib import Path
import hashlib,json,re
import numpy as np
import psutil
import torch
from compact_five_stage_v100 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    records=[]
    for stage in STAGES:
        out=OUT/stage;state=read(out/'runtime_status.json');report=read(out/'report.json');pid=state.get('pid');live=False
        if pid:
            try:
                process=psutil.Process(pid);cmd=process.cmdline()
                live=process.is_running() and any('train_compact_v100.py' in s for s in cmd) and stage in cmd
            except psutil.Error:pass
        records.append(dict(stage=stage,report=report,status=state,live_process_verified=live))
    now=datetime.now().astimezone();launch=read(STUDY/'head_launches.json')
    started=min(datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',r['started'])) for r in launch)
    elapsed=(now-started).total_seconds()
    snapshot=dict(at=now.isoformat(),stages=records,all_new_stages_complete=all(r['report'] for r in records),
                  total_training_stages=5,source_stage_included=True,new_incremental_wall_seconds=elapsed,remaining_folds_started=False)
    atomic_json(STUDY/'progress.json',snapshot)
    if snapshot['all_new_stages_complete']:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        selected=('por','perm','sw_density');pred=np.column_stack([np.load(OUT/s/'channel_pred.npy') for s in selected])
        assert pred.shape==(144665,3) and np.isfinite(pred).all()
        truth=data['truth'][data['validation']];scores=base.metric(truth,pred)
        for c,s in enumerate(selected):assert abs(scores[base.TS[c]]-read(OUT/s/'report.json')['target_score'])<1e-9
        source=read(STUDY/'TIMING_CONTRACT.json')
        # Actual batch wall includes foreground analysis/explicit launch gaps;
        # sum of seconds is also reported, never treated as parallel wall.
        result=dict(scores=scores,total_training_stages=5,source_training_stages=1,new_training_stages=4,
                    original_fifth=True,complete_five_folds=False,remaining_folds_started=False,
                    gain_vs_v96=scores['TOTAL']-80.50419742447689,strict_808_passed=qualifies(scores['TOTAL']),
                    source_pretrain_seconds=source['source_pretrain_seconds'],new_incremental_wall_seconds=elapsed,
                    historical_source_plus_incremental_cost_seconds=source['source_pretrain_seconds']+elapsed,
                    sum_new_stage_seconds=sum(r['report']['seconds'] for r in records),
                    elapsed_scope='actual two-head launch to completed compact assembly; IO/eval/explicit host gaps included; historical source reused and separately counted',
                    original_validation_raw_rows_sha256=hashlib.sha256(np.flatnonzero(np.isin(data['raw_to_valid'],data['validation'])).tobytes()).hexdigest(),
                    selected_stages=list(selected),reports=[read(OUT/s/'report.json') for s in selected])
        np.save(STUDY/'assembled_pred.npy',pred);atomic_json(STUDY/'assembled_report.json',result)
        lines=['# v100实际第五折结论','',f"TOTAL={scores['TOTAL']:.12f}，相对旧80.504197={result['gain_vs_v96']:+.12f}；严格>80.8={result['strict_808_passed']}。",'',
               '实际三通道均来自本批所训练的紧凑阶段，不借用旧18阶段成品。来源预训练计入后共5阶段；仅原第五，未验证完整五折。','',
               '|通道|成绩|阶段|最佳/停止轮|','|---|---:|---|---|']
        for s in selected:
            r=read(OUT/s/'report.json');lines.append(f"|{r['target']}|{r['target_score']:.12f}|{s}|{r['best_epoch']}/{r['stopped_epoch']}|")
        lines+=['',f"源训练历史成本{source['source_pretrain_seconds']:.3f}秒，本批实际墙钟{elapsed:.3f}秒，合计成本估计{source['source_pretrain_seconds']+elapsed:.3f}秒；没有同工作量同时期串行基准，不宣称实测加速比。",'',
                '未通过门槛或无收益时不补其他折，保留分通道收益诊断，改变造成退化的机制；旧交付保持不变。']
        (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        with (STUDY/'README.md').open('a',encoding='utf-8') as f:f.write('\n\n## 本批实际完成结论\n'+ '\n'.join(lines[2:])+'\n')
        note=f"\n\nv100紧凑五总阶段已完成原第五实际组合：TOTAL={scores['TOTAL']:.12f}，相对旧配方{result['gain_vs_v96']:+.12f}，严格>80.8={result['strict_808_passed']}；无其他折，完整OOF>82未完成。证据results/deep_learning/compact_five_stage_v100_20260930/FINAL.md。\n"
    else:
        note='\n\nv100当前实际快照：'+json.dumps({r['stage']:dict(status=r['status'].get('status','prepared'),score=r['report'].get('target_score',r['status'].get('best')),live=r['live_process_verified']) for r in records},ensure_ascii=False)+'。完整紧凑三通道收益尚未证实；无其他折。\n'
    for p in [ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print('COMPACT_RECORD',json.dumps({k:v for k,v in snapshot.items() if k!='stages'}),flush=True)

if __name__=='__main__':main()
