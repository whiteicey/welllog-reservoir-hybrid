"""Sequential LR screens, stop on verified qualification; never restart failed jobs."""
import json,subprocess,sys
from pathlib import Path
repo=Path.cwd();summary=repo/'outputs/lr55_sweep';summary.mkdir(exist_ok=True)
configs=[('cosine',.001,'1e3'),('cosine',.0003,'3e4'),('cosine',.003,'3e3'),('plateau',.001,'1e3')]
results=[]
for schedule,lr,tag in configs:
    out=repo/f'outputs/pure_dl_lr55_{schedule}_{tag}'
    if not (out/'screening_verification.json').exists():
        with (out/'train.log').open('w',encoding='utf-8') as log,(out/'train.err').open('w',encoding='utf-8') as err:
            proc=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_lr_sweep_sequence.py','--out',str(out),'--learning-rate',str(lr),'--schedule',schedule],stdout=log,stderr=err)
            print('TRAINING',schedule,lr,'PID',proc.pid,flush=True)
            assert proc.wait()==0,'Training failed; inspect existing job, no restart'
        subprocess.run([sys.executable,'experiments/pure_dl/verify_lr_sweep_sequence.py','--out',str(out),'--folds','1'],check=True)
    v=json.loads((out/'screening_verification.json').read_text());assert v['passed']
    r=json.loads((out/'fold_1_report.json').read_text())
    item={'schedule':schedule,'initial_lr':lr,'score':v['first_fold_score'],'best_epoch':r['best_epoch'],'stopped_epoch':r['stopped_epoch'],'seconds':r['seconds'],'peak_mib':r['peak_gpu_allocated_mib'],'qualified':v['qualifies_for_remaining_folds'],'output':str(out)}
    results.append(item);(summary/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    line=f"{schedule} LR={lr}: 首折{item['score']:.6f}，最佳{item['best_epoch']}轮，停止{item['stopped_epoch']}轮，{item['seconds']:.2f}秒，峰值{item['peak_mib']:.2f}MiB，核验通过，过门槛={item['qualified']}。"
    with (out/'README.md').open('a',encoding='utf-8') as f:f.write('\n\n## 首折结果\n\n### 探索重点\n固定55轮/早停12的学习率对照。\n\n### 已经探索\n实际LR、停止规则、最佳权重与恢复预测、原标签及官方评分独立核验通过。\n\n### 取得收益\n'+line+'\n\n### 不值得继续探索\n未过82.5不补折，不把扫描最好分当成独立盲测。\n')
    for path in [repo.parent/'PROJECTS.md',repo/'PROJECTS.md',repo/'experiments/pure_dl/PROJECTS.md',repo/'EXPLORATIONS.md']:
        with path.open('a',encoding='utf-8') as f:f.write('\n- LR55扫描已核验：'+line+'\n')
    print('VERIFIED_RESULT',json.dumps(item),flush=True)
    if item['qualified']:
        print('QUALIFIED_STOP_SWEEP_FOR_FULL_CV',flush=True);break
(summary/'status.json').write_text(json.dumps({'completed_screens':len(results),'planned_screens':4,'qualified':any(x['qualified'] for x in results),'all_screens_complete':len(results)==4},indent=2),encoding='utf-8')
