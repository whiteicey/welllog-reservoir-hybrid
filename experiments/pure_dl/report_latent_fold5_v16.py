"""V16 runtime snapshots, prediction accounting, final summary and atlas overlay."""
from pathlib import Path
from datetime import datetime
import argparse,json
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/latent_fold5_v16_20260928'
REFERENCE=ROOT/'outputs/pure_dl_independent_v8_selected_fold5'
MODES=('tcn','latent')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sw_kernel(pred,truth):
    return 100*np.maximum(0,1-np.abs(pred-truth)/(.05*(np.abs(truth)+1e-5)))


def comparison(a,b,truth,wells,default):
    delta=sw_kernel(a,truth)-sw_kernel(b,truth)
    perwell=[dict(well=int(w),n=int((wells==w).sum()),SW_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
    counts=np.array([x['n'] for x in perwell]);sums=np.array([x['n']*x['SW_delta'] for x in perwell])
    rng=np.random.default_rng(20261010);draw=rng.integers(0,len(counts),(10000,len(counts)))
    boot=.35*sums[draw].sum(1)/counts[draw].sum(1)
    return dict(SW_delta=float(delta.mean()),TOTAL_delta=float(.35*delta.mean()),
        default_SW_delta=float(delta[default].mean()),numeric_SW_delta=float(delta[~default].mean()),
        default_TOTAL_contribution=float(.35*delta[default].sum()/len(delta)),numeric_TOTAL_contribution=float(.35*delta[~default].sum()/len(delta)),
        wins=sum(x['SW_delta']>1e-10 for x in perwell),losses=sum(x['SW_delta']< -1e-10 for x in perwell),per_well=perwell,
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]).tolist())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    reference=load(REFERENCE/'fold_5_report.json')['scores'];rows=[];configs=[];starts=[];ends=[]
    with np.load(REFERENCE/'oof.npz') as z:
        use=z['fold']==4;truth=z['truth'][use,2];wells=z['well'][use]
    ref=np.load(REFERENCE/'fold_5_pred.npy')[:,2]
    with np.load(STUDY/'data_outer.npz') as z:
        default=z['default_truth'][z['monitor_raw']]
        assert np.array_equal(z['truth'][z['monitor_raw']],truth)
    predictions={}
    for mode in MODES:
        ident='pure_dl_latent_v16_'+mode+'_fold5';out=ROOT/'outputs'/ident
        status=load(out/'runtime_status.json') or {'status':'not_started'}
        cfg=load(out/'config.json');report=load(out/'fold_5_report.json')
        hist=load(out/'validation_history.json') or []
        row=dict(mode=mode,id=ident,status=status,config=cfg,report=report,last=hist[-1] if hist else None)
        if cfg:configs.append(cfg)
        if report:
            pred=np.load(out/'channel_pred.npy');assert pred.shape==truth.shape
            assert abs(sw_kernel(pred,truth).mean()-report['target_score'])<1e-9
            predictions[mode]=pred;row['vs_old']=comparison(pred,ref,truth,wells,default)
            starts.append(datetime.fromisoformat(load(out/'launch.json')['started_at']));ends.append(datetime.fromisoformat(report['finished_at']))
        rows.append(row)
    fields=('seed','epochs','patience','minimum_epochs','steps_per_epoch','batch_size','lr','eta_min','weight_decay','clip_norm','Q','loss','decoder','fit_wells','monitor_wells','data_manifest_sha256','common_stem_output_initial_hash')
    if len(configs)==2:
        assert all(configs[0][k]==configs[1][k] for k in fields)
    done=all(r['report'] for r in rows)
    assembly=load(STUDY/'assembled_report.json')
    result=dict(observed_at=datetime.now().astimezone().isoformat(),all_training_complete=bool(done),assembly_complete=assembly is not None,
        original_fifth_evaluated=True,reference_scores=reference,rows=rows,assembly=assembly,
        matched_fields=list(fields),parameters_matched=False,complete_five_folds=False,remaining_folds_started=False)
    if done:
        result['latent_vs_tcn']=comparison(predictions['latent'],predictions['tcn'],truth,wells,default)
        result['training_timing']=dict(maximum_concurrency=2,launch_to_both_finished_wall_seconds=(max(ends)-min(starts)).total_seconds(),
            per_run_seconds={r['mode']:r['report']['seconds'] for r in rows},peak_allocated_MiB={r['mode']:r['report']['peak_allocated_MiB'] for r in rows},
            scope='includes cached-data loading,training,per-epoch full fit/fifth evaluations,best restoration;excludes data preparation,code work and final three-model disk replay')
    write(STUDY/'progress.json',result)
    # Append/replace only this study's curation entries; do not regenerate old metadata.
    cpath=ROOT/'docs/deep_learning/atlas/curation.json';cur=load(cpath)
    for row in rows:
        mode=row['mode'];report=row['report'];state=row['status'];ident=row['id']
        value=report['target_score'] if report else state.get('best_target_score')
        source='outputs/'+ident+('/fold_5_report.json' if report else '/runtime_status.json')
        primary=None if value is None else dict(scope='原第五折·SW'+('' if report else '·运行快照'),value=value,scores={'SW':value},kind='target',fold=5,source=source)
        if assembly:
            scores=next(x['scores'] for x in assembly['candidates'] if x['mode']==mode)
            primary=dict(scope='第五折·固定POR/PERM+'+mode,value=scores['TOTAL'],scores=scores,kind='fold',fold=5,
                source='results/deep_learning/latent_fold5_v16_20260928/assembled_report.json')
        conclusion=('完成SW %.9f，最佳%d轮，停止%d轮。'%(value,report['best_epoch'],report['stopped_epoch'])) if report else '尚在训练，无最终得分。'
        conclusion+=('三私有模型已实际磁盘重载并合并评分；' if assembly else '真实三模型磁盘组合尚未完成；')+'未补原其他折，未改旧交付。'
        cur.setdefault('nodes',{})[ident]=dict(title=('TCN控制' if mode=='tcn' else '潜变量读写记忆')+'·原第五折v16',stage='backbone',
            status='recorded' if report else 'running_snapshot',primary=primary,results=[] if primary is None else [primary],
            summary='原64训练井从头训练，损失/输入/评分与匹配对照固定；保持v15架构，其他业务通道不更新。',
            conclusion=conclusion,evidence=[source,'results/deep_learning/latent_fold5_v16_20260928/METHOD.md'])
        parent='pure_dl_architecture_v15_latent_inner' if mode=='latent' else 'pure_dl_sw_group_v14_inner_point80'
        edges=cur.setdefault('edges',[]);edges[:]=[e for e in edges if not(e['source']==parent and e['target']==ident)]
        edges.append(dict(source=parent,target=ident,type='evolution',note='从内层结构筛查推进原第五折，重新拟合64井；没有继承内层权重。',evidence='outputs/'+ident+'/config.json'))
    src,dst=rows[0]['id'],rows[1]['id'];edges=cur['edges'];edges[:]=[e for e in edges if not(e['source']==src and e['target']==dst)]
    edges.append(dict(source=src,target=dst,type='control',note='相同数据、训练和输出，完整结构不同，参数量不匹配。',evidence='results/deep_learning/latent_fold5_v16_20260928/progress.json'))
    write(cpath,cur)
    if done and assembly:
        selected=assembly['selected_scores'];best_new=max(assembly['candidates'],key=lambda x:x['scores']['TOTAL'])
        lines=['# v16：潜变量记忆原第五折实际总分','',
            '本批完成TCN和原样v15 latent在原64训练井上的重新训练，随后重载两个SW和原私有POR/PERM，实际合并144665点评分。','',
            '|SW方案|SW分|三通道TOTAL|相对v5 TOTAL|最佳/停止轮次|','|---|---:|---:|---:|---|',
            f"|旧v5/v8双专家SW|{reference['SW']:.9f}|{reference['TOTAL']:.9f}|0|历史保留|"]
        for row in rows:
            mode=row['mode'];f=row['report'];a=next(c for c in assembly['candidates'] if c['mode']==mode)
            lines.append(f"|{mode}|{a['scores']['SW']:.9f}|{a['scores']['TOTAL']:.9f}|{a['delta_v5_TOTAL']:+.9f}|{f['best_epoch']}/{f['stopped_epoch']}|")
        pair=result['latent_vs_tcn']
        lines+=['',f"当前保留TOTAL={selected['TOTAL']:.12f}；采用新SW={assembly['selected_new_SW']}。本批新候选中最高是{best_new['mode']}。",
            f"latent相对同批TCN：SW{pair['SW_delta']:+.9f}，TOTAL{pair['TOTAL_delta']:+.9f}；{pair['wins']}/16井提高、{pair['losses']}井下降。",
            '', '## 验证与归因边界',
            '两组输入/输出初始参数、预处理缓存、种子、窗口、优化器、预算上限及评分一致，均从头训练；TCN82113和latent174849参数，不是同容量因果消融。',
            '原第五折标签仅用于早停、选模和报告；未拟合训练统计或进行逐点模型选择。原第五折反复用于开发，不是独立测试，也非完整OOF。',
            'latent保持v15结构，Q=1；与历史v5比较的是不同整体方案，不能把单个分差全部归因于网络。历史内层76.590731和本次SW分数属不同井集合，不能相减作为退化量。',
            '两个SW检查点和固定POR/PERM分别从磁盘构建、全量推理；144665点预测精确复现、行序一致、模型间参数存储无交集。POR/PERM预测保持不变。',
            '', '## 后续',
            '原其他四折和旧交付未改；相同规模的记忆模块消融及POR/PERM新架构训练尚未执行。先依据原第五折与逐井差决定结构是否值得继续，不追加自动扫描或队列。',
            '完整OOF>82未完成。本批保存best/current状态，未验证中断恢复与连续训练逐位等价。CUDA attention可能存在非确定性，固定种子不等于逐位训练保证。',
            '', '## 证据',
            'assembled_report.json为真实组合分数；progress.json为匹配配置、逐井和默认/非默认分解；CURRENT_RECIPE.json为保留配方。数据范围见data_outer_manifest.json，方法见METHOD.md。',
            '图谱由本报告非破坏合并curation.json并运行build_atlas.py；保留既有节点，未重置旧主curate.py。']
        (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if args.record:
            marker='## v16原第五折与真实组合验证完成'
            note='\n\n'+marker+'\n'+result['observed_at']+'；'+str({x['mode']:x['scores']['TOTAL'] for x in assembly['candidates']})+'。当前保留'+str(selected['TOTAL'])+'，采用新SW='+str(assembly['selected_new_SW'])+'。已真实磁盘重载各候选SW及固定POR/PERM，144665点合并，未补其他折或改旧交付。报告results/deep_learning/latent_fold5_v16_20260928/FINAL.md。\n'
            for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
                if marker not in path.read_text(encoding='utf-8-sig'):
                    with path.open('a',encoding='utf-8') as f:f.write(note)
            for row in rows:
                out=ROOT/'outputs'/row['id'];rep=row['report']
                local='\n\n## 完成结论\n真实第五折SW='+str(rep['target_score'])+'，组合TOTAL='+str(next(c['scores']['TOTAL'] for c in assembly['candidates'] if c['mode']==row['mode']))+'。最佳/停止轮次='+str(rep['best_epoch'])+'/'+str(rep['stopped_epoch'])+'；磁盘恢复及全量组合验证完成。收益和后续边界见研究FINAL.md。原其他折/旧交付不变，完整OOF目标未完成。\n'
                for name in ('README.md','PROJECTS.md'):
                    if '## 完成结论' not in (out/name).read_text(encoding='utf-8'):
                        with (out/name).open('a',encoding='utf-8') as f:f.write(local)
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            p=proc.info;argv=p.get('cmdline') or []
            if 'python' in (p.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(p['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps(dict(training_complete=bool(done),assembly_complete=assembly is not None,
        rows=[dict(mode=r['mode'],status=r['status'],SW=r['report']['target_score'] if r['report'] else None,
            total=r['report']['reference_replacement_total'] if r['report'] else None,
            best_epoch=r['report']['best_epoch'] if r['report'] else None,
            stopped_epoch=r['report']['stopped_epoch'] if r['report'] else None) for r in rows],
        latent_vs_tcn=result.get('latent_vs_tcn'),retained_scores=assembly['selected_scores'] if assembly else None,active_training_pids=active),ensure_ascii=False))


if __name__=='__main__':main()
