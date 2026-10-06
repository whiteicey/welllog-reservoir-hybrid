"""Build an offline DL lineage atlas from compact local evidence; never load weights/data."""
from pathlib import Path
import ast, csv, json, re
from datetime import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'outputs'
SRC = ROOT / 'experiments/pure_dl'

def read(p):
    try: return p.read_text(encoding='utf-8-sig')
    except (OSError, UnicodeError): return ''

def js(p):
    try: return json.loads(read(p))
    except (ValueError, TypeError): return {}

def rel(p): return p.relative_to(ROOT).as_posix()

def score(d):
    if not isinstance(d, dict): return None
    for k in ['screening_scores', 'oof_scores', 'scores', 'blended_scores', 'tuned_validation_scores', 'recomputed_scores', 'score', 'official_scores', 'best_scores']:
        s = d.get(k)
        if isinstance(s, dict) and isinstance(s.get('TOTAL'), (int, float)): return s
    if isinstance(d.get('TOTAL'), (float, int)): return d
    return None

def module_network(p):
    try: tree = ast.parse(read(p))
    except SyntaxError: return None
    for x in tree.body:
        if isinstance(x, ast.ImportFrom):
            if any(a.name == 'Network' for a in x.names): return x.module
    return None

def compact_cfg(cfg):
    skip = ['dependency_sha256','outer_wells','numeric_indices','conditioning_indices','indices','names_in_input_order','required_names']
    def small(x, depth=0):
        if depth > 3: return str(x)[:400]
        if isinstance(x, dict): return {k:small(v,depth+1) for k,v in x.items() if k not in skip}
        if isinstance(x, list): return x if len(x) <= 16 else x[:8]+['… 共%d项'%len(x)]
        return x
    return small(cfg)

def stage(name):
    if any(t in name for t in ['vote','posterior_overlap','posterior_logpool','posterior_fine','five_model','four_model','fixed_','three_seed','validation_grid']): return 'vote'
    if 'paired_' in name or 'three_view' in name: return 'pair'
    if any(t in name for t in ['global_','conflict','quality','window128','window160','structured','well_tail','full_input','default75','narrow75','joint_default','por_t','por_student','patch_','external_context','near_far','fine_global','input512','cosine','heads8','stacked','shifted','learned_patch','wide150']): return 'context'
    if 'gen_' in name: return 'generation'
    if any(t in name for t in ['blur','norm','absolute','relative','private','likelihood','interaction','gated','population','full_resolution','parallel_depth','learned_upsample','single_gaussian','mixture16','ordered_mixture','density_adapter','curve_token','recurrent','kernel3','regularized','aux20','main_only','sw_','perm_','hurdle','utility_']): return 'backbone'
    if any(t in name for t in ['sequence','batch_size','lr55','onecycle','plateau','warmup','earlystop']): return 'sequence'
    if name.startswith('pure_dl_'): return 'tabular'
    return 'start'

STAGES = [
    ('start','起点 · CNN / MLP','先建立可运行、可比较的纯神经网络基线'),
    ('tabular','表格网络与概率预测','分布预测、特征编码、专家与早期融合'),
    ('sequence','走向深度序列','多尺度卷积、预算、学习率和时序机制'),
    ('backbone','骨干与监督机制','归一化、低通下采样、门控和目标损失'),
    ('generation','生成式岔路','Spline flow、flow matching 与 diffusion'),
    ('context','质量控制与上下文','全点质量表、重尾分布、patch 与全局信息'),
    ('pair','双专家与特征工程','不同输入视图、条件校准、路由和新增特征'),
    ('vote','冻结专家与投票','多种子、解码精度、窗口重叠和学习权重'),
]

def main():
    previous = js(HERE/'atlas-data.json')
    curated = js(HERE/'curation.json')
    overrides = curated.get('nodes', {})
    dirs = [p for p in OUT.iterdir() if p.is_dir() and (p.name.startswith('pure_dl_') or p.name in ['simple_cnn_cv','simple_cnn_anomaly_cv','cnn_mlp_cv'])]
    excluded = {'pure_dl_summary':'汇总目录，结果并入对应实验', 'pure_dl_feature_expansion_study':'输入审计和规划，已关联实际特征实验'}
    dirs = [p for p in dirs if p.name not in excluded]
    # Nested seed runs, decoder variants, and archived sub-configurations remain individual attempts.
    for p in list(dirs):
        for q in sorted(p.iterdir()):
            if q.is_dir() and not re.match(r'fold_\d+_dynamics', q.name) and any((q/f).exists() for f in ['config.json','report.json','fold_5_report.json']): dirs.append(q)
    nodes=[]
    for p in dirs:
        ident=p.relative_to(OUT).as_posix()
        cfg=js(p/'config.json'); report=js(p/'report.json'); screen=js(p/'screening_report.json')
        if not cfg: cfg=report.get('config',screen.get('config',{}))
        if not isinstance(cfg,dict): cfg={}
        stop=js(p/'manual_early_stop.json')
        runtime=js(p/'runtime_status.json')
        ev=[]; results=[]; foldreports=[]
        for f in sorted(p.glob('fold_*_report.json')):
            d=js(f);s=score(d)
            if s:
                num=d.get('fold',int(re.search(r'fold_(\d+)',f.name)[1]))
                results.append({'scope':f'第{num}折','value':s['TOTAL'],'scores':s,'source':rel(f),'kind':'fold','fold':num})
                foldreports.append(d); ev.append(rel(f))
        s=score(report)
        if s and 'oof_scores' in report:
            fs=report.get('folds',[])
            complete=len(fs)==5 or len(foldreports)==5 or js(p/'verification.json').get('rows')==723556
            results.insert(0,{'scope':'完整五折 OOF' if complete else '部分折汇总','value':s['TOTAL'],'scores':s,'source':rel(p/'report.json'),'kind':'oof' if complete else 'partial'})
            ev.append(rel(p/'report.json'))
        for fn in ['router_report.json','ensemble_verification.json','screening_report.json','grid_report.json','result.json']:
            d=js(p/fn);s=score(d)
            if s and not results:
                num=d.get('screening_fold',cfg.get('screening_fold',None))
                if not num and 'fold5' in ident:num=5
                if not num:num=1
                results.append({'scope':f'第{num}折','value':s['TOTAL'],'scores':s,'source':rel(p/fn),'kind':'fold','fold':num});ev.append(rel(p/fn))
        histories=list(p.glob('fold_*_validation_history.json'))
        for f in histories:
            num=int(re.search(r'fold_(\d+)',f.name)[1]);h=js(f)
            if isinstance(h,list) and h and not any(a.get('fold')==num for a in results):
                hh=[v for v in h if isinstance(v,dict) and isinstance(v.get('official_total'),(int,float))]
                if hh:
                    b=max(hh,key=lambda x:x['official_total'])
                    results.append({'scope':f'第{num}折 · 观测最佳','value':b['official_total'],'scores':b.get('official_scores',{}),'source':rel(f),'kind':'observed','fold':num,'epoch':b['epoch'],'last_epoch':hh[-1]['epoch']});ev.append(rel(f))
        ver=js(p/'screening_verification.json')
        if not results:
            for key,num in [('screening_fold_score',ver.get('screening_fold',5)),('first_fold_score',1)]:
                if isinstance(ver.get(key),(int,float)):
                    results.append({'scope':f'第{num}折','value':ver[key],'source':rel(p/'screening_verification.json'),'kind':'fold','fold':num});ev.append(rel(p/'screening_verification.json'));break
        if not results and isinstance(stop.get('best_score'),(int,float)):
            num=cfg.get('screening_fold',5)
            results.append({'scope':f'第{num}折 · 提前停止','value':stop['best_score'],'kind':'observed','fold':num,'source':rel(p/'manual_early_stop.json')})
        text=read(p/'README.md')
        snippets=[]
        for line in text.splitlines():
            if len(line)>15 and len(line)<1800 and not line.startswith('|') and re.search(r'结论|不值得|收益|负结果|negative|no gain|not worth|reason|Final result|below',line,re.I):snippets.append(line)
        excerpt='\n'.join(snippets[-6:])[:5500] or text[:1600]
        cfg_src=cfg.get('dependency_sha256',{})
        paths=[ROOT / k.replace('\\','/') for k in cfg_src]
        stem=ident.split('/')[-1].removeprefix('pure_dl_')
        model=SRC/(stem+'.py')
        trainer=SRC/('train_'+stem+'.py')
        if not trainer.exists():
            candidates=[x for x in paths if x.name.startswith('train_') and x.exists()]
            trainer=max(candidates,key=lambda x:len(set(x.stem.split('_'))&set(stem.split('_'))),default=trainer)
        if not model.exists() and trainer.exists():model=SRC/((module_network(trainer) or '')+'.py')
        parentmodule=module_network(model) if model.exists() else None
        mdoc=''
        if model.exists():
            try:mdoc=ast.get_docstring(ast.parse(read(model))) or ''
            except SyntaxError:pass
        status='recorded'
        if stop:status='stopped'
        if not results:status='unscored'
        if runtime.get('status')=='running' and runtime.get('atlas_current_trial') and not foldreports:status='running_snapshot'
        primary=next((v for v in results if v['kind']=='oof'),None) or next((v for v in results if v.get('fold')==5),None) or (results[0] if results else None)
        for filename in ['README.md','config.json','screening_verification.json','manual_early_stop.json']:
            if (p/filename).exists():ev.append(rel(p/filename))
        if model.exists():ev.append(rel(model))
        display=stem.removesuffix('_sequence')
        display=curated.get('terms',{}).get(display,display.replace('_',' '))
        n={'id':ident,'title':display,'stage':stage(ident),'status':status,'primary':primary,'results':results,'summary':mdoc or cfg.get('architecture','留存记录待进一步解读。'),'conclusion':stop.get('reason',''),'excerpt':excerpt,'config':compact_cfg(cfg),'evidence':list(dict.fromkeys(ev)),'parentModule':parentmodule,'model':model.stem if model.exists() else None,'highlight':False}
        n.update(overrides.get(ident,{}));nodes.append(n)
        if ident=='pure_dl_posterior_validation_grid_fold5' and n['primary']:
            n['primary']['scope']='第五折 · 验证调参诊断';n['primary']['kind']='diagnostic'
        n['_ensemble']=cfg.get('ensemble',{})
    # Explicit opt-in records may live below the scanned output depth or in results/.
    discovered = {n['id'] for n in nodes}
    for ident, annotation in overrides.items():
        if ident in discovered or not annotation.get('virtual'):
            continue
        evidence = annotation.get('evidence', [])
        if not evidence or not all((ROOT / p).is_file() for p in evidence):
            raise ValueError(f'Curated virtual node lacks existing evidence: {ident}')
        n = dict(id=ident, title=ident, stage='backbone', status='unscored',
                 primary=None, results=[], summary='', conclusion='', excerpt='',
                 config={}, evidence=evidence, parentModule=None, model=None,
                 highlight=False, _ensemble={})
        n.update(annotation)
        nodes.append(n)
    # User cleanup retains summaries/results, including scores in this prior snapshot.
    # Keep historical nested attempts even after their empty artifact directory is gone.
    existing={n['id']:n for n in nodes}
    for old in previous.get('nodes',[]):
        if old['id'].startswith('dead:'):continue
        candidate=OUT/old['id'];summary=None
        while candidate.is_relative_to(OUT) and candidate!=OUT:
            if (candidate/'SUMMARY_RESULTS.json').is_file():summary=candidate/'SUMMARY_RESULTS.json';break
            candidate=candidate.parent
        if summary is None or not js(summary).get('cleanup_completed'):continue
        if old['id'] in existing:
            n=existing[old['id']]
            if not n.get('primary') and old.get('primary'):
                n['primary']=old['primary'];n['results']=old.get('results',[])
        else:
            n=dict(old);n['_ensemble']={};nodes.append(n);existing[n['id']]=n
        n['artifacts_compacted']=True
        n['summary'] += ' 历史记录已按用户要求压缩；保留摘要和分数，权重/缓存/逐轮日志已删除，不是本次重载验证。'
        n['evidence']=list(dict.fromkeys([e for e in n.get('evidence',[]) if (ROOT/e).is_file()]+[rel(summary)]))
        for record in n.get('results',[]):
            if record.get('source') and not (ROOT/record['source']).is_file():
                record['original_source']=record['source'];record['source']=rel(summary)
        if n.get('primary') and n['primary'].get('source') and not (ROOT/n['primary']['source']).is_file():
            n['primary']['original_source']=n['primary']['source'];n['primary']['source']=rel(summary)
    by={n['id']:n for n in nodes};edges=[]
    # Explicit curated relationships distinguish experimental ancestry from literal class inheritance.
    for e in curated.get('edges',[]):
        if e['source'] in by and e['target'] in by:edges.append(e)
    for n in nodes:
        def strings(v):
            if isinstance(v,str): yield v
            elif isinstance(v,dict):
                for q in v.values(): yield from strings(q)
            elif isinstance(v,list):
                for q in v: yield from strings(q)
        for s in set(strings(n.pop('_ensemble'))):
            s=s.replace('\\','/').split('outputs/')[-1].strip('/')
            if s in by and s!=n['id'] and not any(e['source']==s and e['target']==n['id'] for e in edges):
                edges.append({'source':s,'target':n['id'],'type':'fusion','evidence':next((p for p in n['evidence'] if p.endswith('report.json')), ''),'note':'报告 config.ensemble 明确列出的融合输入；权重见留存配置。'})
    manualtargets={e['target'] for e in edges}
    modulemap={}
    for n in nodes:
        if n['model'] and '/' not in n['id']:
            expected='pure_dl_'+n['model']
            if n['id']==expected or n['model'] not in modulemap:modulemap[n['model']]=n['id']
    # Multiple experiments reuse this architecture; the original router is not its compact-cache control.
    modulemap['paired_posterior_router_sequence']='pure_dl_paired_posterior_router80_sequence'
    for n in nodes:
        if n['id'] in manualtargets:continue
        parent=modulemap.get(n['parentModule'])
        if not parent:
            key='pure_dl_'+(n['parentModule'] or '')
            if key in by:parent=key
        if parent and parent!=n['id']:
            edges.append({'source':parent,'target':n['id'],'type':'code','evidence':next((s for s in n['evidence'] if s.endswith('.py')), ''),'note':'源码 Network 导入关系；表示结构继承，不保证训练配置完全相同。'})
        elif '/' in n['id'] and n['id'].split('/')[0] in by:
            edges.append({'source':n['id'].split('/')[0],'target':n['id'],'type':'inferred','note':'按子实验目录推定；非已核验的代码继承。'})
    # No forced parent if history cannot establish one.
    for n in nodes:
        if not n.get('conclusion'):
            if n['status']=='running_snapshot':n['conclusion']='训练快照，当前分数仅为观测最佳；尚不能判断最终收益。'
            elif n['status']=='stopped':n['conclusion']='实验提前结束；保留观测结果，不视为完整预算或恢复核验完成。'
            elif n['primary']:n['conclusion']='已有留存结果；尚未人工确认匹配对照，不能仅凭分数认定增益或 dead end。'
            else:n['conclusion']='暂无可提取的训练评分；可能是诊断、准备工作或记录不完整，不等同于失败。'
    for d in curated.get('deadends',[]):
        related=[by[x] for x in d['parents'] if x in by]
        vals=[x['primary'] for x in related if x['primary']]
        n={'id':d['id'],'title':d['title'],'stage':d['stage'],'status':'deadend','primary':None,'results':vals,'summary':d['summary'],'conclusion':d['reason'],'excerpt':'','config':{},'evidence':list(dict.fromkeys(s for x in related for s in x['evidence'] if s.endswith('README.md'))),'highlight':True,'resultText':d.get('resultText','；'.join(f"{x['scope']} {x['value']:.4f}" for x in vals[:3]))}
        nodes.append(n)
        for p in d['parents']:
            if p in by:edges.append({'source':p,'target':d['id'],'type':'closure','note':d['reason']})
    incoming={e['target'] for e in edges}
    manifest={'builtAt':datetime.now().astimezone().isoformat(timespec='seconds'),'source':'本地 outputs 实验记录 + 训练源码 + 人工标注；静态快照','nodes':nodes,'edges':edges,'stages':[{'id':a,'title':b,'description':c} for a,b,c in STAGES],'coverage':{'experimentDirectories':len(dirs),'experiments':len(by),'deadEnds':len(nodes)-len(by),'unresolvedParents':sum(n['id'] not in incoming for n in nodes if n['id']!='simple_cnn_cv'),'excluded':excluded,'scope':'覆盖当前留存的 pure_dl_* 实验目录、可识别的子配置和三个早期 CNN 基线；已被删除且无留存记录的历史无法自动恢复。无父节点表示证据不足，不以时间相邻冒充血缘。'}}
    (HERE/'atlas-data.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    data=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    template=read(HERE/'template.html')
    if template:(HERE/'index.html').write_text(template.replace('__ATLAS_DATA__',data),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest['coverage'].items() if k!='scope'},ensure_ascii=False))

if __name__=='__main__':main()
