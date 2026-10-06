"""Read single-target results and update research records, never starts a trainer."""
from pathlib import Path
from datetime import datetime
import argparse,json,sys
import numpy as np
import psutil
from launch_independent_channels_v8 import ROOT,STUDY,RUNS,PREFIX

REFERENCE='pure_dl_dual8222_v5_location_point3280_fold5'
ATLAS=ROOT/'docs/deep_learning/atlas'


def read(p,default=None):
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default


def save(p,obj):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def section(p,text):
    marker='## 独立模型实际结果'
    old=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    p.write_text(old.split(marker)[0].rstrip()+'\n\n'+marker+'\n'+text+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--brief',action='store_true');a=ap.parse_args()
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    original=read(ROOT/'outputs'/REFERENCE/'fold_5_report.json')['scores']
    cur=read(ATLAS/'curation.json');records=[]
    for key,(target,arch) in RUNS.items():
        folder=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        runtime=read(folder/'runtime_status.json',{});history=read(folder/'validation_history.json',[])
        final=read(folder/'fold_5_report.json');initial=read(folder/'initial_verification.json');cfg=read(folder/'config.json',{})
        alive=False
        try:
            p=psutil.Process(runtime.get('pid',-1));alive=p.is_running() and 'train_independent_channels_v8.py' in ' '.join(p.cmdline())
        except(psutil.NoSuchProcess,psutil.AccessDenied,ValueError):pass
        score=final['target_score'] if final else (history[-1]['best_target_score'] if history else None)
        row={'key':key,'target':target,'architecture':arch,'id':folder.name,'status':runtime,'process_alive':alive,
             'target_score':score,'delta_shared_reference':score-original[target] if score is not None else None,
             'latest':history[-1] if history else {},'initial':initial,'final_report':final,
             'trainable_parameters':cfg.get('trainable_parameters'),'gradient_scope':read(folder/'gradient_scope.json'),
             'failure':read(folder/'failure.json'),'startup_failure_preserved':(folder/'startup_failure_attempt1.json').exists()}
        records.append(row)
        state='完成并恢复' if final else runtime.get('status','planned')
        text=f"{now}；{target}/{arch}；{state}；{target}通道分={score}；相对旧参考={row['delta_shared_reference']}。"
        text+='\n单项目标分数不能冒称三通道TOTAL。完整模型有独立骨干和检查点；实际梯度='+str(row['gradient_scope'])+'。'
        if not final:text+='\n当前分数仅为运行记录，未最终恢复。'
        if row['failure']:text+='\n实际失败='+str(row['failure'])
        if a.record:
            for name in ['README.md','PROJECTS.md']:section(folder/name,text)
            cur['nodes'][folder.name]={'title':target+'独立模型·'+('双视图' if arch=='dual' else '单视图'),
                'summary':'端到端单任务私有骨干，只有'+target+'损失，预训练初始化；单目标成绩不与TOTAL混排。',
                'conclusion':text+'\n未补其他折。','stage':'loss','highlight':True}
    complete=all(r['final_report'] is not None for r in records)
    choices=[]
    for target in ('POR','PERM','SW'):
        candidates=[r for r in records if r['target']==target and r['final_report']]
        if candidates:
            best=max(candidates,key=lambda r:(r['target_score'],-r['trainable_parameters']))
            choices.append({'target':target,'key':best['key'],'target_score':best['target_score'],'delta_reference':best['delta_shared_reference']})
    assembly=read(STUDY/'assembled_report.json')
    payload={'observed_at':now,'reference_scores':original,'runs':records,'all_six_complete':complete,'best_completed_per_target':choices,
             'assembled_report':assembly,'limitations':'Original fifth is development data repeatedly used for selection; per-target candidates initialize from same-fold multitarget pretraining; no newfullOOF evidence.'}
    save(STUDY/'progress.json',payload)
    if a.record:
        pairs={(REFERENCE,r['id']) for r in records}
        cur['edges']=[e for e in cur['edges'] if (e['source'],e['target']) not in pairs]
        cur['edges'] += [{'source':REFERENCE,'target':r['id'],'type':'evolution','note':'复制预训练起点后，完整私有骨干单目标训练；单视图与双视图起点不同。','evidence':'outputs/'+r['id']+'/README.md'} for r in records]
        save(ATLAS/'curation.json',cur)
        lines=['# 三通道独立模型：实际结果','',now,'','|目标|结构|状态|通道分|相对旧参考|最佳/停止轮次|','|---|---|---|---:|---:|---|']
        for r in records:
            f=r['final_report'];score=r['target_score']
            lines.append('|'+r['target']+'|'+r['architecture']+'|'+('完成' if f else r['status'].get('status','planned'))+'|'+(f'{score:.9f}' if score is not None else '尚无')+'|'+(f"{r['delta_shared_reference']:+.9f}" if score is not None else '—')+'|'+(str(f['best_epoch'])+'/'+str(f['stopped_epoch']) if f else '—')+'|')
        lines+=['','## 当前选择（只使用已完成候选）',json.dumps(choices,ensure_ascii=False,indent=2),'',
            '六项全部完成='+str(complete)+'；独立三模型共同重载='+str(assembly is not None)+'。',
            '每个目标只用自身监督、私有模型与优化器。初始化来自既有多任务预训练；当前骨干允许更新。不同目标不共享live参数或训练中间特征。',
            '单通道替换参考TOTAL只用于诊断；只有固定三模型真实预测合并后才报告本轮完整TOTAL。',
            'single为一条完整特征骨干，dual为完整+校准视图两条私有骨干，各通道允许选择不同结构。没有额外宽度/LR搜索。',
            '独立模型相对旧共享模型也增加了更新范围和总训练预算；不能把总收益单独归因为梯度冲突消除。',
            '初次POR/SW启动漏sys导入、未进入训练；已修复并保留首轮失败日志。其后真实状态以各run记录为准。',
            '保持原9曲线/206特征、原分折/144665评分人口、Q和单覆盖解码；不自动补折，不覆盖旧交付。完整OOF>82尚未达成。']
        if assembly:lines+=['','## 真实三模型结果',json.dumps(assembly,ensure_ascii=False,indent=2)]
        (STUDY/('FINAL.md' if complete and assembly else 'SNAPSHOT.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        body='观察时间：'+now+'。已完成'+str(sum(r['final_report'] is not None for r in records))+'/6项。\n已做：模型拆分和显式启动，结果如本目录progress.json。\n计划：完成有限六项并固定三模型共同重载。\n未做：完整OOF和独立泛化、续跑等价性。目标达成后验证固定配方，当前无补折。'
        section(STUDY/'PROJECTS.md',body)
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ATLAS/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write('\n\n## '+now+' 独立通道v8\n'+body+'\n')
    small={**payload,'runs':[{k:v for k,v in r.items() if k!='final_report'} for r in records]}
    if a.brief:
        small={'observed_at':now,'all_six_complete':complete,'runs':[{'key':r['key'],'status':r['status'].get('status'),'alive':r['process_alive'],'epoch':r['latest'].get('epoch'),
            'current_score':r['latest'].get('target_score'),'best_score':r['target_score'],'best_epoch':r['latest'].get('best_epoch'),'bad_epochs':r['latest'].get('bad_epochs'),
            'initial_score':r['initial']['target_score'] if r['initial'] else None,'delta_shared_reference':r['delta_shared_reference'],
            'failure':r['failure']['error'] if r['failure'] else None} for r in records],'selected':choices,'assembly_done':assembly is not None}
    print(json.dumps(small,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
