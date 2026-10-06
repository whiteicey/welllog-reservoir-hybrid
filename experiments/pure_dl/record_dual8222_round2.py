"""Record current/final readout evidence and extend the offline atlas, no training."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'docs/deep_learning/atlas'
STUDY=ROOT/'results/deep_learning/dual8222_round2_20260928'
PARENT='pure_dl_paired_local_context_router80_sequence'
ARMS=[
 ('linear','pure_dl_dual8222_v2_linear80_fold5','双专家·独立线性头对照','只微调原线性行，19350参数；检验前轮独立训练收益。'),
 ('cross80','pure_dl_dual8222_v2_cross8080_fold5','双专家·目标专属联合读出','每目标联合两位专家256维特征，零输出残差，93180参数。'),
 ('gate_only','pure_dl_dual8222_v3_gate_only80_fold5','双专家·仅门控因子更新','原线性密度参数固定，仅774门控参数更新；原完整双专家似然。'),
 ('density_only','pure_dl_dual8222_v3_density_only80_fold5','双专家·仅密度因子更新','原门控参数固定，仅18576密度参数更新；效用默认决策仍可变化。'),
]


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def replace_section(path,marker,text):
    original=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    original=original.split(marker)[0].rstrip()
    path.write_text(original+'\n\n'+marker+'\n'+text+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    curation=read(HERE/'curation.json')
    parent=read(ROOT/'outputs'/PARENT/'fold_5_report.json')
    records=[]
    for arm,run_id,title,summary in ARMS:
        folder=ROOT/'outputs'/run_id
        if not folder.exists():continue
        status=read(folder/'runtime_status.json',{})
        final=read(folder/'fold_5_report.json')
        history=read(folder/'fold_5_validation_history.json',[])
        initial=read(folder/'initial_verification.json',{})
        last=history[-1] if history else {}
        alive=False
        try:
            p=psutil.Process(status.get('pid',-1));cmd=' '.join(p.cmdline())
            alive=p.is_running() and ('train_dual8222_heads_v2.py' in cmd or 'train_dual8222_factors80.py' in cmd)
        except (psutil.NoSuchProcess,psutil.AccessDenied,ValueError):pass
        if final:
            scores=final['scores'];scope='最终固定三头共同恢复已验证';best_epochs=final['best_epochs']
        elif last:
            scores=dict(zip(['POR','PERM','SW'],last['best_channel_scores']))
            scores['TOTAL']=last['best_assembled_total'];scope='运行快照候选，非最终恢复';best_epochs=last['best_epochs']
        else:
            scores=None;scope='尚无训练评分';best_epochs=[]
        description=f"{status.get('status','pending')}；{scope}；最后完整轮次{last.get('epoch',0)}。"
        if scores:
            description+=f"TOTAL={scores['TOTAL']:.9f}，源模型差{scores['TOTAL']-parent['scores']['TOTAL']:+.9f}，最佳轮次{best_epochs}。"
        description+='原单覆盖评分，不补折；完整OOF>82未完成。'
        curation['nodes'][run_id]={'title':title,'summary':summary,'conclusion':description,'stage':'loss','highlight':True}
        log=(folder/'train.log').read_text(encoding='utf-8',errors='replace')
        record={'arm':arm,'id':run_id,'runtime':status,'process_alive':alive,'scores':scores,'scope':scope,
            'final_report':final,'last_completed_epoch':last.get('epoch',0),'best_epochs':best_epochs,
            'initial_verification':initial,'gradient_scope_verified':'GRADIENT_SCOPE_VERIFIED' in log,
            'failure':read(folder/'failure.json'),'last_state_exists':(folder/'last_state.pt').exists()}
        records.append(record)
        body=description+'\n初始差异元素数：'+str(initial.get('different_elements'))+'；首批梯度范围验证：'+str(record['gradient_scope_verified'])+'。\n'
        body+='已保存训练状态不等于已验证中断续跑等价性。本次没有重新启动历史任务或剩余折。\n'
        for name in ['README.md','PROJECTS.md']:
            replace_section(folder/name,'## 本轮最新核对',body)
    edges=[
        (PARENT,ARMS[0][1],'evolution','补只微调线性行的独立目标对照。'),
        (ARMS[0][1],ARMS[1][1],'control','同起点/损失/预算；增加两个内部专家特征的联合读出。'),
        ('pure_dl_dual8222_heads_pointwise80_fold5',ARMS[1][1],'control','近参数预算，分开读出与联合256维读出。'),
        (ARMS[0][1],ARMS[2][1],'control','仅门控参数更新，其他配置同独立线性头。'),
        (ARMS[0][1],ARMS[3][1],'control','仅密度参数更新，其他配置同独立线性头。'),
    ]
    pairs={(a,b) for a,b,_,_ in edges}
    curation['edges']=[e for e in curation['edges'] if (e['source'],e['target']) not in pairs]
    curation['edges'] += [{'source':a,'target':b,'type':t,'note':n,'evidence':'outputs/'+b+'/README.md'} for a,b,t,n in edges]
    (HERE/'curation.json').write_text(json.dumps(curation,ensure_ascii=False,indent=2),encoding='utf-8')
    payload={'observed_at':now,'source_scores':parent['scores'],'runs':records,'remaining_folds_started':False,
        'full_oof_best':81.71323318479078,'full_oof_goal_met':False,
        'atlas_extension':'record_dual8222_round2.py; run after existing curation generators, before build_atlas.py'}
    (STUDY/'progress.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['观察时间：'+now,'','|方法|状态|TOTAL|最佳轮次|','|---|---|---:|---|']
    for r in records:
        value=f"{r['scores']['TOTAL']:.9f}" if r['scores'] else '尚无'
        lines.append('|'+r['arm']+'|'+r['scope']+'|'+value+'|'+str(r['best_epochs'])+'|')
    lines+=['','已完成：新增线性归因/联合内部专家读出/因素分离实现与实际运行记录。',
            '待完成：所有候选的最终恢复、逐通道配方固定后的共同推理；不存在完整OOF增益证明。',
            '目标：当前双专家可复现优化，第五折>80.5且明确收益；完整OOF>82仍未达成。',
            '达成后：固定方案后评估泛化，当前不自动补折、不改旧交付。',
            '图谱在既有curation生成后运行本脚本再build_atlas；不修改此前被保护的curate.py。']
    replace_section(STUDY/'README.md','## 当前结果', '\n'.join(lines))
    replace_section(STUDY/'PROJECTS.md','## 当前结果','\n'.join(lines))
    if not (STUDY/'AGENTS.md').exists():
        (STUDY/'AGENTS.md').write_text('# 分析记录约束\n目标：记录独立头各项对照和固定组合的可复现结果。维护PROJECTS.md，逐批记载已做、计划、未做、目标和后续工作。分析脚本不得启动训练或用标签逐点/逐井选择模型；原数据与交付不改。\n',encoding='utf-8')
    summary='\n\n## '+now+' 双专家继续优化结果核对\n'+'\n'.join(lines[2:])+'\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',HERE/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(summary)
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
