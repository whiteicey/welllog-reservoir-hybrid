"""Finalize evidence for the completed two-arm study; no training or model changes."""
from pathlib import Path
from datetime import datetime
import json,sys
from prepare_dual8222_query_v7 import trainers

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_component_query_v7_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    progress=json.loads((STUDY/'progress.json').read_text(encoding='utf-8'))
    proof=json.loads((STUDY/'implementation_and_reload.json').read_text(encoding='utf-8'))
    recipe=json.loads((STUDY/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    assert progress['all_complete'] and not trainers()
    baseline=progress['source_scores']['TOTAL']
    assert not recipe['new_v7_model_adopted'] and all(r['final_report']['best_epoch']==0 for r in progress['runs'])
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# 分量查询预测头v7：最终结论','',now,'',
           '**两组均完成18轮并恢复第0轮，未取得新增精度。当前保留80.274135337791的v5预测器，未替换交付。**','',
           '## 正式训练结果','',
           '|方案|可训练参数|更新后最高TOTAL（不含第0轮）|对应新增轮次|相对原模型|最终最佳/停止轮次|',
           '|---|---:|---:|---:|---:|---|']
    for r in progress['runs']:
        peak=r['trained_peak'];final=r['final_report']
        lines.append(f"|{r['mode']}|{r['trainable_parameters']}|{peak['TOTAL']:.9f}|{peak['epoch']}|{peak['TOTAL']-baseline:+.9f}|{final['best_epoch']}/{final['stopped_epoch']}|")
    lines += ['', '两个最终报告均为80.274135，是最佳选择回到原函数的结果，不是新query结构的训练成绩。最终预测相同，因此对应差值/重采样机械地为零，不构成两种训练方法等效的统计证明。',
        '', '## 本次结构','每位专家8个混合分量，共16个查询。查询融合32维专家隐藏表示、分量身份及当前预测的位置/log标准差/log权重。2头交叉注意力从34个记忆槽位读信息，按分量生成有界位置修正。',
        'center与local的模块均14081参数，初始化逐张量相同；前者把当前点特征复制到槽位，后者读取前后8点的真实隐藏特征。相对位置、掩码、分量描述、学习率和预算一致。',
        '原预测器包括线性位置头全部冻结，只有新模块更新。本轮较v6同时改变了更新范围和结构，不能通过跨轮差异单独归因为分量query。本批只在两组内部比较邻近隐藏信息的作用。',
        '', '## 已验证内容','初始144665点逐元素复现；梯度仅进入新模块；最终参数恢复和全部冻结参数一致性检查通过。PERM/SW每轮预测保持不变。',
        '已用最终训练状态检查模块，避免在第0轮零输出上做无效检查。全padding输出有限且为零、有效query不受padding内容影响、半径8限制均通过。center改变相邻位置特征不影响当前输出，local对相邻变化有非零反应。',
        '查询投影、分量身份向量、状态投影与输出投影相对初始化均发生变化。参数更新与局部敏感性证明通路有实际运行，不证明新井泛化有效。',
        '没有胜出候选，不重复加载原v5做全量重载；沿用已存在的v5磁盘重载证据。本次各训练器的最终恢复已实际执行。',
        '', '## 结论与下一步边界','当前证据未支持：完全冻结已训练双专家，追加这套分量query模块并只修正POR位置，可以提高开发集分数。它不排除其他监督、完整分布参数或其他通道的query设计。',
        '建议下一次改变一个尚未被充分检验的限制：用相同读出结构比较POR位置限定输出与完整条件密度输出，并保留训练目标不变的对照；或单独检验监督效用对齐。两条不可同时修改后直接归因。上述都尚未实现或启动。',
        '本批不追加半径/头数/宽度/LR扫描；当前第五折严格>80.5、完整原分组OOF>82仍未达到。完整OOF已知最好81.713233；本次没有其他折、旧ABD、测试伪标签或新交付。',
        '', '## 证据与产物','- `progress.json`：实际状态、初始/最终报告、训练后最高分、原样本重算。',
        '- `implementation_and_reload.json`：初始化相同、参数量、查询更新、遮挡与局部读取检查。',
        '- `CURRENT_RECIPE.json`：继续引用旧v5，`new_v7_model_adopted=false`。',
        '- `METHOD.md`：完整结构、符号形状与比较条件。',
        '- `trainer_derivation.json`：新训练器的修改记录，旧训练器保持原样。',
        '- `outputs/pure_dl_dual8222_v7_query_{center,local}80_fold5/`：独立记录、代码配置、自然早停状态、最佳和当前权重。',
        '', '## 时间口径',json.dumps(progress['timing'],ensure_ascii=False,indent=2),
        '两项并发启动，未超过2GPU限制。该范围不包括后续诊断，不作为部署推理速度对照。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    status='# 项目进度\n\n目标：验证分量查询读出在原双专家上的增量。\n\n已完成：两项正式训练、同参数/初始化确认、最终恢复、旧参数与其他通道冻结验证、查询更新与局部遮挡诊断；均未超越80.274135，原模型保持。\n\n未完成：独立泛化、完整OOF>82、中断续跑等价性；完整分布query/监督对齐尚未实现。\n\n后续计划：不追加本批候选；下一次选择明确机制并固定对照，避免同时改变输出范围与损失。\n\n达成目标后：固定配方验证原分组泛化；当前不补其他折，不改旧交付。\n'
    (STUDY/'PROJECTS.md').write_text(status,encoding='utf-8')
    summary='\n\n## '+now+' 分量query v7完成\n两组均18轮自然早停并恢复第0轮80.274135；训练后最高center80.270925285299@9、local80.271695049988@8，无新增收益。原预测器全冻结；两组14081参数/初始化一致，query真实更新，局部与mask检查通过。保留旧v5，不重复旧权重重载，无旧任务/补折/交付变更；完整分布query和损失对齐未启动。详见results/deep_learning/dual8222_component_query_v7_20260928/FINAL.md。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ATLAS/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(summary)
    cur=json.loads((ATLAS/'curation.json').read_text(encoding='utf-8'))
    for r in progress['runs']:
        folder=ROOT/'outputs'/r['id']
        for fn in ['PROJECTS.md','README.md']:
            with (folder/fn).open('a',encoding='utf-8') as f:f.write(summary)
        cur['nodes'][r['id']]['conclusion']=f"完成18轮，best_epoch0，恢复原80.274135；训练后最高{r['trained_peak']['TOTAL']:.9f}@{r['trained_peak']['epoch']}。没有新增收益；原预测器保持，未补折。"
    dead='dead:dual8222_v7_component_query_location'
    cur['deadends']=[v for v in cur.get('deadends',[]) if v['id']!=dead]
    cur['deadends'].append({'id':dead,'title':'分量query位置修正：本轮无收益','stage':'loss',
        'parents':[r['id'] for r in progress['runs']],
        'summary':'同14081参数和初始化的center/local记忆，旧80.274135预测器全冻结，两组都恢复第0轮。',
        'reason':'不继续本设置附近的半径/宽度扫描；不泛化至完整分布头、不同监督或所有attention。参数更新与mask已检查。'})
    (ATLAS/'curation.json').write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'all_complete':True,'new_model_adopted':False,'retained_scores':recipe['scores'],
                      'trained_peaks':{r['mode']:r['trained_peak'] for r in progress['runs']},
                      'active_trainers':trainers(),'report':str(STUDY/'FINAL.md')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
