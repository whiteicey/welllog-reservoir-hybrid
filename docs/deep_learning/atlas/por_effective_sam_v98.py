"""Keep a prepared or trained optimizer ablation distinct from its parent score."""
import json


def annotate(root, node, edge, end):
    study = root/'results/deep_learning/por_effective_sam_v98_20260930'
    if not (study/'PLAN.json').exists():
        return
    out = root/'outputs/pure_dl_por_effective_sam_v98_POR_fold5'
    report = json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    runtime = json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else None
    status = 'recorded' if report else ({'running': 'running_snapshot', 'failed': 'failed',
                                       'stopped_by_user': 'stopped'}.get(runtime.get('status'), 'prepared')
                                      if runtime else 'prepared')
    ident = 'pure_dl_por_effective_sam_v98_POR_fold5'
    conclusion = '准备未训练，候选尚无成绩；父配方80.504197不是本候选结果。'
    if report:
        conclusion = '两项已结束；同容量SAM−AdamW TOTAL差=' + str(report['predefined_ablation_delta_TOTAL']) + '，仅原第五折证据，不是完整OOF。'
    elif runtime:
        conclusion = '实际状态=' + str(runtime.get('status')) + '；独立选模/早停，完成后恢复头并实际组合。'
    node(ident, 'v98 POR有效模块 AdamW/SAM',
         '从v31 anchored当前POR完整函数热启动；只现有路由和幅值位置模块训练。同容量/初始化/抽样，own POR损失，其他目标固定验证。80/18，rho0.05，无新增骨干或自动补折。',
         conclusion, stage='optimization', status=status,
         evidence=['results/deep_learning/por_effective_sam_v98_20260930/PLAN.json',
                   'results/deep_learning/por_effective_sam_v98_20260930/README.md'],
         resultText=conclusion)
    edge('pure_dl_mechanism_v31_POR_fold5', ident, 'evolution', '加载所选anchored完整head；仅旧有效模块再优化')
    edge('pure_dl_sw_factor_sam_v96_SW_likelihood_fold5', ident, 'comparison', '固定v96其他目标验证参考；不加载SW预测进POR训练')
    for mode in ('adamw', 'sam'):
        arm = next((r for r in report['arms'] if r['mode'] == mode), None) if report else None
        shown = dict(resultText='未训练，尚无成绩')
        if arm:
            result = dict(value=arm['reference_replacement_total'],
                          scores={'POR': arm['target_score'], 'TOTAL': arm['reference_replacement_total']},
                          kind='fifth', fold=5, scope='原第五折 · 固定其他通道实际组合',
                          source='outputs/pure_dl_por_effective_sam_v98_POR_fold5/'+mode+'/fold_5_report.json')
            shown = dict(primary=result, results=[result])
        elif runtime:
            shown['resultText'] = '状态=' + str(runtime.get('status')) + '；候选快照=' + str(runtime.get('modes', {}).get(mode))
        child = ident+'/'+mode
        node(child, 'v98 POR '+mode, '同容量既有效路由/位置参数；仅优化器方法不同。',
             '单折、独立目标checkpoint选择；初始函数不算新收益。', stage='optimization',
             status='recorded' if arm else status, **shown)
        edge(ident, child, 'comparison', '预定义同容量优化器对照')
    if report and not any(a['positive_gain_and_gate'] for a in report['arms']):
        end('dead:v98_por_effective_optimizer', 'v98 当前POR路由/位置联合再优化无收益',
            'optimization', [ident+'/'+m for m in ('adamw', 'sam')],
            '两组43635参数、同初始函数与抽样；18轮停止并恢复epoch0。更新后最佳TOTAL分别80.502927440098/80.503227187203，低于80.504197424477；选定SAM减AdamW=0。',
            '关闭当前热启动、更新范围、学习率3e-5与SAM半径0.05组合；不扫描其邻域，不泛化为所有SAM或POR优化无效。')
