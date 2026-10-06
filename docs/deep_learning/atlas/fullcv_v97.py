"""Editorial annotations for the explicitly resumed fixed-recipe fullCV."""
import json


def annotate(root,node,edge):
    study=root/'results/deep_learning/fullcv_v97_20260930'
    if not (study/'batch_snapshot.json').exists():return
    snapshot=json.loads((study/'batch_snapshot.json').read_text(encoding='utf-8'))
    overlapping=(study/'controller_takeover_done.json').exists()
    execution='主阶段依序执行，与其他折来源预训练重叠；最多2GPU、最多1SW，原子last' if overlapping else '串行/原子last'
    for item in snapshot['folds']:
        f=item['fold'];ident=f'pure_dl_fullcv_v97/fold_{f}';assembled=item.get('assembled');shown={}
        if assembled:
            result=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fold',fold=f,
                scope='原第'+str(f)+'折 · 实际独立三通道',source='outputs/pure_dl_fullcv_v97/fold_'+str(f)+'/fold_report.json')
            shown=dict(primary=result,results=[result]);conclusion='固定配方本折已实际组合；完整OOF另行汇总，单折不代表目标完成。'
        else:
            shown['resultText']='来源骨干最近轮='+str(item.get('root_last_epoch'))+'，最佳TOTAL='+str(item.get('root_best_total'))+'；有效阶段='+str(len(item.get('completed_stages',[])))+'/18，最终三通道尚无成绩'
            conclusion='明确恢复其余折；本折64训练井重训全部来源，分折不改。来源成绩仅是依赖快照，失败须显式恢复。'
        node(ident,'v97 原第'+str(f)+'折固定配方补全','v96第五80.504197固定配方，18有效阶段加local-context80来源；独立POR/PERM/SW，训练统计/Q和80/18保持，原分折不改。v8 epoch0私有提取经权重证明固定复用，负对照不重跑；'+execution+'。',conclusion,stage='validation',status='recorded' if assembled else 'running_snapshot',highlight=bool(assembled),evidence=['results/deep_learning/fullcv_v97_20260930/authorization.json','results/deep_learning/fullcv_v97_20260930/PROGRESS.md'],**shown)
        edge('pure_dl_sw_factor_sam_v96_SW_likelihood_fold5',ident,'evolution','固定全部有效机制，其他折重新训练，绝不加载第五训练权重')
    final=json.loads((study/'FINAL_REPORT.json').read_text()) if (study/'FINAL_REPORT.json').exists() else None
    shown={}
    if final:
        r=dict(value=final['oof_scores']['TOTAL'],scores=final['oof_scores'],kind='oof',scope='完整原五折OOF',source='results/deep_learning/fullcv_v97_20260930/FINAL_REPORT.json')
        shown=dict(primary=r,results=[r])
    else:shown['resultText']='完整OOF尚未完成，已核验第五80.504197'
    node('pure_dl_fullcv_v97','v97 固定配方补齐原五折','明确恢复原其余四折；第五80.504197核验复用，另外四折各自重训完整来源/有效模块。','独立折进度与原行序完整OOF分开；来源骨干不等于最终三通道。',stage='validation',status='recorded' if final else 'running_snapshot',evidence=['results/deep_learning/fullcv_v97_20260930/PROGRESS.md'],**shown)
    edge('pure_dl_sw_factor_sam_v96_SW_likelihood_fold5','pure_dl_fullcv_v97','evolution','固定第五通过的完整配方，原其他折独立训练')
    source_complete=bool(snapshot['folds'][0].get('source_complete'))
    node('pure_dl_fullcv_v97/root','v97 第一折来源骨干','原local-context80结构从本折种子随机初始化；统计/Q/标签只用原64训练井，80/18。','本折初始化来源；最终18阶段和三通道单独记录。',stage='validation',status='recorded' if source_complete else 'running_snapshot')
    edge('pure_dl_paired_local_context_router80_sequence','pure_dl_fullcv_v97/root','evolution','同结构在原第一折重新训练，不加载第五权重')
