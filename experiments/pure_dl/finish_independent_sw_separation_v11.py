"""Finish the recorded bounded v11 study; no training or model mutation."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
from launch_independent_sw_separation_v11 import ROOT,WORK,STUDY,active


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data=json.loads((STUDY/'comparison.json').read_text(encoding='utf-8'))
    shared,split=data['candidates']
    assert all(c['reload_exact'] for c in (shared,split))
    matched=data['matched_split_minus_shared']
    sd=shared['populations'];sp=split['populations']
    nd=sp['true_nondefault']['new_SW']-sd['true_nondefault']['new_SW']
    dd=sp['true_default']['new_SW']-sd['true_default']['new_SW']
    conditional=(split['state_and_numeric_diagnostic']['conditional_decoder_SW_on_true_nondefault']-
                 shared['state_and_numeric_diagnostic']['conditional_decoder_SW_on_true_nondefault'])
    diag=shared['report']['gradient_diagnostic']
    note=f'''

## 本轮结论与下一步限定
两项训练及磁盘重载完成，无运行中的本批任务。sharedSW={shared['report']['target_score']:.12f}，splitSW={split['report']['target_score']:.12f}。同参数与同初始化状态已实核，split相对shared的等价TOTAL变化{matched['TOTAL_equivalent']:+.12f}，{matched['well_wins']}/16井提高、{matched['well_losses']}井下降。选择未校正的探索性井重采样区间为{matched['selection_unadjusted_exploratory95']}，不能据此宣布新井上的普遍结构排序。
拆分后真实非默认段通道分{nd:+.9f}，真实默认段{dd:+.9f}。非默认段的小幅改善不足以补偿默认段损失。门控0.5阈值召回提高，但门控与密度一起决定最终动作，分类准确率不等价于官方分数。
在真实非默认段上仅用连续分布解码（将门控质量设0的诊断），split较shared为{conditional:+.9f}。这是标签限定子集的离线诊断，没有用于选择逐点模型或推理；不能与旧模型未经同样处理的普通得分直接比较。
共享组128个固定训练窗口中，{diag['negative_cosine_batches']}/8批BCE/CE编码器梯度余弦为负，均值{diag['mean_cosine']:.6f}。split跨任务梯度严格为零，但验证未提高。这个小样本结果不支持将当前瓶颈直接归结为普遍梯度冲突；共享信息和每任务有效容量也随连接方式改变。
本次只关闭相同模型/预算附近的强制完全分离扫描，不推论所有独立编码器无效；POR/PERM/SW三个目标之间继续使用私有模型。
下一步尚未启动：保持一种固定SW表示，比较原BCE+条件CE与把默认动作竞争也纳入的联合效用监督。与v9仅对冻结门控下连续分布加效用项的试验区分；不调验证集阈值，不追加自动队列或补折。
当前继续保留已验证三私有模型TOTAL={data['kept_scores']['TOTAL']:.12f}。本批新模型未替换旧交付；第五折严格>80.5和完整OOF>82仍未完成。
'''
    marker='## 本轮结论与下一步限定'
    for path in [STUDY/'FINAL.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',WORK/'PROJECTS.md']:
        text=path.read_text(encoding='utf-8-sig')
        if marker not in text or path.parent in [ROOT,WORK]:
            with path.open('a',encoding='utf-8') as f:f.write(note)
    curpath=ROOT/'docs/deep_learning/atlas/curation.json'
    cur=json.loads(curpath.read_text(encoding='utf-8'))
    dead={'id':'dead:sw_v11_fully_separated_state_density','title':'SW内部完全分离：本次无增益','stage':'loss',
          'parents':['pure_dl_independent_sw_v11_shared80_fold5','pure_dl_independent_sw_v11_split80_fold5'],
          'summary':f"同92273参数和初始化的两TCN，共享SW{shared['report']['target_score']:.6f}，分离SW{split['report']['target_score']:.6f}。",
          'reason':'仅关闭本次模型/数据/预算附近的强制分离扩展。6/16井提高，跨任务梯度已隔离但验证无增益；不否定其他独立模型。'}
    if not any(n['id']==dead['id'] for n in cur['deadends']):cur['deadends'].append(dead)
    curpath.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'docs/deep_learning/atlas/build_atlas.py')],
                     capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=40)
    print('ATLAS_BUILD',p.returncode,p.stdout,p.stderr)
    if p.returncode:raise RuntimeError('Atlas build failed')
    atlas=json.loads((ROOT/'docs/deep_learning/atlas/atlas-data.json').read_text(encoding='utf-8'))
    for mode,c in zip(('shared','split'),data['candidates']):
        key=f'pure_dl_independent_sw_v11_{mode}80_fold5'
        node=next(n for n in atlas['nodes'] if n['id']==key)
        assert node['primary']['scope']=='第五折 · SW单目标'
        assert abs(node['primary']['value']-c['report']['target_score'])<1e-10
        print('ATLAS_VERIFIED',key,node['primary']['value'])
    print('REMAINING_TRAINERS',active())
    print('FINAL_REPORT',STUDY/'FINAL.md')
    print('TIME',datetime.now().astimezone().isoformat())


if __name__=='__main__':main()
