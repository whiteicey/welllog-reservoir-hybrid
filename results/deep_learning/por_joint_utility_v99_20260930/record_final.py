"""Finish the actual matched loss study and remove only its negative weights."""
import hashlib,json
from datetime import datetime
from pathlib import Path
import psutil
root=Path(__file__).resolve().parents[3];study=Path(__file__).resolve().parent
out=root/'outputs/pure_dl_por_joint_utility_v99_POR_fold5'
report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
assert all(a['best_epoch']==0 and not a['positive_gain_and_gate'] for a in report['arms'])
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=json.loads((study/'PLAN.json').read_text(encoding='utf-8'))
for name,sha in plan['source_hashes'].items():
    assert hashlib.sha256((root/'experiments/pure_dl'/name).read_bytes()).hexdigest()==sha
like,joint=(next(a for a in report['arms'] if a['mode']==m) for m in ('likelihood','joint'))
updated_delta=joint['best_after_update']['reference_replacement_total']-like['best_after_update']['reference_replacement_total']
plan.update(status='completed_no_gain',candidate_scores=report['arms'],
            completed_at=report['finished_at'],actual_seconds=report['seconds'],
            peak_allocated_MiB=report['peak_allocated_MiB'],selected_ablation_delta=0.,
            post_update_diagnostic_delta=updated_delta)
write(study/'PLAN.json',plan)
(study/'README.md').write_text('''# v99 POR联合动作收益代理：最终无收益

探索重点：原v31保留完整函数、43635现有路由/幅值位置参数、同初始化/抽样/AdamW3e-5，仅差似然 versus 似然+0.5联合动作收益代理。原Student-t3密度、256格/800动作与默认动作、推理解码保持，softaction温度0.05；假设评分目标对齐能改善官方泛化。

已经探索与验证：两组实际训练18轮后早停，均恢复第0轮；全部144665预测恢复一致、冻结状态逐张量一致，训练梯度实际到达路由/位置。likelihood更新后最佳第1轮TOTAL80.502927440098；joint第1轮80.502757307294。训练物理POR查表只含原578891训练点，复制验证标签0，其他点为NaN；其他目标预测不进损失。可微CDF只用有限内边界避免无穷端点梯度，1024点激活重算控制显存。

取得收益：选定两组TOTAL均80.504197424477，相对v96与预定义joint减likelihood均0，positive_gain_and_gate均false；该分数来自原第0轮而非新训练收益。更新后诊断joint减like=-0.000170132804，也没有优势。真实总墙钟355.399583秒、峰值754.907227MiB，包含加载/预热/epoch0/训练/验证/保存；单GPU进程共享冻结特征、两头分别更新，与v97第4折重叠，没有同工作量串行对照或加速结论。仅原第五折，不是完整OOF。

不值得继续的点与证据：关闭当前原函数/43635参数/AdamW3e-5/联合动作系数0.5/温度0.05组合的邻域扫描。先前v98优化器与本项损失对照都没有选定收益；该更新范围不继续连续扫参。不能据此泛化为所有效用损失或POR优化无效。

边界：原9曲线/206+已采用幅值63、原分折/seed20261010/Q50；80/18/min10/b64/64步、cosine80*64/eta_min1e-5；旧双骨干/尺度/分量权重/锚点冻结。无其他折、旧任务恢复或交付覆盖。v97补折固定配方不改，完整原OOF>82仍待最终验证。

新训练源码/配置/日志/历史/初始与梯度证明/最终恢复报告/预测保留；冗余best0、负结果更新权重和last-state按deletion_manifest.json删除，原v31/v96正来源完整保留。该负实验不自动恢复。
''',encoding='utf-8')
(study/'PROJECTS.md').write_text('''# v99 PROJECTS

目标：公平比较POR联合动作效用；第五折严格>80.5且真实收益，最终完整OOF>82。

已做：两组43635参数实际18轮训练/各自最佳恢复、144665点验证、来源哈希/训练标签隔离/实际梯度/冻结状态记录。两组best0，选定收益0，更新后joint-like为负；负权重按清单删除、图谱及总项目记录更新。

计划：关闭此固定更新范围的系数/温度/优化器邻域；继续盯v97最后一折。

未做：本候选其他折、完整OOF及任何系数扫描；不补折。

达成后可选：独立记录的其他目标或结构机制，须真实最终收益证明；不修改当前OOF配方和交付。
''',encoding='utf-8')
def active():
    for p in psutil.process_iter(['name','cmdline']):
        try:
            if (p.info['name'] or '').lower()=='python.exe' and any(Path(a).name=='train_por_joint_utility_v99.py' for a in p.info['cmdline'] or []):
                raise RuntimeError('v99 still live; cleanup forbidden')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
manifest=study/'deletion_manifest.json'
if not manifest.exists():
    active();rows=[]
    for p in [out/'last_state.pt']+[out/m/n for m in ('likelihood','joint') for n in ('model_best.pt','model_best_updated.pt')]:
        p=p.resolve();assert p.is_relative_to(out.resolve())
        if p.exists():rows.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason='finished no-gain duplicate or negative recovery weights'))
    doc=dict(status='planned',planned_at=datetime.now().astimezone().isoformat(),files=rows,total_bytes=sum(r['bytes'] for r in rows));write(manifest,doc)
    for row in rows:
        active();p=Path(row['path']);assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        p.unlink();row['deleted']=True
    doc.update(status='completed',completed_at=datetime.now().astimezone().isoformat());write(manifest,doc)
for p,link in ((root/'PROJECTS.md','results/'),(root.parent/'PROJECTS.md','welllog-reservoir-hybrid/results/')):
    text=p.read_text(encoding='utf-8')
    if '## v99最终：无收益' not in text:
        text+='\n## v99最终：无收益\n\nPOR likelihood/joint各18轮早停并恢复epoch0，两组TOTAL80.504197424477，相对父函数/选定消融均0；更新后joint-like=-0.000170132804。实际355.400秒/峰值754.907MiB（含加载/预热/验证/保存、与第4折重叠），无加速结论。关闭本固定配置邻域，按清单删除负权重，保留来源与紧凑证据。见[独立最终记录]('+link+'deep_learning/por_joint_utility_v99_20260930/README.md)。v97授权补折继续，当前只剩第4折，完整OOF>82尚未验证。\n'
        p.write_text(text,encoding='utf-8')
print(json.dumps(dict(status='final_recorded',deleted_bytes=json.loads(manifest.read_text(encoding='utf-8'))['total_bytes']),ensure_ascii=False))
