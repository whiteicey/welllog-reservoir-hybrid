"""Record actual negative results; remove only this finished study's duplicate weights."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[3]
study = Path(__file__).resolve().parent
out = root/'outputs/pure_dl_por_effective_sam_v98_POR_fold5'
report = json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
assert all(a['best_epoch'] == 0 and not a['positive_gain_and_gate'] for a in report['arms'])
def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
plan = json.loads((study/'PLAN.json').read_text(encoding='utf-8'))
for name, sha in plan['source_hashes'].items():
    assert hashlib.sha256((root/'experiments/pure_dl'/name).read_bytes()).hexdigest() == sha
plan.update(status='completed_no_gain', candidate_scores=report['arms'],
            completed_at=report['finished_at'], actual_seconds=report['seconds'],
            peak_allocated_MiB=report['peak_allocated_MiB'], fair_ablation_delta=0.0)
write(study/'PLAN.json', plan)
text = '''# v98 POR有效模块的优化器对照：最终记录

探索重点：从v96固定组合TOTAL80.504197424477的POR来源v31 anchored完整函数热启动，只联合更新现有路由与幅值位置修正。预定义AdamW与SAM(rho0.05)同容量43635参数、同初始权重、同训练窗口、学习率3e-5的对照。原双骨干、其他密度因子和锚点冻结，主损失为自身POR Student-t3混合NLL+BCE/Q50。

已探索与验证：两组实际训练至第18轮，patience18停止，均恢复第0轮原模型。完整144665点恢复一致，冻结状态逐张量一致，实际训练梯度覆盖路由/位置，参数实际更新。AdamW更新后最佳为第1轮TOTAL80.502927440098，SAM第1轮80.503227187203。不存在完整五折证据，也未启动本候选其他折。

取得收益：相对v96 TOTAL均为0；预定义选定SAM减AdamW消融差0。训练损失有所下降但官方验证成绩未提高。80.504197仍来自保留的第0轮父函数，不属于本次新收益，positive_gain_and_gate两项均false。耗时256.761939秒，峰值719.845703MiB；包括加载、预热、初始全点评估、训练、验证与保存。两组在同一GPU进程共享冻结特征、串行更新，和v97补折任务重叠；无同等工作量串行耗时对照，不声称加速。

不继续的点及证据：当前热启动/43635参数/学习率3e-5/SAM0.05两项均回退原函数，关闭此配置邻域的rho/LR扫描。结论仅限本固定对照，不泛化为SAM或POR优化整体无效。下一机制可测试评分目标对齐，须独立记录。

数据与训练边界：原9曲线、206维与既有幅值63侧特征、原第五折/seed20261010/Q50/原单覆盖解码；80/18/min10/batch64/64步、cosine80*64/eta_min1e-5。最大2GPU任务；原模型、交付与v97固定补折配方不变。

保留新源码、配置、日志、历史、梯度/初始/最终证明与预测。两组冗余第0轮权重、负结果更新权重和last-state按deletion_manifest.json记录删除；原v31/v96正来源不动。不能再对本负实验自动恢复。
'''
(study/'README.md').write_text(text, encoding='utf-8')
(study/'PROJECTS.md').write_text('''# v98 PROJECTS

目标：检验现有POR有效模块的同容量优化器收益；第五折严格>80.5且正收益，最终完整OOF>82。

已做：AdamW/SAM实际训练、各自最佳恢复和144665点验证；两组18轮停止/best0，绝对与消融收益均0；记录范围有限的负结果、来源哈希和清理清单，更新图谱。

计划：关闭此固定配置邻域；独立研究评分目标对齐，当前v97补折保持已授权固定配方。

未做：本候选其他折、完整OOF及任何参数扫描；本次不补折。

完成后可选：新的有界机制对照须单独记录，保留所有父模型与交付。
''', encoding='utf-8')

def active():
    for p in psutil.process_iter(['name','cmdline']):
        try:
            if (p.info['name'] or '').lower() == 'python.exe' and any(
                    Path(arg).name == 'train_por_effective_sam_v98.py' for arg in p.info['cmdline'] or []):
                raise RuntimeError('v98 training still live; cleanup forbidden')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

manifest = study/'deletion_manifest.json'
if not manifest.exists():
    active()
    targets = [out/'last_state.pt'] + [out/m/name for m in ('adamw','sam')
                for name in ('model_best.pt','model_best_updated.pt')]
    rows = []
    for p in targets:
        resolved = p.resolve(); assert resolved.is_relative_to(out.resolve())
        if p.exists():
            rows.append(dict(path=str(resolved), bytes=p.stat().st_size,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             reason='finished no-gain duplicate epoch0 or negative updated/recovery state'))
    doc = dict(planned_at=datetime.now().astimezone().isoformat(),files=rows,
               total_bytes=sum(r['bytes'] for r in rows), status='planned')
    write(manifest,doc)
    for row in rows:
        active();p=Path(row['path'])
        assert p.stat().st_size == row['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        p.unlink();row['deleted']=True
    doc.update(status='completed',completed_at=datetime.now().astimezone().isoformat())
    write(manifest,doc)
note='''
## v98最终：POR有效模块AdamW/SAM无新收益

两组同43635参数、同v31完整热启动，18轮停止并恢复epoch0，TOTAL均80.504197424477，相对v96及选定SAM减AdamW均0。更新后最佳均更低，不扫描此配置rho/LR邻域。实际256.762秒/峰值719.846MiB（含加载/预热/验证/保存，与v97重叠），无加速结论。负实验冗余权重按独立清单删除，原正来源保留。详见[最终记录](results/deep_learning/por_effective_sam_v98_20260930/README.md)。v97授权补折继续，完整OOF>82尚未验证。
'''
for p in (root/'PROJECTS.md',root.parent/'PROJECTS.md'):
    current=p.read_text(encoding='utf-8')
    if '## v98最终：' not in current:
        p.write_text(current+note.replace('(results/', '(welllog-reservoir-hybrid/results/') if p.parent == root.parent else current+note,
                     encoding='utf-8')
print(json.dumps(dict(status='recorded',deleted_bytes=json.loads(manifest.read_text(encoding='utf-8'))['total_bytes']),ensure_ascii=False))
