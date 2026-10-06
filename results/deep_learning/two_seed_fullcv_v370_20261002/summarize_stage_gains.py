"""Compact actual adjacent-stage changes from immutable terminal evidence."""
import csv,json
from pathlib import Path

study=Path(__file__).resolve().parent
stages=['SOURCE','POR113','POR151','PERM154','SW104','POR341','PERM336','SW351']
records=[json.loads(p.read_text(encoding='utf-8')) for p in study.glob('fold_*/*_TERMINAL.json')]
records.sort(key=lambda r:(r['fold'],r['identity'],stages.index(r['stage'])))
rows=[]
for record in records:
    if record['stage']=='SOURCE':continue
    gain=record['stage_gains'][-1]
    assert gain['stage']==record['stage']
    rows.append(dict(identity=record['identity'],fold=record['fold'],stage=record['stage'],target=gain['target'],
        previous_score=gain['previous_target_score'],selected_score=gain['selected_target_score'],
        actual_target_gain=gain['actual_target_gain'],TOTAL_contribution=gain['actual_TOTAL_contribution'],
        DEFAULT_gain=gain['strata']['DEFAULT']['delta'],CONT_gain=gain['strata']['CONT']['delta'],
        optimizer_initial_gain=gain['optimizer_gain'],best_epoch=gain['best_epoch'],stopped_epoch=gain['stopped_epoch'],
        wall_seconds=gain['process_wall_seconds'],concurrency=gain['concurrency']))
assert rows
with (study/'STAGE_GAINS.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# 已完成阶段的实际增益','',
    '仅汇总已结束且最佳权重重载核验通过的阶段。按相邻阶段计算实际通道差值，TOTAL贡献按官方通道权重计算；训练初始值差另列于CSV，不能与实际前驱收益混用。',
    '这是增量轨迹记录，不能替代删掉某阶段并重训后续阶段的消融。单折不代表完整五折收益；本批原8阶段训练配方保持。',
    '单项墙钟含加载、预热、训练、重载和写盘；恢复项包含失败及恢复两次调用，排除等待授权时间。并发列保留，不把重叠运行单项耗时当作同等工作量总耗时。','',
    '| 模型 | 折 | 阶段 | 通道增益 | TOTAL贡献 | 默认值增益 | 连续值增益 | 最佳/停止轮 |',
    '|---|---:|---|---:|---:|---:|---:|---|']
for r in rows:
    lines.append(f"| {r['identity']} | {r['fold']} | {r['stage']} | {r['actual_target_gain']:+.9f} | {r['TOTAL_contribution']:+.9f} | {r['DEFAULT_gain']:+.9f} | {r['CONT_gain']:+.9f} | {r['best_epoch']}/{r['stopped_epoch']} |")
lines+=['','证据：各折的 `*_TERMINAL.json`；精确数值和单项耗时见 `STAGE_GAINS.csv`。',
    '未完成的阶段不填入表格，不将训练中分数或初始投影的微小差异写成训练收益。','']
(study/'STAGE_GAINS.md').write_text('\n'.join(lines),encoding='utf-8')
print(dict(terminal_stage_rows=len(rows),folds=sorted({r['fold'] for r in rows}),CPU_only=True,training_recipe_changed=False))
