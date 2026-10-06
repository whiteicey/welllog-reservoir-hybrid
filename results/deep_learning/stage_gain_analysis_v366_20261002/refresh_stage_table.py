"""Render already-verified stage JSONs; no label scoring/model import/training."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    final = read(HERE / 'FINAL.json')
    original = final['models']['original']
    first = next(v for k, v in final['models'].items() if k != 'original')
    planned = read(ROOT / 'results/deep_learning/complete_seed_compare_v362_20261002/PLAN.json')['seeds']
    records = {}
    for seed in planned:
        own = ROOT / f'outputs/pure_dl_complete_seeds_v362_fold5/seed_{seed}'
        source_path = own / 'SOURCE/fold_5_report.json'
        if not source_path.exists():
            continue
        snapshot_path = HERE / f'SEED{seed}_STAGE_SNAPSHOT.json'
        snapshot = read(snapshot_path) if snapshot_path.exists() else None
        stages = snapshot['stages'] if snapshot else first['stages'] if seed == 20261010 else []
        complete_path = own / 'COMPLETE_MODEL.json'
        records[seed] = dict(source=read(source_path)['scores']['TOTAL'],
                             stages={row['stage']: row for row in stages},
                             snapshot=snapshot,
                             complete=read(complete_path) if complete_path.exists() else None)
    header = ['阶段', '原模型'] + [f'新{seed}' for seed in records]
    lines = [
        '# 各阶段真实前驱增益比较', '',
        '固定原第五折144665点。阶段行为对应通道相对实际前驱的增益；SOURCE行为起点总分。总分贡献需乘POR 0.30、PERM/SW 0.35。POR113可能含初始化分布家族变化，不全部算成优化增益。', '',
        '历史耐心18/14，新训练10→14，各自SOURCE/Q不同；第一个新SOURCE与历史使用相同实际种子。因此不是种子单因素因果对照。未完成项不得记为零收益，前缀分数不参与完整模型排名。', '',
        '| ' + ' | '.join(header) + ' |',
        '|---|' + '---:|' * (len(header) - 1),
        '| SOURCE总分 | ' + f'{original["source_scores"]["TOTAL"]:.6f} | ' +
        ' | '.join(f'{row["source"]:.6f}' for row in records.values()) + ' |',
    ]
    for stage in original['stages']:
        name = stage['stage']
        gains = [f'{stage["actual_target_gain"]:+.6f}']
        gains += [f'{row["stages"][name]["actual_target_gain"]:+.6f}' if name in row['stages'] else '未完成' for row in records.values()]
        lines.append('| ' + name + ' | ' + ' | '.join(gains) + ' |')
    totals = [f'{original["full_scores"]["TOTAL"]:.6f}']
    totals += [f'{row["complete"]["scores"]["TOTAL"]:.6f}' if row['complete'] else '未完成' for row in records.values()]
    lines += ['| 八阶段总分 | ' + ' | '.join(totals) + ' |', '',
              '前三个完整新模型SW351均无有效增益，第二/第三与SW104预测完全相同；原模型去351仍损失总分0.018151、SW连续值分0.347510。尚无通用无损删法已验证。', '',
              '具体best0初始化等价证明仅适用于对应运行，不适用于其他种子。151、154在其他运行有实际收益，不能统一删除。']
    for seed, record in records.items():
        snapshot = record['snapshot']
        if not snapshot or snapshot['complete_model']:
            continue
        lines += ['', f'新{seed}已核验{snapshot["completed_stages"]}/8阶段；前缀TOTAL{snapshot["prefix_scores"]["TOTAL"]:.12f}，不是完整八阶段或五折收益。']
        for stage in snapshot['stages']:
            lines.append(f'- {stage["stage"]}：通道增益{stage["actual_target_gain"]:+.12f}、总分贡献{stage["actual_TOTAL_contribution"]:+.12f}；连续值{stage["strata"]["CONT"]["delta"]:+.12f}、默认值{stage["strata"]["DEFAULT"]["delta"]:+.12f}；best{stage["best_epoch"]}/stop{stage["stop_epoch"]}。')
    lines += ['', '[删除次序与合并训练建议](STAGE_REMOVAL_DECISION.md)。[各自Q来源与比较限制](QUALITY_PROVENANCE.md)。当前五种子八阶段批次不变；合并方案未训练，历史训练成本不改写。']
    (HERE / 'STAGE_GAIN_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(rendered_seeds=list(records), CPU_cached_JSON_only=True, training_runs_added=0)))


if __name__ == '__main__':
    main()
