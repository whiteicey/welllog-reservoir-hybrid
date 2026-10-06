"""Explicit one-shot launch of two predeclared SW-from-scratch models; no queue."""
from __future__ import annotations
import ast, json, subprocess, sys
from pathlib import Path
from datetime import datetime
import psutil

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
STUDY = ROOT/'results/deep_learning/independent_sw_scratch_v10_20260928'
MODES = ('tcn', 'bigru')
METHOD = '''# 独立SW从头训练：TCN与双向GRU

## 探索重点
v8六项和v9两项均已完成，三独立模型实际组合80.274135337791。此前全部从成熟多任务权重出发或修正固定先验。本轮将SW建模改为从头训练，比较两种不同私有时序编码器；POR/PERM继续使用已验证的独立模型。

## 预定义对照与已实现内容
仅tcn/bigru两项。九曲线生成的原206特征加窗口相对深度，输入207维。两者stem207->64、读出64->257相同；257输出为1默认状态logit和256连续概率格logits。
TCN四个带门控残差块，深度卷积核5、扩张率1/2/4/8，直接感受野61点。BiGRU每方向64维，输出128->64残差；有效段移到左侧后pack，完成后恢复原位置，窗口上下文双向且不跨井。
全部网络参数从头学习，不加载旧骨干、旧预测、教师或概率先验。只复用原训练折预处理、目标坐标和解码网格。输出零权重，bias由64训练井的默认率和非默认软直方图初始化，两方案初始预测相同。保留原训练井Q50作为固定监督权重，不重新学习Q。
损失相同：SW门控BCE+真实非默认SW的两格线性插值CE，按固定Q加权后除3；无先验KL、无残差幅度界。原第五折/seed20261010/80epochs/pat18/min10/64steps/batch64；AdamW6e-4、cosine->1e-5、weight_decay.01、clip2。
两方案规模相近但非完全同参，且TCN用AMP、GRU循环块用float32，属于同数据/预算的方案对比，不宣称严格的单因素因果消融。旧v8/v9与新模型还存在初始化、头/训练设置差异，不将跨代差值全部归因于GRU。
验证沿用144665原点、原行序、单覆盖256格/800动作的官方效用解码；未知测试标签不参与。最多2GPU任务，显式启动，无自动续跑或补折。

## 收益与证据
启动时尚未证实。训练前新模型不等于旧73.622384767129的SW分；每模型按自己的真实最佳验证分保存。最终记录相对自己初始及相对旧SW两种差值；不把从随机模型提高等同于超过旧模型。
每项保留独立完整模型、优化器、调度器、RNG与早停状态。真实失败落failure.json，失败不自动重试。最后按目标固定选择并进行实际模型重载和预测核对。

## 不值得继续的点及边界
不重复v9固定先验的小幅概率修正；不追加无界宽度、邻域或学习率搜索。两组即使接近也不能据单折宣布模型家族普遍优劣。只有超过旧SW才考虑替换；严格第五折>80.5与完整OOF>82仍未达成，旧交付和暂停ABD不变。

## 方法说明
h_i为64维隐藏特征；g_i=sigmoid(w_g h_i+b_g)为默认概率；p_i=softmax(W_p h_i+b_p)为256格连续分布。候选动作a的效用为g_i K(a,99.9)+(1-g_i) sum_r p_ir K(a,y_r)，K使用原SW相对容差评分。对非默认真值z，用相邻格的插值权重监督p。网络完整地学习状态与数值分布，不需要其他通道的隐藏状态。
GRU支持packed variable-length sequences的接口依据：PyTorch官方GRU与pack_padded_sequence文档。此处仅为实现依据，不保证本数据上的收益。
'''
AGENTS = '''# 本实验目标与约束
目标：在不共享通道骨干的框架内，比较SW从头训练的TCN与BiGRU。
遵循工作区原九曲线/原分折/评分/80-18/最多2GPU及不自动补折规则；旧模型和交付不覆盖。
维护PROJECTS.md，记录已做、计划、未做、目标与达成后的事项，每批结束更新。实验分数必须区分初始、训练中、恢复完成和完整OOF。保留真实失败，不伪称连接失败后任务已执行。
'''
PROGRESS = '''# 项目进度

## 目标
检验从头学习的SW专用时序模型能否超过旧SW73.622384767129，同时保持三个目标独立。

## 已完成
v8全部六项及三模型恢复已经核实，v9两项结束但未增益。本批两种结构和统一训练入口已实现。

## 计划
显式运行两个候选、完成独立最佳模型恢复、分井/默认状态误差比较；仅有实际改善才固定新SW并合并验证。

## 未完成
本批训练成绩、独立磁盘重载、异构组合收益、完整OOF>82均尚未完成。

## 达成后
固定胜出配方评估泛化与真实推理成本；补折仍需明确授权，旧交付保持原样。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    active = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            d = p.info; argv = d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'base64' not in ' '.join(argv):
                if any('pure_dl/' in a.replace('\\', '/') and
                       Path(a.replace('\\', '/')).name.startswith(('train_', 'benchmark_')) for a in argv):
                    active.append(d)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if active:
        raise RuntimeError(f'Refuse pair launch with active training processes: {active}')
    for name in ('independent_sw_scratch_v10.py', 'train_independent_sw_scratch_v10.py'):
        ast.parse((ROOT/'experiments/pure_dl'/name).read_text(encoding='utf-8-sig'))
    dirs = [ROOT/'outputs'/f'pure_dl_independent_sw_v10_{m}80_fold5' for m in MODES]
    if any(p.exists() for p in dirs):
        raise RuntimeError('Run directories already exist; inspect status instead of duplicate launch')
    STUDY.mkdir(parents=True, exist_ok=False)
    for name, text in [('METHOD.md', METHOD), ('AGENTS.md', AGENTS), ('PROJECTS.md', PROGRESS)]:
        (STUDY/name).write_text(text, encoding='utf-8')
    for p in dirs:
        p.mkdir()
        (p/'README.md').write_text(METHOD+'\n本目录架构：'+p.name+'\n', encoding='utf-8')
        (p/'AGENTS.md').write_text(AGENTS, encoding='utf-8')
        (p/'PROJECTS.md').write_text(PROGRESS, encoding='utf-8')
    rule = '\n\n## 2026-09-28 本轮继续优化：独立SW从头训练v10\n本轮预定义tcn/bigru两项，均从原207维输入直接学习SW门控与256格连续分布；不加载旧网络或预测，复用原训练侧预处理/网格/Q以保持评分边界。POR/PERM保留独立模型。原第五折、seed20261010、80/18/min10、batch64/64步、最多2GPU；无旧任务恢复、无自动搜索/补折。每批维护PROJECTS/README/EXPLORATIONS/图谱，严格区分冷启动改善与超越旧SW。\n'
    for p in [WORK/'AGENTS.md', ROOT/'AGENTS.md', ROOT/'experiments/AGENTS.md', ROOT/'experiments/pure_dl/AGENTS.md']:
        with p.open('a', encoding='utf-8') as f:
            f.write(rule)
    launches = []
    for mode, out in zip(MODES, dirs):
        args = [sys.executable, '-X', 'utf8', '-u',
                (ROOT/'experiments/pure_dl/train_independent_sw_scratch_v10.py').as_posix(),
                '--architecture', mode, '--out', out.as_posix()]
        with (out/'train.log').open('wb') as log, (out/'train.err').open('wb') as err:
            p = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=err, stdin=subprocess.DEVNULL,
                                 close_fds=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        record = {'mode': mode, 'pid': p.pid, 'command': args,
                  'started_at': datetime.now().astimezone().isoformat(), 'id': out.name}
        (out/'launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        launches.append(record)
    (STUDY/'launches.json').write_text(json.dumps(launches, ensure_ascii=False, indent=2), encoding='utf-8')
    note = '\n\n## '+datetime.now().astimezone().isoformat()+' 独立SW冷启动v10\n已完成模型/训练实现并显式启动'+str([(x['mode'], x['pid']) for x in launches])+'。此时尚无新分数。计划完成两模型恢复和同评分比较；不自动补折或改旧交付，完整OOF>82未完成。记录：results/deep_learning/independent_sw_scratch_v10_20260928。\n'
    for p in [WORK/'PROJECTS.md', ROOT/'PROJECTS.md']:
        with p.open('a', encoding='utf-8') as f:
            f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a', encoding='utf-8') as f:
        for x in launches:
            f.write('\n- [SW从头训练 '+x['mode']+'](outputs/'+x['id']+'/README.md)：独立新骨干与门控/概率格，同评分，启动时收益未证实。\n')
    print(json.dumps({'launched': launches, 'study': str(STUDY)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
