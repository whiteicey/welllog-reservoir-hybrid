"""Explicit one-shot launch of the user-authorized two-head-architecture study.
No polling, queue, retry, promotion, or restart of older experiments.
"""
from pathlib import Path
import ast
import datetime
import json
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
PYTHON = Path(r'C:\Users\z5364\.conda\envs\normal\python.exe')

README = '''# 双专家80.221861：独立输出头结构对照

## 探索重点
用户明确要求回到约80.22双专家后优化独立POR/PERM/SW输出头。实际来源为保留权重的local-context router，第52轮、第五折80.22186108303967；另外compact posterior80.227015只保留结果，没有checkpoint，本批没有冒充恢复它。
保持原两个128维骨干、输入校准、输入条件局部路由；不加入A6/B43/D52外部模型。每目标两个专家读出形成一个独立参数组，只对自身双专家预测的BCE+非默认NLL训练。

## 已经探索的内容与验证
代码已实现，启动与实时结果见runtime_status.json/train.log；真实初始预测、反向与最终成绩须以运行结果为准。
原线性输出行复制且可训练，增加输出层零初始化的非线性残差。两种候选都从原函数起步，均93420可训练参数、每目标31140。
pointwise残差128->80->25；local5残差128->77->(identity+depthwise5)->25。参数量一致，但宽度与局部结构同时不同，不能宣称是单独固定宽度的卷积因果消融。
保留原始128维特征幅值，不新增LayerNorm。原9曲线/206特征/原第五折与seed20261010不变。独立CUDA窗口生成器确保两结构抽样不被头初始化顺序改变。
固定源epoch50生效的训练点Q（1755点低于1，最小0.1），不启新动态Q或辅助头监督。三个AdamW，lr3e-5 cosine->1e-5，wd0.01，clip2，max80/pat18/min10，bs64、64步/轮；每头独立选模。
使用源模型单覆盖256分布格/800候选动作解码，不混入ABD四重窗口/精细解码收益。训练损失不使用验证标签；全部监督仅64训练井。
每轮保存heads_best.pt以及原子更新last_state.pt；后者含当轮参数、优化器、调度器、各类随机状态、最佳头/预测/历史和早停状态。只有显式--resume才恢复，不自动重启。保存与加载路径已实现；尚无中断后续跑等价性实测，不能宣称已验证精确续跑。

## 取得的收益
尚未证实。初始分数不算训练增益，历史最高组合不算最终共同恢复。训练完成后固定三个最佳头重推理并与所选预测比较，同时验证原骨干不变。
第五折严格>80.5且明确增益仍是后续门槛，完整OOF>82未完成；本批不补折、不修改已交付包。

## 不值得继续探索的点及其证据
不重启已暂停ABD/统计头或投票搜索，不把五分支方案当这次的骨干。源80.227015缺权重，不用其他模型名义替代。首轮只检验等参数预算的两种头结构，未有实际证据前不扫大量宽度、核长、优化器或损失族。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for name in ['dual8222_independent_heads.py', 'train_dual8222_heads80.py']:
        ast.parse((CODE/name).read_text(encoding='utf-8-sig'))
    active = []
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            value = process.info
            args = value.get('cmdline') or []
            command = ' '.join(args)
            if ('python' in (value.get('name') or '').lower() and 'pure_dl' in command
                and any(Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in args)):
                active.append(value)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    if active:
        raise RuntimeError('Do not start this pair alongside existing trainers: '+str(active))
    specs = [(mode, ROOT/'outputs'/('pure_dl_dual8222_heads_'+mode+'80_fold5'))
             for mode in ['pointwise','local5']]
    for mode, out in specs:
        if out.exists():
            raise RuntimeError('Refuse duplicate launch directory: '+str(out))
    launches = []
    for mode, out in specs:
        out.mkdir()
        (out/'README.md').write_text(README+'\n本目录结构：'+mode+'\n', encoding='utf-8')
        (out/'AGENTS.md').write_text(
            '# 双专家独立头实验约束\n\n目标：在保留双专家80.221861基础上改进独立输出头。遵循工作区最新数据/分折/预算/并发限制。维护PROJECTS.md，记录已做、计划、未做、目标和完成后事项；每批更新。保留运行中/最佳/可恢复权重，不覆盖原模型和交付。禁止自动补折或启动其他训练。\n', encoding='utf-8')
        (out/'PROJECTS.md').write_text(
            '# 实验进度\n\n目标：双专家独立头结构改善；完整OOF>82未完成。\n\n已完成：结构与原始评估口径实现，来源确认80.221861。当前准备启动。\n\n计划：最多80轮，各通道独立选最佳；最终共同恢复并与等参数对照比较。\n\n未完成：真实初始复现、训练收益、最后恢复及跨折验证。\n\n达成后：固定有效通道结构，再评估泛化；补折仍需明确恢复授权，原交付保持不变。\n', encoding='utf-8')
        args = [str(PYTHON), '-u', (CODE/'train_dual8222_heads80.py').as_posix(),
                '--out', out.as_posix(), '--architecture', mode]
        with (out/'train.log').open('wb') as log, (out/'train.err').open('wb') as err:
            process = subprocess.Popen(args, cwd=str(ROOT), stdout=log, stderr=err,
                stdin=subprocess.DEVNULL, close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        info = {'architecture':mode,'directory':out.name,'pid':process.pid,
                'started_at':datetime.datetime.now().astimezone().isoformat(),'command':args}
        (out/'launch.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(info)
        print('LAUNCHED',json.dumps(info,ensure_ascii=False),flush=True)
    entry = ('\n\n## 2026-09-28 双专家80.22独立头结构优化启动\n'
        '用户明确恢复新方向，旧ABD暂停不动。采用保留的local-router80.221861@52；compact80.227015未保留权重。'
        '新增同参数pointwise80与local5-width77两种残差头，每组93420参数，复制原输出起点、固定2骨干/路由/原256格800动作单覆盖；'
        'own-channel损失、固定源epoch50Q、lr3e-5、80/18/min10，各头独立选模。新训练器保存完整epoch-boundary状态。'
        '已启动PID '+','.join(str(x['pid']) for x in launches)+'；实际梯度/初始回放/训练分数待读取，不声称收益。'
        '没有补折或改交付；后续读取共同恢复与逐通道对照，完整OOF>82仍未完成。\n')
    for path in [ROOT.parent/'PROJECTS.md', ROOT/'PROJECTS.md']:
        with path.open('a',encoding='utf-8') as stream: stream.write(entry)
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as stream:
        for row in launches:
            stream.write('\n- [双专家80.221861独立头：'+row['architecture']+'](outputs/'+row['directory']+'/README.md)：原函数初始化、等参数结构比较；本批新启动，收益未证实。\n')
    print('PAIR_STARTED_NO_OLD_RESUME',len(launches),flush=True)


if __name__ == '__main__':
    main()
