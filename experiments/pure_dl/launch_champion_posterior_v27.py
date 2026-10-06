"""Record this finite experiment and explicitly launch selected targets only."""
from pathlib import Path
from datetime import datetime
import argparse,ast,json,subprocess,sys
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_posterior_v27_20260929'
PREVIOUS=ROOT/'results/deep_learning/all_heads_v26_20260928'
TARGETS=('POR','PERM','SW')

METHOD='''# v27：80.300730612039上的后验条件修正

## 探索重点与预先固定对照
旧版有效部件：POR位置幅值修正、PERM门控、SW分量权重。只继续这些部件，在旧版完整输出后叠加零初始化的小修正，不重新扫描其他因子，不替换骨干。
每目标两项self/pair，共六项。每位专家头输入128隐藏+63幅值+4自身后验+4第二后验=199维，199→32→输出。self的第二描述重复自身，pair使用另一专家。两项完整参数初始化相同；输入通道数相同，self特征有冗余，因此有效信息维度不同正是被检验机制。
描述是默认概率、混合权重归一化熵、条件混合均值、log1p标准差。均值/标准差均在原目标标准化变换空间；均值截断[-8,8]，log1p标准差截断[0,8]。全部stop-gradient，只计算本目标，不使用其他目标标签或输出。

## 已实现与验证计划
POR、SW每候选13328参数，PERM12866参数。基座及先前有效修正全部冻结；只有新小头有梯度。门控/混合logit修正以0.25有界，位置修正以0.25当前固定分量标准差有界。第0轮必须精确复现v26对应列，否则停止。训练后恢复各候选自己的最佳，最终采用每目标至多一个完整方案并实际评分。正收益候选从磁盘重建复核。未做的验证不写成已完成。

## 训练与证据边界
原9曲线/206特征+原63幅值侧视图，原64训练井/16验证井，578891/144665有效点；全部预处理与Q50沿用原训练侧文件。相同随机窗口生成器seed20261010，head seed20261103；max80/patience18/min10、64步/轮、batch64。沿用v26 AdamW3e-4、cosine→1e-6、wd0.01、clip2。此eta_min为已完成v26匹配设置，非改动历史训练。
原单覆盖256格/800动作及默认动作不变，原own-channel似然/3不变；不同候选的训练参数、优化器及早停互不共享。同目标只复用固定骨干的计算；跨目标不共享参数。

## 收益
启动前尚未证实。初始分不算训练增益；报告排除epoch0的训练后最高分及最终保留分。pair-self只衡量本设置下另一个专家描述的增量，不把相对v26的全部变化归给后验信息（还增加了新的小头）。六项开发选择未校正，不视为独立泛化证据。

## 不再展开的方向
不重启冷启动替换、FFN扩容、无收益形态侧分支、负因子网格、旧ABD或其他折。本批以六项有限比较结束，失败或负结果完整记录；旧交付不变。

## 产物与运行
模型champion_posterior_v27.py，训练train_champion_posterior_v27.py。launch仅显式传入目标，其他目标不排队。最多2GPU。last_state原子保存参数/优化器/调度器/RNG/早停，只有显式--resume恢复。各目录维护README/PROJECTS，研究目录维护汇总与图谱。
'''


def append_once(path,marker,text):
    old=path.read_text(encoding='utf-8-sig') if path.exists() else ''
    if marker not in old:
        with path.open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')


def jobs():
    result=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=proc.info;a=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and '-c' not in a and any(
                'pure_dl/' in s.replace('\\','/') and Path(s.replace('\\','/')).name.startswith(('train_','benchmark_','assemble_','verify_')) for s in a):
                result.append(v)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--targets',nargs='*',choices=TARGETS,default=[]);args=ap.parse_args()
    if len(set(args.targets))!=len(args.targets):raise ValueError('Repeated target')
    for name in ['champion_posterior_v27.py','train_champion_posterior_v27.py']:
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8-sig'))
    STUDY.mkdir(parents=True,exist_ok=True)
    pred=PREVIOUS/'assembled_pred.npy'
    if not pred.exists():raise FileNotFoundError(pred)
    ref=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    (STUDY/'reference.json').write_text(json.dumps(dict(prediction=str(pred.relative_to(ROOT)),recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),scores=ref['scores']),ensure_ascii=False,indent=2),encoding='utf-8')
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    p=STUDY/'PROJECTS.md'
    if not p.exists():p.write_text('# v27进度\n\n目标：保留80.30强基座，检验跨专家后验条件是否改进三个有效输出因子。\n\n已完成：模型/训练入口、六项预定义对照和数据引用。\n\n计划：显式分批运行，恢复最佳，正收益候选磁盘重载、固定三模型合并评价。\n\n未完成：真实训练增益、最终选择、跨折证据。\n\n达成后：固定配方再讨论跨折确认；当前仍不补折、旧交付保留。\n',encoding='utf-8')
    marker='## v27基于80.30的后验条件增量'
    text=marker+'\n当前用户要求继续基于80.30优化。本批预定义三目标各self/pair六项，冻结v26全预测器，仅有效位置/门控/权重因子的小头更新；同参数同初始状态，只差第二份后验输入来源。原数据/Q50/分折/评分/80-18/max2保持，无冷启动/扩容/旧队列或其他折。维护champion_posterior_v27_20260929/PROJECTS.md。'
    for path in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md']:
        append_once(path,marker,text)
    for target in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_champion_posterior_v27_{target}_fold5'
        out.mkdir(exist_ok=True)
        for mode in ('self','pair'):
            arm=out/mode;arm.mkdir(exist_ok=True)
            if not (arm/'README.md').exists():
                (arm/'README.md').write_text(METHOD+f'\n本目录：{target}/{mode}，尚无训练成绩。\n',encoding='utf-8')
                (arm/'PROJECTS.md').write_text('# 当前进度\n\n目标：有限后验条件比较。已完成配置，训练与收益未证实。计划运行80/18并恢复最佳；后续有益候选联合评分。其他折/旧交付不变。\n',encoding='utf-8')
        if not (out/'PROJECTS.md').exists():(out/'PROJECTS.md').write_text(f'# {target} v27\n\n两项独立候选预定义；仅显式启动，真实状态见runtime_status.json。\n',encoding='utf-8')
        if not (out/'AGENTS.md').exists():(out/'AGENTS.md').write_text((STUDY/'AGENTS.md').read_text(encoding='utf-8'),encoding='utf-8')
        append_once(ROOT/'EXPLORATIONS.md',f'pure_dl_champion_posterior_v27_{target}_fold5',
            f'- [{target}有效头：自身/跨专家后验条件](outputs/pure_dl_champion_posterior_v27_{target}_fold5/self/README.md)：v26父模型冻结，六项有限对照之一。')
    active=jobs()
    if len(active)+len(args.targets)>2:raise RuntimeError(f'GPU concurrency would exceed2: {active}')
    for target in args.targets:
        out=ROOT/'outputs'/f'pure_dl_champion_posterior_v27_{target}_fold5'
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Existing launch not repeated: '+target)
    for target in args.targets:
        out=ROOT/'outputs'/f'pure_dl_champion_posterior_v27_{target}_fold5'
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_champion_posterior_v27.py')),'--target',target,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        rec=dict(target=target,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
        (out/'launch.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
        for path in [STUDY/'PROJECTS.md',out/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
            with path.open('a',encoding='utf-8') as f:f.write('\n\n## 显式启动\n'+json.dumps(rec,ensure_ascii=False)+'\n其余目标不自动排队。\n')
        print('LAUNCHED',json.dumps(rec,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
