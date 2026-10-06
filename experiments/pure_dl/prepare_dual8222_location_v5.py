"""Prepare three location-readout arms; launch only explicit named modes, no queue."""
from pathlib import Path
from datetime import datetime
import argparse, ast, hashlib, json, subprocess, sys
import psutil

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
STUDY=ROOT/'results/deep_learning/dual8222_location_v5_20260928'
MODES=['linear','point32','local32']
PREFIX='pure_dl_dual8222_v5_location_'
PLAN='''# 双专家位置读出：线性/逐点/局部匹配比较

## 探索重点
从已共同验证的80.268734508852配方出发，只优化POR分量位置。三个预定义对照：linear继续训练原线性位置读出；point32在同线性读出上加128→32→8非线性残差；local32在相同初始化和宽度上再加入5点逐通道卷积残差。可训练参数2064/10848/11232。三组均有同一个可训练线性位置部分。
新增非线性修正按每点、每专家、每分量的固定标准差限制：delta=(0.25*sigma)*tanh(r/(0.25*sigma))，在r=0的导数为1；0.25为预先固定值，未搜索。仅新增残差受此界约束，线性位置参数允许正常更新。末层零初始化，local内部卷积零初始化，初始函数及point/local基础随机参数保持一致。

## 已实现和验证计划
保留原两个骨干/内部路由/校准、所有混合权重与尺度、门控及PERM/SW。冻结行使用buffer，不受AdamW权重衰减影响。输入九曲线/206维、原第五折和训练人口不变；沿用单覆盖、256分布格、800候选动作解码。
从80.268735的selected_heads.pt加载三目标，重新初始化优化器与80轮余弦调度；该新增训练预算对三组相同，不冒称原轨迹续跑。固定Q50、seed20261010独立窗口RNG、batch64、64步/轮、lr3e-5至1e-5、wd0.01、clip2、patience18/min10，POR似然损失除3。
先并行linear与point32，释放任务位后明确启动local32，最多2GPU，没有自动队列。每轮保存完整可恢复状态；原脚本/已有模型/已交付包不修改，未自动补折。
初始全第五折逐元素复现，首批梯度范围检查，最终恢复与固定因子核对内置在正式训练中，不额外启动预检实验。新的非线性读出保存完整模块状态，不能折叠成旧线性矩阵。

## 收益状态
尚未训练。不能预先认定非线性/卷积更优；应先与同起点linear比较，再比较point32与local32。point/local相差384个参数来自局部卷积本身，不宣称等参数。第一项读文件shell引号失败（零修改、零训练）已记录，不影响模型。

## 不继续的方向和验收
本批仅三项，不扩张核长、宽度、学习率、界限系数搜索。若无超过同起点线性控制的收益，就记录该设置的限制，不将更多训练预算当结构贡献。验证井多次用于开发，任何最高分都不是独立泛化证明。第五折严格>80.5且有明确收益，完整OOF>82仍未达成；当前补折继续暂缓。必须维护PROJECTS.md记录已做/计划/未做/目标/完成后的工作。
'''


def active_trainers():
    result=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=proc.info; args=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(args) and any(Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in args):
                result.append(d)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--launch',nargs='*',choices=MODES,default=[]);args=ap.parse_args()
    if len(args.launch)!=len(set(args.launch)):raise ValueError('Duplicate launch mode')
    trainer=CODE/'train_dual8222_location_v5.py'
    if args.prepare:
        if STUDY.exists() or trainer.exists():raise RuntimeError('Refuse overwrite existing study')
        source=CODE/'train_dual8222_por_subspace80.py'
        text=source.read_text(encoding='utf-8-sig')
        edits=[
            ('from dual8222_por_subspace import Network,MODES','from dual8222_location_readout_v5 import Network,MODES',1),
            ("SELECTION=ROOT/'outputs/pure_dl_dual8222_selected_round2_fold5'","SELECTION=ROOT/'outputs/pure_dl_dual8222_v4_por_mean_only80_fold5'",1),
            ('POR subspace comparison, all arms warm-start the fixed 80.257638 recipe.','POR location-readout comparison from the verified 80.268735 recipe.',1),
            ('80.25763862460869','80.26873450885158',1),
            ("Path(__file__).with_name('dual8222_por_subspace.py'),","Path(__file__).with_name('dual8222_location_readout_v5.py'), Path(__file__).with_name('dual8222_por_subspace.py'),",1),
            ("'test_access':False,'remaining_folds_authorized':False,","'test_access':False,'remaining_folds_authorized':False,\n         'location_readout':'linear base trainable in all arms; optional bounded32wide residual',\n         'residual_bound_fraction':0.25 if args.mode!='linear' else None,\n         'local_kernel':5 if args.mode=='local32' else 1,\n         'comparison':'same initial80.268735; same linear parameters trainable; point32 versus local32 differs only added192per-expert context parameters',",1),
            ("              'head_state_dicts':model.export_legacy_heads(),'source_kinds':['target_group']*3,", "              'readout':model.export_readout(), 'initialization_manifest':selected,",1),
            ("'warm_start':'density_only@1'","'warm_start':'v4_mean_only@12'",1),
            ("atomic_torch(out/'selected_heads.pt',exported)","atomic_torch(out/'readout_checkpoint.pt',exported)",1),
        ]
        for old,new,count in edits:
            if text.count(old)!=count:raise RuntimeError('Unexpected source occurrence count: '+old)
            text=text.replace(old,new)
        ast.parse(text)
        ast.parse((CODE/'dual8222_location_readout_v5.py').read_text(encoding='utf-8'))
        trainer.write_text(text,encoding='utf-8')
        STUDY.mkdir()
        (STUDY/'trainer_derivation.json').write_text(json.dumps({'source':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'target':trainer.name,'edits':[{'old':a,'new':b,'count':n} for a,b,n in edits]},ensure_ascii=False,indent=2),encoding='utf-8')
        agents='# 目标与约束\n本批只比较同起点双专家POR位置读出结构。遵守原九曲线/分折/评分/80-18/最多2GPU，旧交付不可覆盖，无自动补折。维护PROJECTS.md：已做、计划、未做、目标及达成后事项，每批更新。\n'
        project='# 当前进度\n目标：验证位置读出结构增量，完整OOF>82未完成。\n已做：来源与三项对照已锁定，模型和训练入口实现。\n计划：三组正式训练、完整第五折最终恢复、逐井及误差人口分解。\n未做：真实训练收益、序列化重载验证；续跑等价性未实测。\n达成后：固定有效方案再做泛化确认，当前不自动补折。\n'
        for p in [STUDY]+[ROOT/'outputs'/(PREFIX+m+'80_fold5') for m in MODES]:
            if p!=STUDY:p.mkdir(exist_ok=False)
            for name,value in [('README.md',PLAN),('AGENTS.md',agents),('PROJECTS.md',project)]:
                (p/name).write_text(value,encoding='utf-8')
        for mode in MODES:
            p=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
            (p/'runtime_status.json').write_text(json.dumps({'status':'planned','mode':mode,'atlas_current_trial':False}),encoding='utf-8')
        rule='\n\n## 2026-09-28 当前继续：有界位置残差读出\n本批实现方案限linear/point32/local32三项，从80.268734508852同起点新优化，只POR位置更新；三组共享线性位置更新，新增非线性残差限制为0.25固定分量标准差。该界仅约束新增支路。原9曲线/折/评分/Q/80-18/2GPU保持；其余头和混合尺度冻结，无外部模型或旧ABD恢复、无补折。原子完整状态保存，逐批更新PROJECTS/README/图谱，保留实际失败记录。\n'
        for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md',CODE/'AGENTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(rule)
        with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
            for mode in MODES:f.write('\n- [POR位置读出v5 '+mode+'](outputs/'+PREFIX+mode+'80_fold5/README.md)：同80.268735起点；启动前无成绩。\n')
    if not STUDY.exists():raise RuntimeError('Prepare first')
    active=active_trainers()
    if len(active)+len(args.launch)>2:raise RuntimeError('GPU concurrency limit: '+str(active))
    for mode in args.launch:
        out=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        if (out/'launch.json').exists() or (out/'config.json').exists():raise RuntimeError('Duplicate launch '+mode)
    launches=[]
    for mode in args.launch:
        out=ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        command=[sys.executable,'-u',trainer.as_posix(),'--out',out.as_posix(),'--mode',mode]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':mode,'pid':proc.pid,'command':command,'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        launches.append(row);print('LAUNCHED',json.dumps(row,ensure_ascii=False),flush=True)
    if launches:
        old=json.loads((STUDY/'launches.json').read_text()) if (STUDY/'launches.json').exists() else []
        (STUDY/'launches.json').write_text(json.dumps(old+launches,ensure_ascii=False,indent=2),encoding='utf-8')
        note='\n\n## '+datetime.now().astimezone().isoformat()+' POR位置读出v5\n三项预定义匹配实验已实现；本次实际启动'+str([(r['mode'],r['pid']) for r in launches])+'。初始、梯度和收益以实际日志为准，未自动启动其余队列/旧路线/补折。完整OOF>82未完成，旧交付不变。详细记录results/deep_learning/dual8222_location_v5_20260928。\n'
        for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
            with p.open('a',encoding='utf-8') as f:f.write(note)

if __name__=='__main__':main()
