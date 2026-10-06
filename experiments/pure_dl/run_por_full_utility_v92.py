"""Launch only one explicitly selected arm; no queue and no other folds."""
from datetime import datetime
import argparse,json,subprocess,sys
from por_full_utility_v92 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v92 POR完整私有模型与动作效用\n\n探索重点：v86只更新30819路由、专家固定，两项无新增收益，不足以否定完整POR效用训练。SW效用已有完整同容量收益，但跨目标不先验成立。本批仅likelihood/joint两个POR完整热启动，从当前v31 anchored第7轮/v30 bundle/v25幅值修正/v8私有双专家全部函数开始，原参数不新增、原输出先验不重置。解除真实隐藏/输入路由梯度detach/no_grad，实际bundle读出和已有效位置修正可训练；未使用source读出副本、路由anchor参考副本、其他目标packed行冻结。明确绕过旧v25 frozen-body train override，训练时骨干原dropout恢复、eval关闭。两组同容量、同初始函数/参数/seed/预算，仅效用系数0/1不同，同损失计算图。\n\n对照：当前POR77.551936801885，固定PERM89.396724049366/SW73.793310952488组合TOTAL80.382093291214。原9曲线206/已有训练侧幅值63、原64/16训练578891/验证144665、seed20261010/Q50/官方评分/t3分布/log1p-expm1/256支持800均匀动作+default0.1保持。POR官方相对容差.08，主Q50 BCE+t3NLL/3与效用1/3，softaction温度.05、512点梯度checkpoint，代理损失不等于精确官方argmax。新AdamWbody1e-5/head3e-5/wd.01/分组clip2/cos80*64到1e-5/80上限pat18min10batch64steps64，body等于eta_min固定。不是旧轨迹续跑，不宣称CUDA逐位复现。\n\n已探索与验证：建立时尚未训练，收益未证实。直接训练，初始144665完整当前函数一致，实际第二batch的两骨干/输入路由/实际输出读出/v28/v29与效用gate/density梯度，冻结无梯度/终态权重不变，source/预处理/评分hash、最佳恢复及原三列实际评价。没有独立预检/审查或重复Q检查。保存排除epoch0的训练后最佳完整state/pred；正绝对或完整匹配增益两侧磁盘重载，不能选定轮数重训补证据。\n\n收益与不继续：分开报告绝对对当前、含起点joint-likelihood、完整训练后最佳同容量差、同轮轨迹、默认/数值/各井。epoch0恢复不算收益，开发折非完整OOF/独立泛化。没有绝对或完整公平局部增益则关闭本全网络/来源/低LR/系数1/温度.05预算配置，不临时扫参数或扩容；不泛化所有POR效用学习。\n\n已做：两有限入口。计划：显式分批训练、实际组合/必要重载/清理/图谱。未做：真实收益、其他折、独立泛化。normal最多2GPU，以实测显存启动第二项，无自动队列。记录单项加载预热/训练验证保存、同工作量两项总墙钟/重叠、allocated/reserved，准备收尾单列。原子完整last含current/best/更新后最佳/优化器/scheduler/RNG/停止，恢复显式。负项删除计划含hash/bytes先落盘，正源/当前/对照依赖/旧交付保留。目标后：原第五严格>80.5且明确收益才考虑其余四折，完整原OOF>82后新目录交付与独立井确认，无旧队列/其他折/交付覆盖。\n'
RULES='# v92目标与约束\n仅当前v31 anchored完整POR likelihood/joint两项；两私有骨干/输入路由/实际读出和有效修正更新，未用读出副本/route anchor/其他输出行固定，无新增容量或先验重置。原9曲线206+幅值63/训练侧统计/原分折/Q50/评分/t3/log1p/官方.08/default.1保持。body1e-5/head3e-5/cos到1e-5/80-18/min10/b64/64steps，normal/max2，以实测显存显式分批。原子完整last、训练后最佳/显式恢复；维护PROJECTS/独立README/EXPLORATIONS/图谱；清理计划先落盘，正源和交付保留。第五严格>80.5且明确收益后才考虑其他折，原完整OOF>82未完成，无旧队列/交付覆盖。\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_por_full_utility_v92_POR_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT),('PLAN.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_por_full_utility_v92.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v92 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','por_full_utility_v92_20260930','- [v92 POR完整模型联合效用](results/deep_learning/por_full_utility_v92_20260930/README.md)：当前v31完整私有POR，似然/效用0/1同初始函数；冻结旧未用头/anchor，原数据/Q50/评分不改，启动尚无收益。')
    
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(path,'## v92 POR完整模型效用范围','## v92 POR完整模型效用范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
