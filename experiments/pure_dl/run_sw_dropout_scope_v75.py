"""Explicitly launch matched additive/multiplicative full-network candidates."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_dropout_scope_v75 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='# v75 SW注意力随机失活的监督范围\n\n## 探索重点与父证据\nv74训练期dropkey使SW相对同起点control提高0.120507，低于当前冠军0.027986。默认子群+0.709668而数值子群-1.232741。本轮固定检验：保留门控的随机上下文，数值密度训练改用完整上下文，是否改善该权衡。\n\n## 预定义对照\n仅both/gate_only，两组同839481参数、同v56 additive第11轮完整骨干/FiLM和重置SW训练先验输出；不是加载v74最佳权重续训，不声称保留父73.713976函数。每批同一模型前向两次，第一次10%key丢弃，第二次完整上下文。两组门控BCE均来自第一次；both的密度NLL来自第一次，gate_only来自第二次。零系数项保留两图反向，双方同两次前向/反向遍历和抽样消耗，不改变训练窗口序列。共享全部参数，不添加独立模型或预测融合；门控与密度梯度仍可能在共享骨干相互影响。\n推理只做一次完整上下文，原8Gaussian分布及默认概率组合、256支持/800动作评分不变。两个训练前向均用SW自身标签；其他目标不更新。两次前向的RNG轨迹不同于v74单前向，因此重新跑匹配both控制，不把v74历史曲线当数值相同的控制。\n\n## 约束、已做与验证\n已实现，启动时无成绩。原9曲线206、64训练井/第五16井144665点/Q50、seed20261010、1024随机上下文中心128不变；训练侧统计/输入库，无新数据或外部权重。AdamW1e-4/wd.01/clip2、cos80*64至1e-5、max80/pat18/min10/b64/64steps。全部网络训练，原子完整状态/RNG，仅显式恢复；最多2GPU，先实际训练测资源再开第二项。若两图并存显存影响可记录实际串行，不为并发而降batch。\n直接训练保存实际损失与梯度、初始参数/预测哈希、最佳恢复与源哈希；结束真实三列评分，正候选磁盘原始输入重建。无冗余独立预检/审查，无其他折/旧队列/交付覆盖。\n\n## 收益、未做与不继续\n最终配对差和当前冠军绝对差分开，另列默认/数值子群，仅开发折诊断，未完成不得宣称收益。若gate_only不优于both，不扫本两图损失系数或该监督范围；v74正候选仍保留。结束删除负权重并保留源/配置/紧凑证据，更新PROJECTS/README/EXPLORATIONS/图谱。单项和总墙钟、并发、显存实记，包含加载/预热/训练/评估/启动间隔，代码与收尾另计。第五严格>80.5且明确收益才考虑其他折，完整原五折OOF>82未完成；目标后独立井确认和新目录交付。\n'
RULES='# v75规则\n仅both/gate_only两项；同v56骨干与重置先验，839481参数同两次前向，门控用dropkey，密度用dropkey或完整上下文。原数据/Q/评分/80-18/max2，不开其他折/旧队列或改交付。维护PROJECTS目标/已做/未做/后续。\n'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_dropout_scope_v75_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md',RULES),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_dropout_scope_v75.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v75 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_dropout_scope_v75_20260929','- [v75 SW失活监督范围](results/deep_learning/sw_dropout_scope_v75_20260929/README.md)：both/gate_only同两前向，密度分别用随机上下文或完整上下文；初始无成绩。')
    append(ROOT.parent/'AGENTS.md','## v75 SW失活监督范围','## v75 SW失活监督范围\n'+RULES)
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
