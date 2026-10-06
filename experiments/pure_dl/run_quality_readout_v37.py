"""Explicit two-target launch; never queues or resumes jobs."""
from pathlib import Path
from datetime import datetime
import argparse,json,subprocess,sys
from quality_readout_v37 import ROOT,STUDY
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v37 Q50 与等权训练对照

目标：以第五折80.336513436816为固定基线，检验旧Q50降低权重的样本是否仍含可学习信号。
本批仅POR/SW各q50、uniform两个候选。PERM固定；每个目标两头同11857参数、同初始化、同训练窗口与新优化器，仅质量权重不同。新319→32→49逐点头同时修正门控、混合权重、位置和尺度；旧模型全冻结。门控/权重/rawscale增量界.25，位置界为父标准差的.25。零输出初始化保留原函数。
原9曲线/206特征、已有63幅值侧特征、原64训练井/16验证井与144665点评分、单覆盖256格800动作解码不变。原Q50是训练侧历史质量权重，不使用验证/测试标签构造权重。uniform不把困难点预先认定为噪声。
max80/patience18/min10/batch64/64步，AdamW lr3e-4/weight_decay.01，cosine80*64至1e-5，clip2。两目标独立进程、至多2GPU任务；同目标两候选共享冻结前向以节省时间，参数/优化器/早停独立。新头NLL/3，不增加辅助标签或外部模型。
已完成：实现模块、复用原训练侧数据和评分；启动后成绩见输出状态。
待完成：真实训练、最佳恢复、固定三列组装验证、记录消融差与基线差；当前尚未证实收益。
不继续的点：v32至v36输入侧统计/微型协同/检索未证实新增收益；不扩展检索库扫描。本实验不代表Q1从头重训，也不能把单折正差宣称完整OOF。
最终目标：严格第五>80.5且明显收益后才考虑其余折；完整OOF>82未达成。无旧队列恢复、其他折或交付覆盖。完成后根据实测决定是否有必要扩大当前有效方向。
每批更新PROJECTS.md、README、EXPLORATIONS与图谱；原子last_state保存，恢复须显式授权。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True,choices=('POR','SW'));a=ap.parse_args()
    active=jobs()
    if len(active)>=2:raise RuntimeError(('max2',active))
    out=ROOT/f'outputs/pure_dl_quality_readout_v37_{a.target}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    rules='# v37 工作约束\n仅固定基线POR/SW的Q50/等权输出头对照。遵守根AGENTS原分折、训练隔离、max80/pat18/max2GPU及交付不覆盖。维护PROJECTS.md记录目标、已完成、待做及后续。无自动续跑/补折。\n'
    for name,txt in [('AGENTS.md',rules),('PROJECTS.md',TEXT),('README.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','PROJECTS.md','README.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_quality_readout_v37.py'),'--target',a.target,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=err,stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,target=a.target,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v37 '+a.target+' 显式启动'
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','quality_readout_v37_20260929','- [v37 Q50与等权输出头](results/deep_learning/quality_readout_v37_20260929/README.md)：固定80.3365，POR/SW同容量质量权重对照；训练结果待定。')
    append(ROOT.parent/'AGENTS.md','## v37质量权重对照','## v37质量权重对照\n继续80.33授权，本批POR/SW的q50/uniform四项，旧预测器冻结、新11857参数全因子小头同初始化，仅损失质量权重不同。原分折/解码/max80-pat18/max2保持；无其他折或旧任务恢复，维护quality_readout_v37_20260929记录。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
