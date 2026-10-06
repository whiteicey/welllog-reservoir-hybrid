"""Explicit base versus parallel128 growth after the adopted v56 gain."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_density_private_v62 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v62 SW密度私有骨干与状态辅助监督

## 探索重点与依据
当前v56 SW73.713975628494/TOTAL80.354325927816；默认样本100786点官方SW97.883627，数值43879点18.198526。该验证分区只用于提出假设，不作逐行选择、训练权重或条件oracle。v59/v60冻结RQS头、v61插值均未改善，改为完整密度骨干与默认状态预测分离。
从v56第11轮复制两套完整SW网络：fixed全部冻结提供默认logit；density全骨干/FiLM/输出可训练，只提供连续密度。两组推理结构及完整初始参数相同，总1678962参数，其中839481在优化器。gate_aux对density自己的默认头加训练BCE；numeric_only系数0，仅数值密度梯度进入私有骨干。实际推理始终使用fixed的默认头，不用density的辅助默认预测。
两组损失=原Q加权[fixed默认BCE(常数)+非defaultGaussian NLL]/3 + alpha*原Q加权density默认BCE/3，alpha=1或0。保留常数项使日志与原NLL实现一致，它不产生梯度。numeric_only的129个辅助gate输出参数不接收梯度且不用于推理；名义容量相同，监督路径是变量。推理使用的密度分支参数为839352；不宣称两种目标梯度等价。
这不是旧模型投票：一个固定默认概率和一个私有条件密度组成单一SW分布，不读其他通道预测或验证标签。连续分布会影响最终动作，因此固定gate也不保证默认样本得分不变。v11曾在小型随机TCN上比较共享/分离，本轮是新冠军完整热启动、固定已验证gate、只改密度骨干辅助监督，不能将不同设置混同。

## 数据、执行与已实现
原9曲线/206、64训练井/16第五验证井144665点、Q50/seed20261010、1024随机锚点中心128、原8Gaussian/256支持点/800动作+默认评分保持。POR/PERM固定。训练标签仅原训练SW；整井bank只含输入，验证井仅自身推理，不使用标签拟合。
新AdamW3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps；两项显式启动，最多2GPU，无旧队列/其他折/交付覆盖。完整原子last_state含当前/最佳权重、优化器/调度/RNG/早停，恢复需显式。初始起点纳入选择，不算新训练收益。
已实现固定默认分支和私有密度分支，启动前无成绩。直接训练，真实批次检查固定分支无梯度、auxgate梯度与模式一致、私有骨干/头/FiLM有效；结尾固定权重哈希、源哈希、初始/最佳144665点及三列评分验证，不追加冗余预检或独立审查。采用或保留的模型需原始输入磁盘重建。

## 收益、待做和不继续
收益尚未证实。待两项训练、numeric_only-gate_aux配对差和相对当前分、记录参数/时间/显存、保留验证、清理负权重、PROJECTS/EXPLORATIONS/atlas。
若numeric_only无最终配对收益，不扫描当前辅助监督系数或复制支路数量；结论限该SW私有密度配置，不否定所有任务解耦。若两项都有更差结果则关闭此配置；不把起点回退当成功。重复开发折差未经过独立井确认。
单项与两任务墙钟分开，记录加载/输入库/预热/训练/评估和并发，代码与最终重建另列。第五严格>80.5及完整原五折OOF>82仍未完成；未经门槛与明确收益不补折，旧交付不改。目标完成后可独立确认并新目录交付。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_density_private_v62_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v62规则\n仅gate_aux/numeric_only，同v56完整函数，固定默认分支和私有密度骨干；仅辅助默认监督系数1/0不同，密度分支lr3e-5。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_density_private_v62.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v62 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_density_private_v62_20260929','- [v62 SW密度私有骨干](results/deep_learning/sw_density_private_v62_20260929/README.md)：同v56完整函数，固定默认分支及密度私有骨干，辅助默认监督1/0对照；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v62 SW密度私有骨干','## v62 SW密度私有骨干\n本批gate_aux/numeric_only，同v56完整函数，默认分支冻结，密度分支全骨干lr3e-5；只改私有密度的默认辅助监督1/0。原数据/Q/评分/80-18/max2，无旧任务/其他折/交付覆盖，维护sw_density_private_v62_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
