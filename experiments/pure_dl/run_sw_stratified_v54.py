"""Explicitly launch either arm; no automatic training queue."""
from datetime import datetime
import argparse,json,subprocess,sys
from sw_stratified_v54 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v54 SW状态分层采样

## 探索重点与固定对照
当前SW73.680963725271、组合TOTAL80.342771761689。v53的SW到POR迁移有局部收益但绝对分数不足，不继续该随机架构扩容。本轮转到训练随机性：默认状态占较多训练点时，随机窗口批次可能给非默认段不稳定的更新；检验固定分层覆盖是否改善泛化，不预断言方差或成绩一定下降。
uniform/stratified同836921参数、同v46 SW第9轮完整网络/头起点，全参数只用SW标签训练。uniform从所有原训练锚点有放回随机抽64个；stratified按锚点SW状态各有放回抽32个。状态分组仅用原64井训练标签，验证标签不参与采样，验证方式不变。
两组共同采用窗口内有效128中心点Q加权BCE+8高斯NLL的均值，再对64个窗口取均值并除3。这个平均方式与历史“批次全部有效点统一取均值”略有不同，两组一起改，不能把相对历史改进全归因于分层。
分层组每窗口乘2*N_state/N，权重由全训练锚点状态比例固定；不再按批次重归一化。分层平均的期望与uniform窗口风险相同，改变的是抽样组成及随机性。Q50保持且独立，未改默认/非默认的目标风险比例。中心锚点状态不等于窗口所有标签同状态；周围有效训练标签均正常参与。这里是状态采样，不是v14的井风险加权。

## 数据、训练与复现边界
原9曲线206、原训练64/验证16井、固定第五144665点、原评分/解码保持。1024上下文/中心128、训练侧预处理/Q、原始256支持点/800动作+默认；输入井库无标签/井ID特征，验证井库仅用于自身推理。网络初始化来自本折既有SW模型，无外部模型或旧预测投票；来源同开发折选模的局限保留。
共同新AdamWlr3e-5/wd.01/clip2，cosine80*64至1e-5，max80/pat18/min10/b64/64steps，seed20261010。初始函数与完整参数相同，训练样本不同是变量；不要求轨迹相同。原子last_state含当前/最佳模型、优化器、调度器、RNG、早停，仅显式恢复。最多2GPU，无自动队列/旧任务/其他折/交付覆盖。

## 已做、待做与收益判定
已实现训练侧状态池和概率校正，以及共同窗口风险；开始前尚无成绩。直接真实训练，记录状态池数量/哈希、实际批次状态数/权重、梯度、曲线和源哈希，不做额外逐试验数值预检或独立审查。
待做两组训练、最佳恢复、完整三列实际评分、局部配对差与相对当前绝对差分开、候选保留或清理、PROJECTS/README/EXPLORATIONS/atlas更新。有收益候选磁盘重建；无收益不扫32/32比例/学习率，不将本配置负结果泛化所有采样方法。含初始回退选择无改善时，不以训练后两个较差成绩的差值冒称可采用的收益。
按单项与双并发总墙钟分别记录耗时/显存，含加载/预热/输入库/启动间隔，代码与收尾另计。第五严格>80.5且明显收益再考虑其余折；完整原五折OOF>82、独立井确认、新目录交付仍未完成。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_sw_stratified_v54_SW_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for name,txt in [('AGENTS.md','# v54规则\n仅uniform/stratified两项，v46SW同函数全骨干训练；训练锚点状态32/32分层加概率校正与统一抽样对照，共同窗口平均风险。原数据/Q/评分/max80-pat18/max2保持，无其他折/旧队列/交付覆盖。维护PROJECTS目标/已做/未做/后续。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/name).exists():(STUDY/name).write_text(txt,encoding='utf-8')
    out.mkdir()
    for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_stratified_v54.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=proc.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v54 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','sw_stratified_v54_20260929','- [v54 SW状态分层采样](results/deep_learning/sw_stratified_v54_20260929/README.md)：同v46完整SW起点，uniform与32/32状态分层+概率校正，共同窗口风险；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v54 SW状态分层采样','## v54 SW状态分层采样\n本批uniform/stratified两项，同v46SW完整起点，全参数训练；64随机或32/32状态锚点，分层损失按采样概率校正。两组共同窗口均值风险，原数据/Q/评分/80-18/max2不变；无其他折/旧队列/交付覆盖，维护sw_stratified_v54_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
