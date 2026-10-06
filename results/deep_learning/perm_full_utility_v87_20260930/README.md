# v87 PERM完整私有网络与联合动作效用

## 探索重点和固定对照
当前独立三通道原第五折TOTAL80.382093291214，PERM89.396724049366。v86 POR冻结专家/只路由两项无绝对或匹配收益，停止其具体范围；SW全网络效用已有收益，为跨目标验证提供假设，不能先验认为PERM也有效。
本批预定义likelihood/joint两项，均从v8 PERM双专家+v26 gate+v28 joint+v29 coupled当前完整函数初始化，旧epoch2路由和所有依赖不改。无额外参数，不重置预测或先验。两组完整私有backbone、内部输入路由、PERM读出与保留的小头更新，解除旧no_grad/detach接口；旧其他目标packed输出行仍是buffer，不进入优化器，不用其他目标训练标签或预测。
两组只有训练效用系数0/1不同，计算同一个损失图。原Q50的PERM BCE+GaussianNLL /3，加Q50的soft动作遗憾/3；温度0.05。PERM官方核max(0,1-|log10动作-log10真值|)，不能套用POR/SW相对容差；log10归一化和训练median/scale不改，默认0.01。原256支持/800几何动作+默认/单覆盖解码保持。可微Gaussian CDF以安全无穷尾处理，512点梯度checkpoint控制激活显存；仅训练目标改变，动作softmax是原argmax的代理。

## 已探索与验证
当前已实现、尚未启动，无新成绩。直接训练，无独立数值预检/审查。真实初始全144665点预测应与原PERM精确相同；实际第二batch记录两专家、输入路由、原读出修正/v28/v29全部梯度与效用gate/density梯度，确保不只是接口声称可训练。完整buffer不变、来源hash、最佳恢复和实际原三列评价在结束核对；有收益模型原始输入磁盘重建。
原9曲线206+既有训练侧幅值63、原64/16井/578891训练/144665验证/seed20261010/Q50保持。新AdamW骨干与输入路由1e-5、读出和小头3e-5，wd0.01/分组clip2；80上限/pat18/min10/bs64/64步，cos80*64/eta1e-5，因此body1e-5固定在下界、head3e-5渐降1e-5。新优化器而非旧训练轨迹续跑，dropout随完整模型训练模式恢复，评估关闭；两组同初始化及seed/window策略，不宣称CUDA逐位轨迹相同。
原子last完整state、优化器、scheduler、RNG、早停、当前/最佳，恢复显式授权。

## 收益和不继续
收益尚未证实。最终独立报告joint-likelihood、对当前绝对差、默认/数值/各井以及固定POR/SW实际TOTAL。按整个PERM官方分选择checkpoint；不按行挑模型或用未知测试标签。匹配机制增益和绝对超越分别报告，不把epoch0恢复当新收益。两项都无收益则关闭本warm-start/全网络/LR/系数1/温度组合，不临时无限扫权重；适用范围不能泛化所有PERM效用训练。

## 已做、计划、未做及目标后
已做：新模型/可导捕获/训练接口、两项固定协议。计划：显式两项训练、实际组合、正重载、负清单、图谱。未做：真实成绩、独立泛化、其他折；当前不补折。最多2GPU训练，先运行一项，依据实测GPU空闲和峰值决定第二项是否并发，不设自动队列。记录单项训练墙钟、两项同工作量总墙钟、串并行、GPU allocated/reserved与全卡快照，并区分预热/数据/验证/保存/准备收尾。
目标后：原第五严格>80.5且明确收益再考虑其余四折，完成原OOF>82后新目录交付与独立井确认。原冠军、正对照和旧交付不变。


## 最终结果
# v87 PERM完整网络联合效用结果

独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|PERM|固定POR/SW TOTAL|最佳/停止轮|更新后最高PERM|
|---|---:|---:|---:|---:|
|likelihood|89.396724049366|80.382093291214|0/18|89.241381002864|
|joint|89.396724049366|80.382093291214|0/18|89.257370007363|

## 探索与验证
同1324879学习参数、完整当前函数同起点；原骨干/输入路由/本目标头和小修正均实际更新，其他目标packed行仍是buffer。
原log10 Gaussian/默认0.01/官方log误差/原支持动作/分折/Q50不改；新body1e-5、head3e-5、80/18、温度.05效用0/1。实际梯度、初始144665一致、source/预处理/评分hash、buffer、最佳恢复、原行序和实际三列评价通过；正候选重载=[]。
CUDA线性上采样backward只支持warn_only确定性模式，不宣称逐位训练复现。

## 收益与不继续
最终joint-likelihood PERM +0.000000000000；排除起点的更新后最高差 +0.015989004499（诊断独立报告，非当前采用收益）。采用=None。恢复第0轮不算新增收益。
likelihood 默认/数值差={"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.24399196112769, "channel_delta": 0.0, "TOTAL_contribution": 0.0}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.13837825814508, "channel_delta": 0.0, "TOTAL_contribution": 0.0}}；实测停止=patience18
joint 默认/数值差={"default": {"n": 95533, "reference": 97.24399196112769, "candidate": 97.24399196112769, "channel_delta": 0.0, "TOTAL_contribution": 0.0}, "numeric": {"n": 49132, "reference": 74.13837825814508, "candidate": 74.13837825814508, "channel_delta": 0.0, "TOTAL_contribution": 0.0}}；实测停止=patience18
无绝对或匹配最终收益时关闭本源/全网络/低LR/系数1/温度.05组合，不继续临时扫效用或延长预算；结论不泛化所有PERM训练。
一个反复开发的第五折不是完整OOF或独立泛化证明。

## 成本、清理、未完成与目标后
{
  "candidates": [
    {
      "mode": "likelihood",
      "start": "2026-09-30T12:48:03.378445+08:00",
      "end": "2026-09-30T12:53:07.288306+08:00",
      "seconds": 298.9389990000054,
      "peak_allocated_MiB": 917.45361328125,
      "peak_reserved_MiB": 1022.0,
      "launch_to_completion_seconds": 303.909861
    },
    {
      "mode": "joint",
      "start": "2026-09-30T12:51:25.899575+08:00",
      "end": "2026-09-30T12:56:33.055661+08:00",
      "seconds": 301.050231399975,
      "peak_allocated_MiB": 917.45361328125,
      "peak_reserved_MiB": 1022.0,
      "launch_to_completion_seconds": 307.156086
    }
  ],
  "first_launch_to_last_completion_seconds": 509.677216,
  "first_launch_to_verified_seconds": 722.222559,
  "process_overlap_seconds": 101.388731,
  "peak_training_processes": 2,
  "sum_individual_launch_seconds": 611.065947,
  "finalize_seconds": 0.40619530002004467,
  "scope": "input loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring and atlas separate; not a capacity benchmark"
}
删除4个冗余权重共57604292字节，正源和旧交付不改。
未运行其他折、旧队列。仍需原第五严格>80.5且明确收益再考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。


## 收尾与结论范围
更新后最高的+0.015989004499发生在似然第1轮/效用第2轮，不能写成同一轮差。18个同轮比较只有6轮效用更高、12轮更低，平均差-0.002843593202，最后一轮-0.019259906293。因此存在最高值的小正诊断，但没有一致轨迹改善或采用收益，当前完整私有微调设置不继续扫描。源/config/学习曲线保留；负权重4个57604292字节已按清单清理。图谱632节点、87个dead end，主节点显示原第五实际固定POR/SW TOTAL80.382093291214及恢复起点，不冒称新收益。原正源/交付保留，没有GPU训练或其他折。
