# v70 当前冠军PERM条件CDF校准

## 探索重点与固定对照
转回当前冠军PERM v29 coupled89.396724049366，冻结全部双专家/原读出/路由，不继续RQS负优化设置。仅新增10306参数小头，读取两个128维隐藏及已有63幅值侧特征；Conv319→32→2，affine中间恒等，nonlinear中间GELU，同参数张量与零输出初始化。两组原始完整预测相同。
头产生u/v=0.25*tanh(output)，变换G(F)=F+uF(1-F)+vF(1-F)(1-2F)。端点G(0)=0/G(1)=1；导数1+u(1-2F)+v(1-6F+6F²)，因两个基导数绝对值≤1且|u|/|v|≤.25，导数≥.5。训练新连续密度logpdf=原Gaussian mixture logpdf+log G'(F)；默认门控原值冻结。不是验证分数校准或标签作输入，两个神经头仅训练64井PERM似然/Q50。
验证沿原256格概率累计作G变换，并以原概率+校正CDF差的残差形式保持零头逐位一致；原800动作/default/评分不变。端点校正为0、总质量守恒，浮点误差clamp非负，不再次归一化破坏零头身份。无需改变原混合分量位置或尺度。

## 训练边界与已做
原9曲线/206输入及已有63侧特征/原64训练井和第五16井144665点/seed20261010/Q50/256随机窗口中心128不变，无额外特征数据或其他模型投票。原模型完全冻结eval，新增头AdamW3e-4/wd.01/clip2、cosine80*64至1e-5、max80/pat18/min10/b64/64steps。初始模型纳入选择，回退不算新收益；完整原子last_state保存头/优化器/调度/RNG/停止，仅显式恢复。
已实现，启动时无成绩，直接训练。实际新头梯度/原模型状态不变、初始/最佳144665点、来源哈希和三列评分作为结果证据；保留正候选前原始输入磁盘重建，不做独立数值预检/审查。

## 收益、待做与不继续
待两项实测/配对差/当前绝对差、负权重清理、PROJECTS/EXPLORATIONS/图谱。若非线性无最终配对收益，不扫本头宽度/激活；若两者都无收益，不扫当前多项式阶数/幅度。结论只限这个条件CDF变换，不泛化全部分布校准。第0轮回退不算学习收益，开发折仍需独立确认。
最多2GPU，显存允许时双跑，无自动队列；分别记录单项及两项实际总墙钟、并发与显存，含加载/预热/训练/评估，代码/收尾另计。先清单后删负权重，保留所有既有正候选/交付。第五严格>80.5且明显收益后才考虑其他折，完整原五折OOF>82未完成；无其他折/旧任务/交付覆盖。目标后独立确认、新目录交付。


## v70 affine显式启动
{"pid": 56364, "mode": "affine", "started_at": "2026-09-29T23:10:50.457158+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_cdfwarp_v70.py", "--mode", "affine", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_cdfwarp_v70_PERM_affine_fold5"], "automatic_queue": false}


## v70 nonlinear显式启动
{"pid": 60140, "mode": "nonlinear", "started_at": "2026-09-29T23:13:28.380994+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_cdfwarp_v70.py", "--mode", "nonlinear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_cdfwarp_v70_PERM_nonlinear_fold5"], "automatic_queue": false}


## v70最终结果
两项完成，TOTAL=80.354325927816，nonlinear-affine PERM=+0.000000000000（训练后差+0.000421751979）。采用=None，保留[]，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见perm_cdfwarp_v70_20260929/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
