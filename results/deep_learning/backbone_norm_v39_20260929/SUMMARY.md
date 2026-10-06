# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v39 SW骨干归一化仿射更新

## 探索重点
当前第五折TOTAL80.336513436816，SW73.663082797066。v37/v38冻结骨干的新输出头未能改善，转而更新当前私有双专家骨干已有归一化层的scale/bias。
decoder_norm仅两个专家dec0/dec1/dec2中的GroupNorm/LayerNorm仿射参数；all_norm更新两个专家所有骨干归一化仿射参数。两个范围的参数量不同，是更新范围对照，不是同容量结构消融。无新增参数、无宽度扩展、无冷启动。两者均从同一已验证权重出发。
原卷积/注意力投影/头/路由/校准权重固定；其计算对骨干特征保持可微。移除旧推理捕获的detach/no_grad路径，保留原前向数值与eval/dropout模式。原输入、相对深度、63幅值侧视图、路由和解码保持。

## 固定训练条件
仅SW自身原NLL/3，Q50不改，原64训练井/16第五验证井144665点，原9曲线/206特征。batch64、每轮64步、max80/patience18/min10；AdamW lr1e-4至cosine80*64末端1e-5，无norm权重衰减，clip2。所有归一化都是GroupNorm/LayerNorm，不更新任何批运行统计，也不在验证上更新参数。
两候选各一个GPU进程，最大并发2；独立优化器/早停/完整原子状态。每轮原官方SW分选最佳，最后恢复并与固定POR/PERM实际组装。无需所有目标同时提高。

## 已完成、待做及收益
已实现可微SW骨干读出和两个显式更新范围；待运行结果。收益尚未证实。首批真实训练记录各可训练参数梯度，最终检查冻结参数与最佳预测恢复，不另开预检/审查流程。
不再延伸v37全因子NLL小头和v38直接奖励小头的扫描；本次负结果若出现，只限本更新范围与学习率设置。
最终完整OOF>82未完成；严格第五>80.5且明显收益才考虑其余折。本批无其他折、旧任务或交付覆盖。维护PROJECTS.md/README/EXPLORATIONS/图谱；训练恢复须显式启动，不自动排队。


## v39骨干归一化对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见backbone_norm_v39_20260929/FINAL.md。




## FINAL.md

# v39 SW骨干归一化仿射：最终结果

第五折TOTAL 80.336513436816 → 80.336513436816，差值 +0.000000000000。

|更新范围|可训练参数|训练后最高SW|含原模型回退SW|最佳/停止轮|
|---|---:|---:|---:|---|
|decoder_norm|1152|73.650652113|73.663082797|0/18|
|all_norm|5120|73.645195910|73.663082797|0/18|

## 探索重点与验证
从当前已验证SW权重起步，仅更新两个私有专家的归一化仿射。卷积、头、路由权重不变，没有新增容量。训练前向保持可微，原梯度分离捕获被替换；仅自身NLL和原Q50。原动作、输入和评分保持。
两个更新范围参数量不同，不能解释成同容量结构消融。第一真实训练批确认每个所选参数有梯度、冻结参数无梯度；最终冻结状态及144665点最佳恢复通过。
新采用SW完整磁盘重建：False；实际固定POR/PERM后的三列评分完成。

## 收益与不继续方向
全骨干-解码端：含回退SW差+0.000000000000，训练后最佳差-0.005456202996。采用=None。
第0轮回退不是训练收益；正比较仍是反复使用的第五开发折证据，不是独立新井泛化。
若两个范围均无新增收益，关闭本次SW归一化仿射微调设置，不继续扩大同范围学习率扫描；不能据此否定所有骨干适配。

## 耗时、目标及未做
同工作量墙钟182.47秒，实际峰值并发2；单项耗时/显存见timing.json。
PyTorch本地CUDA线性上采样反向不保证确定性，已记录warn_only；不声称训练逐位可重复。验证和权重恢复按原确定性路径核对。
完整原五折OOF>82未完成；本批未动POR/PERM、未启动其他折/旧任务、未覆盖交付。




