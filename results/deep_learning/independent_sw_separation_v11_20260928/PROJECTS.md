# 进度

目标：用同参数连接方式对照验证SW内部解耦是否改善非默认段及总分。
已完成：核实v10两模型结束且未超旧SW；实现两套TCN共享/分离读出与训练接口。
计划：显式启动两候选，完成共同评分、每井/状态误差分解和训练侧梯度诊断。
未完成：本批训练收益、磁盘重载、独立新井泛化、完整OOF>82。
达成后：固定有益模型再评估推广；不自动补折，不覆盖旧交付。


## 2026-09-28T14:11:30.433442+08:00 SW共享/分離v11完成
sharedSW=73.093512782，splitSW=72.927481759；保留TOTAL=80.274135337791，使用新SW=False。同参/初始化已核对；诊断与磁盘恢复完成，非独立OOF结论。
详见results/deep_learning/independent_sw_separation_v11_20260928/FINAL.md。无新补折或交付变更；下一步尚未启动。


## 本轮结论与下一步限定
两项训练及磁盘重载完成，无运行中的本批任务。sharedSW=73.093512781564，splitSW=72.927481759099。同参数与同初始化状态已实核，split相对shared的等价TOTAL变化-0.058110857863，6/16井提高、10井下降。选择未校正的探索性井重采样区间为[-0.1688249200617162, 0.05241575243032869]，不能据此宣布新井上的普遍结构排序。
拆分后真实非默认段通道分+0.096188933，真实默认段-0.280193202。非默认段的小幅改善不足以补偿默认段损失。门控0.5阈值召回提高，但门控与密度一起决定最终动作，分类准确率不等价于官方分数。
在真实非默认段上仅用连续分布解码（将门控质量设0的诊断），split较shared为-0.311991269。这是标签限定子集的离线诊断，没有用于选择逐点模型或推理；不能与旧模型未经同样处理的普通得分直接比较。
共享组128个固定训练窗口中，2/8批BCE/CE编码器梯度余弦为负，均值0.146004。split跨任务梯度严格为零，但验证未提高。这个小样本结果不支持将当前瓶颈直接归结为普遍梯度冲突；共享信息和每任务有效容量也随连接方式改变。
本次只关闭相同模型/预算附近的强制完全分离扫描，不推论所有独立编码器无效；POR/PERM/SW三个目标之间继续使用私有模型。
下一步尚未启动：保持一种固定SW表示，比较原BCE+条件CE与把默认动作竞争也纳入的联合效用监督。与v9仅对冻结门控下连续分布加效用项的试验区分；不调验证集阈值，不追加自动队列或补折。
当前继续保留已验证三私有模型TOTAL=80.274135337791。本批新模型未替换旧交付；第五折严格>80.5和完整OOF>82仍未完成。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
