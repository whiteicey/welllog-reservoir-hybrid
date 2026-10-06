# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v156 沿PERM联合监督正收益扩大系数

探索重点：PERM v154同容量正收益+0.017628761135，以同一有效128位置/权重头仅提高jointRPS系数1->2。POR系数2负不泛化为PERM必负；也不预设此处收益。

已经探索与验证：实现完成，尚未启动、无成绩。同v15482720参数/原SOURCE初始化/seed/优化器/80-18预算/lr，仅NLL+2RPS改变。原PERMGaussian、log10、DEFAULTS/训练median-scale、255有限边界联合CDF真实默认/连续监督、Q50及/3归一化、1024行gradient checkpoint保持，不改动作/区间/解码阈值。未加载v154/v136训练头，替换PERM阶段，不追加第六阶段。

收益：尚未证实。恢复后与系数1PERM89.405564856919、NLL-only89.387936095784比较，固定POR151/SW104实际合并。已有PERM1收益来自非默认+0.055721972822、默认-0.001962329657、7/16井正；单折单种子、小收益，不泛化独立井。新尝试不能将初始学习当新增益。

不值得继续：POR2比POR1低0.024725743891，已关闭；负CDF/通信/专家交互/扩宽机制不堆叠。本批PERM只预定义系数2一项，不自动开系数网格。SW联合RPS正在独立训练，结果未完成。

约束：原9曲线/206+63/64训练井统计/Q50/原第五折/seed20261010、冻结骨干/门控/尺度、自身PERM真实标签；normal最多2GPU、80/18/min10/b64/64步/3e-4/cos80*64，完整原子last/显式恢复/进程级防睡眠。source+POR113+POR151+PERM156+SW104共5阶段，严格第五TOTAL>80.8且收益后补折，完整OOF>82未完成。正控制/旧交付不改。

待完成：空位显式启动，与SW155总任务<=2；最佳144665磁盘恢复/冻结完整性/实际合并，记录单项/并发总墙钟/显存/预热，最终结论、负权重SHA清理及PROJECTS/图谱。

RUNTIME FAILURE: CUDA illegalmemory access duringbackward after completeepoch8; notOOM, not a mechanism negative/convergence result. PID72544 absent, no autoresume. Preserve atomiclast/source/config/log/error for diagnosis. Earlier launchsuccess records remain historical.




