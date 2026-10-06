# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v93 SW两次计算匹配的SAM优化

探索重点：v92完整POR低LR效用两项负（排除起点joint-likelihood -0.029103、同轮2升16降/均值-0.040396），结束该组合。回到当前SW有效全训练Gaussian/效用1/DropKey.4，在839481参数同架构上改变优化几何。SAM提出同时压低训练损失及其参数邻域损失；本批仅rho0控制与rho.05两项。来源论文：https://arxiv.org/abs/2010.01412；官方仓库：https://github.com/google-research/sam。不下载released模型或额外数据。此处是自编PyTorch一阶L2 SAM加AdamW变体，不冒称论文原数据复现。

固定对照：两项从原v56训练折内骨干/FiLM、原64井SW训练先验重置开始，不加载v82训练后模型；当前SW73.793310952488/固定POR-PERM TOTAL80.382093291214只作绝对历史对照。两项都每次更新两次前后向，完全同batch、回放同DropKey随机态，首梯度全局未clip L2归一化扰动，第二损失梯度用于AdamW更新，更新前从clone精确恢复参数（不用减扰动近似）。rho0与.05计算工作量/更新步/窗口人口匹配；第二forward结束恢复首forward之后RNG，随机流每更新只消费一轮DropKey。不把新增计算当免费，历史单次AdamW与重跑两次控制不声称逐位CUDA训练相同。每更新最后梯度clip2，wd.01/1e-4/cos80*64到1e-5/max80pat18min10/b64/64steps。

原9曲线206+相对深度/训练侧全井128输入bank/原64-16/578891训练/144665第五/seed20261010/Q50/评分、GaussianNLL+效用1/3、温度.05/256支持800动作+default99.9完全不改。只有自身SW训练标签；POR/PERM固定，不使用外部预测/投票/未知测试标签。学习全SW网络，无新参数。SAM无验证梯度或额外Q拟合。

已探索与验证：建立时未训练，收益未证实。正常初始化raw state/旧头/初始动作hash对原Gaussian，实际第二batch记录全网络/效用梯度、请求与实际扰动L2和同mask，原子last在完整更新后保存；保持best/更新后best/optim/sched/RNG/停止，恢复显式。最终核对源/预处理/评分和全144665恢复、实际三列，正绝对或完整固定匹配收益需要两侧磁盘重载。无独立数值预检/审查/重复原Q。

收益与不继续：预定义sam005-control完整训练后最佳差与相对历史Gaussian绝对差分开报告，补同轮轨迹/默认数值/各井。自身较差先验改善不是超越当前。无两种收益则关闭本rho.05/原始SW来源/AdamW/DropKey.4/预算变体，不临时扫rho/LR/ASAM；不泛化全部SAM。

已做：两次计算匹配的有限方案。计划：显式分批训练、真实组合/正重载/清理/图谱。未做：真实收益/其他折/独立泛化。normal最多2GPU，以实测显存启第二项；记录单项数据预热/两次前后向/验证保存、同工作量总墙钟/并行/al

_CUDA_DSA` to enable device-side assertions.\\n')", "launch_to_terminal_seconds": 644.860525}, {"mode": "sam005", "pid": 25968, "start": "2026-09-30T15:02:42.390749+08:00", "end": "2026-09-30T15:10:51.305602+08:00", "end_basis": "failure.json modification timestamp; process subsequently confirmed absent", "status": "failed", "last_complete_epoch": 17, "error": "RuntimeError('CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`')", "launch_to_terminal_seconds": 488.914853}, {"mode": "sam005", "pid": 72556, "start": "2026-09-30T15:16:48.639902+08:00", "status": "complete", "serial_recovery": true, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "sam005", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_sam005_fold5", "--resume"], "end": "2026-09-30T15:23:39.620693+08:00", "end_basis": "final report timestamp; fresh process inventory empty", "launch_to_terminal_seconds": 410.980791}, {"mode": "control", "pid": 53900, "start": "2026-09-30T15:25:38.053973+08:00", "status": "complete", "serial_recovery": true, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "control", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_control_fold5", "--resume"], "end": "2026-09-30T15:28:17.701428+08:00", "end_basis": "final report timestamp; fresh process inventory empty", "launch_to_terminal_seconds": 159.647455}], "first_launch_to_last_completion_seconds": 1691.256351, "first_launch_to_verified_seconds": 1806.332006, "peak_process_concurrency": 2, "approximate_overlap_seconds": 488.914853, "sum_process_segment_seconds": 1704.403624, "finalization_seconds": 4.524683399999049, "concurrent_attempt_failed": true, "recovered_serially": true, "unpersisted_failed_batches_in_report_seconds": false, "scope": "actual segments use failure file and final report timestamps, not logical run lifespans; total includes failures/resume gaps; report seconds include only saved complete epochs/loading/warmup/eval/save; no safe concurrency/capacity claim"}
负项清理4个权重，共38121804字节；完整删除计划先落盘，正源/比较依赖/旧交付保留。
已做：真实两项、必要重载、实际组合、记录清理。未做：其他折和独立泛化；第五严格>80.5且明确收益才考虑其余四折，原完整OOF>82仍未达成。目标后独立井和新目录交付。



## FINAL.md

# v93 SW两次计算匹配SAM结果

当前独立路线TOTAL 80.382093291214 → 80.417837208128。

|读出|SW|固定POR/PERM TOTAL|最佳/停止|
|---|---:|---:|---:|
|control|73.785877274491|80.379491503915|25/43|
|sam005|73.895436429384|80.417837208128|30/48|

探索与验证：两项同839481参数、v56骨干/FiLM+训练先验重置，raw全state/旧头和144665点初始动作匹配历史Gaussian。原9曲线206/全井bank/分折/Q50/Gaussian/效用1/DropKey.4/1e-4/80-18保持。两项都两次前后向、同batch/RNG回放DropKey，首未clip全局L2梯度构造rho0/.05扰动，第二梯度AdamW，先clone精确恢复再更新；实际训练扰动半径/全网络和效用梯度/source/评分/预处理/最佳恢复/三列评价通过，没有独立预检或重复Q。历史单forward控制不声称同轨迹；本批rho0工作量匹配重跑。原始输入磁盘重建=[{'mode': 'control', 'rows': 144665, 'fresh_disk_replay_exact': True, 'score': 73.78587727449113, 'sha256': 'a7c6d27882aeaec972767ff13e96acd177e52fd6ca88bdad5d42acb3ecade9be'}, {'mode': 'sam005', 'rows': 144665, 'fresh_disk_replay_exact': True, 'score': 73.89543642938351, 'sha256': '306fc7cfd8975d872ce1c43cfdde8d7ecb931f848bf62e96082cca7319884fdd'}]
方法来源：[SAM论文](https://arxiv.org/abs/2010.01412)与官方google-research/sam仓库；自编AdamW变体，不声称原论文任务复现。旧v55 836921参数/3e-5/原模型warm-start SAM best0负证据见HISTORICAL.md；新配方不复活旧队列。
control对Gaussian SW-0.007433677997/TOTAL-0.002601787299，默认+0.012224814056/数值-0.052587413952，井升/降8/8。
sam005对Gaussian SW+0.102125476896/TOTAL+0.035743916913，默认+0.608445693533/数值-1.060845177723，井升/降9/7。

收益：sam005-control SW+0.109559154892，完整训练后最佳匹配差+0.109559154892，采用=sam005；绝对和公平局部收益分开，局部小正不等同SOTA或独立泛化。
不继续：若无绝对或完整匹配收益，关闭本v56-prior/Gaussian/效用1/DropKey.4/AdamW/

E_CUDA_DSA` to enable device-side assertions.\\n')", "launch_to_terminal_seconds": 644.860525}, {"mode": "sam005", "pid": 25968, "start": "2026-09-30T15:02:42.390749+08:00", "end": "2026-09-30T15:10:51.305602+08:00", "end_basis": "failure.json modification timestamp; process subsequently confirmed absent", "status": "failed", "last_complete_epoch": 17, "error": "RuntimeError('CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`')", "launch_to_terminal_seconds": 488.914853}, {"mode": "sam005", "pid": 72556, "start": "2026-09-30T15:16:48.639902+08:00", "status": "complete", "serial_recovery": true, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "sam005", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_sam005_fold5", "--resume"], "end": "2026-09-30T15:23:39.620693+08:00", "end_basis": "final report timestamp; fresh process inventory empty", "launch_to_terminal_seconds": 410.980791}, {"mode": "control", "pid": 53900, "start": "2026-09-30T15:25:38.053973+08:00", "status": "complete", "serial_recovery": true, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "control", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_control_fold5", "--resume"], "end": "2026-09-30T15:28:17.701428+08:00", "end_basis": "final report timestamp; fresh process inventory empty", "launch_to_terminal_seconds": 159.647455}], "first_launch_to_last_completion_seconds": 1691.256351, "first_launch_to_verified_seconds": 1806.332006, "peak_process_concurrency": 2, "approximate_overlap_seconds": 488.914853, "sum_process_segment_seconds": 1704.403624, "finalization_seconds": 4.524683399999049, "concurrent_attempt_failed": true, "recovered_serially": true, "unpersisted_failed_batches_in_report_seconds": false, "scope": "actual segments use failure file and final report timestamps, not logical run lifespans; total includes failures/resume gaps; report seconds include only saved complete epochs/loading/warmup/eval/save; no safe concurrency/capacity claim"}
负项清理4个权重，共38121804字节；完整删除计划先落盘，正源/比较依赖/旧交付保留。
已做：真实两项、必要重载、实际组合、记录清理。未做：其他折和独立泛化；第五严格>80.5且明确收益才考虑其余四折，原完整OOF>82仍未达成。目标后独立井和新目录交付。


