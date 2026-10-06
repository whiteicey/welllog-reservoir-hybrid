# DL 演进图谱进展

- 目标：保留从早期纯神经网络到当前特征探索的路线、结果与失败证据，提供可离线浏览、可继续维护的图谱。
- 已完成：从当前留存目录提取实验/子配置；中文主线标注、源码继承及融合来源；独立 dead end 节点；搜索、章节过滤、拖拽缩放、全图导航、列表、节点档案和源文件链接；AGENTS.md 持续维护规则。
- 已完成：浏览器实际验证桌面/窄屏、搜索→详情→追溯→定位、全部节点、空结果和关闭路线。GR条件特征已自然早停，第五折79.9923191，结论已并入图谱。
- 待持续完成：新实验更新；逐步确认自动识别不足的父关系、修复历史文字说明；若以后归并同类变体，保留每次配置与分数。
- 范围限制：当前留存记录可追溯，不宣称已恢复删除且没有备份的历史。没有父关系证据时不强行连线；源码结构继承不是收益的因果证明。
- 完成后可选：独立发布网页、增加按训练预算的对照视图、导出局部子图。此次没有推送远程或上传权重/数据。

## 主题泳道与单向演进修订

- 已按用户意见改为每行一个主题。同主题只向右演进；不同主题之间通过跨行边分叉/融合。
- 排列由有向无环血缘关系和同主题顺序共同约束，不再按主题分列。下游节点必定位于上游卡片右边界之后。
- 同行跨过多个节点的边从卡片上方经过，所有路径横坐标单调前进；未将无证据的相邻节点连成伪血缘。
- 已用真实浏览器检查全部649条边：反向边0，回头路径0；8个主题均只占一行。搜索、追溯、桌面和手机交互仍通过。


- Verified live local-feature-gate: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:10:00+08:00", "epoch": 4, "best_epoch": 4, "best_score": 77.05803312821072, "control_same_epoch_best": 76.62364985406369, "matched_delta": 0.4343832741470379, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; initial results only, no established benefit"}. Actual1329491 parameters, +373 vs compact control. First printed elapsed60.8s at1 includes preprocessing/probes, no separate warmup; not final elapsed. No OOM observed.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:12:29+08:00", "epoch": 11, "best_epoch": 11, "best_score": 78.4294059904545, "control_same_epoch_best": 78.25898459131336, "matched_delta": 0.17042139914113363, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; early lead not stable final gain"}. Keep original protocol; no second launch or restart. Fifth>80.4/fullOOF>82 not achieved.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:22:28+08:00", "epoch": 36, "best_epoch": 36, "best_score": 79.68708215484588, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.03708351027057688, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; small middle-stage difference is not final benefit"}. No restart/extraGPU trial; no fifth804 or fullOOF82 qualification. Last printed fold elapsed868.8s at35 includes preprocessing/probes and concurrent documentation/browser work, no separate warmup; not final elapsed.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:25:17+08:00", "epoch": 44, "best_epoch": 44, "best_score": 80.03050256733324, "control_same_epoch_best": 79.89736194351701, "matched_delta": 0.1331406238162316, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; no stable late-stage gain established"}. Original training unchanged. Last printed wall979.2s at40 includes preprocessing/probes, no separate warmup; not final elapsed. FullOOF goal unmet.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:29:00+08:00", "epoch": 54, "best_epoch": 48, "best_score": 80.03119287568968, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.19582225983938883, "bad_epochs": 6, "observed_peak_gpu_mib": 795.0791015625, "concurrency": "soleGPU training", "decision": "continue briefly; early lead has not become late gain, terminate if stagnation persists; no gate parameter sweep"}. Original baseline80.2508938 also remains higher; no promotion. Last printed wall1199.2s at50 includes preprocessing/probes, no separate warmup; not final elapsed. FullOOF>82 unmet.


- Local-feature-gate80 early terminated: {"status": "early_terminated_negative", "atlas_current_trial": false, "session_id": 46097, "pid": 88964, "exit_code": 1, "epoch": 58, "best_epoch": 48, "best_score": 80.03119287568968, "bad_epochs": 10, "delta_original": -0.21970089912954904, "delta_compact": -0.19582225983938883, "best_checkpoint_restore_verified": false, "observed_gpu_peak_mib": 795.826171875, "checkpoint_files_retained": 0, "reason": "Late stagnation for10 epochs; observed best below both original and compact controls; user-authorized early termination. No gate-width/embedding/gain sweep."}. Partial trial only; final restored-best verification not completed. Source/config/validation history retained, no checkpoints. Last printed fold wall1326.3s at55 includes preprocessing/probes and concurrent browser/doc work, soleGPU training, no separate warmup; not final elapsed. No remainingfolds. Preserve80.3868919215 five-model recipe. Full original OOF>82 unmet; next exploration must change mechanism rather than gate parameters.


- Context-calibration80 launched session95689: second expert calibration hidden h gains zero-init32x32*(masked_mean17(h)-h), +1024parameters. Original206 inputs/68view, original full expert/router/loss/Q/seed/split/decoder unchanged. Sole requested GPU task,80max/18patience/bs64/LR.0006cosine, strictfifth>80.4 before remainingfolds. No gain yet, no extra preflight. Distinct from raw-curve smoothing/local robust hand features/router context. Record outputs/pure_dl_paired_context_calibration80_sequence/README.md. No17-window/width grid if negative. Previous local-feature gate closed negative; fullOOF>82 unmet.


- Verified live context-calibration: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:37:16+08:00", "epoch": 2, "best_epoch": 2, "best_score": 76.63408234332434, "control_same_epoch_best": 76.62364985406369, "matched_delta": 0.010432489260651323, "bad_epochs": 0, "observed_peak_gpu_mib": 774.03564453125, "concurrency": "soleGPU training", "decision": "continue; initial contextual-calibration result only, no accuracy gain established"}. Actual1330142parameters (+1024); first fold elapsed67.5s at1 includes preprocessing/probes, no standalone warmup, not final elapsed. No OOM. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:39:10+08:00", "epoch": 7, "best_epoch": 6, "best_score": 77.72937727954321, "control_same_epoch_best": 77.74331901761342, "matched_delta": -0.013941738070201382, "bad_epochs": 1, "observed_peak_gpu_mib": 774.150390625, "concurrency": "soleGPU training", "decision": "continue; early result only, no established benefit"}. Last printed elapsed176.9s at5 includes preprocessing/probes, no standalone warmup; not final elapsed. Original protocol unchanged, no additional GPU job. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:40:58+08:00", "epoch": 12, "best_epoch": 12, "best_score": 78.46814231998061, "control_same_epoch_best": 78.48230718472027, "matched_delta": -0.014164864739655059, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; early comparison is not a final efficacy result"}. Last printed fold elapsed287.2s at10 includes preprocessing/probes, no separate warmup; not final elapsed. No restart, no otherGPU job, no promotion. FullOOF>82 remains outstanding.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:42:53+08:00", "epoch": 16, "best_epoch": 15, "best_score": 78.66626503594227, "control_same_epoch_best": 78.73471136844955, "matched_delta": -0.06844633250727838, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; no stable contextual-calibration gain established"}. Last printed fold elapsed422.9s at15 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:44:42+08:00", "epoch": 20, "best_epoch": 20, "best_score": 79.06045166721934, "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.03121245983416543, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; not enough evidence of lasting contextual-calibration benefit"}. Last printed fold elapsed535.6s at20 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion, fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:46:30+08:00", "epoch": 25, "best_epoch": 23, "best_score": 79.25280825505253, "control_same_epoch_best": 79.14794099275363, "matched_delta": 0.10486726229889598, "bad_epochs": 2, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; intermediate lead needs late-training confirmation"}. Protocol unchanged; no remainingfolds or secondGPU trial. Intermediate scores do not establish fullOOF improvement; original OOF>82 goal remains unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:48:09+08:00", "epoch": 29, "best_epoch": 28, "best_score": 79.35778300279328, "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.10894348378816687, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; middle-stage comparison does not establish final gain"}. Last printed elapsed673.1s at25 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:49:57+08:00", "epoch": 33, "best_epoch": 31, "best_score": 79.45711440123819, "control_same_epoch_best": 79.56187209225561, "matched_delta": -0.10475769101742571, "bad_epochs": 2, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; recent new best but no established gain over matched control"}. Last printed fold elapsed788.5s at30 includes preprocessing/probes, no separate warmup; not final elapsed. Protocol unchanged; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:51:53+08:00", "epoch": 38, "best_epoch": 37, "best_score": 79.72132588702519, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.002839778091271228, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; near matched control, no demonstrated final benefit"}. Last printed elapsed930.1s at35 includes preprocessing/probes, no separate warmup; not final elapsed. No protocol changes or promotion; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:53:39+08:00", "epoch": 42, "best_epoch": 42, "best_score": 79.78053044891338, "control_same_epoch_best": 79.76759879965745, "matched_delta": 0.012931649255932598, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; recent improvement but no stable matched-control gain"}. Last printed elapsed1041.8s at40 includes preprocessing/probes, no separate warmup; not final elapsed. No promotion or protocol change. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:55:30+08:00", "epoch": 48, "best_epoch": 47, "best_score": 79.91417887365749, "control_same_epoch_best": 79.93028980255141, "matched_delta": -0.01611092889392296, "bad_epochs": 1, "observed_peak_gpu_mib": 783.85595703125, "concurrency": "soleGPU training", "decision": "continue; near matched control, await later-stage comparison"}. Last printed fold elapsed1152.4s at45 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion and fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:57:16+08:00", "epoch": 54, "best_epoch": 52, "best_score": 80.18459751154603, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.04241762398304161, "bad_epochs": 2, "observed_peak_gpu_mib": 783.85595703125, "concurrency": "soleGPU training", "decision": "continue after new best; still below original and compact baselines, no promotion"}. Last printed elapsed1235.9s at50 includes preprocessing/probes, no separate warmup; not final elapsed. No proven gain or qualification; fullOOF>82 unmet.


- USER OVERRIDE for context-calibration80: continue this exact active trial to configured completion; withdraw discretionary early-stop plan. Keep originalmax80/patience18/best-validation checkpoint selection, no configuration or source mutation. Session95689 verifiedlive; {"epoch": 62, "best_epoch": 52, "best_score": 80.18459751154603, "bad_epochs": 10}. This is not qualification for otherfolds; fullOOF>82 remains outstanding.


- Context-calibration80 COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 95689, "exit_code": 0, "best_epoch": 52, "stopped_epoch": 70, "scores": {"POR": 77.21970642705807, "PERM": 89.36964247878716, "SW": 73.54088775958033, "TOTAL": 80.18459751154603}, "delta_original": -0.06629626327320182, "delta_compact": -0.04241762398304161, "best_checkpoint_predictions_reproduced": true, "all_validation_rows_covered_once": true, "seconds": 1651.5405914000003, "peak_gpu_allocated_mib": 783.85595703125, "checkpoint_persisted": false, "decision": "Completed without manual interruption per user request; original patience18 triggered at70. No measured net benefit; no remainingfolds or window/width sweep."}. User-requested completion honored: no manual termination. SingleGPU fold time1651.54s includes preprocessing/training/validation/probes/final export, no separate warmup; monitoring/docs also ran, not a concurrency benchmark. Negative checkpoint not persisted; source/config/compact results retained. Preserve80.3868919215 ensemble; full original OOF>82 unmet.


- Started first-downsample Haar detail trial (session70576), singleGPU; original paired80 baseline, no context-calibration stacking. Fixedfold5/max80/pat18/strict>80.4; initial score pending. FullOOF>82 remains unmet; next inspect actual training then update final evidence. [Experiment](outputs/pure_dl_paired_haar_detail80_sequence/README.md).


- Haar detail trial live update: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; early comparison only; continue unchanged80/18 protocol.", "epoch": 10, "best_epoch": 9, "best_score": 78.1287160339638, "observed_peak_gpu_mib": 775.78466796875, "verified_at": "2026-09-27T17:14:16.699687", "control_same_epoch_best": 78.2303687909218, "matched_delta": -0.10165275695800347, "bad_epochs": 1}. No final gain established; fullOOF>82 remains unmet.


- Haar detail verified live: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; recent successive best scores; continue, no final gain established.", "epoch": 19, "best_epoch": 18, "best_score": 78.93776797376363, "observed_peak_gpu_mib": 775.89501953125, "verified_at": "2026-09-27T17:17:54.000455", "control_same_epoch_best": 78.7476960609833, "matched_delta": 0.19007191278032565, "bad_epochs": 1}. SingleGPU, no restart/config change. Latest printed elapsed403.4s at15 includes preprocessing/probes, not final elapsed; fullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Session70576 verified live; continue original80/18; no final mechanism gain or promotion established.", "epoch": 28, "best_epoch": 28, "best_score": 79.4562704980624, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:21:23.827229", "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.010455988519055381, "bad_epochs": 0}. Training remains singleGPU; fullOOF>82 outstanding.


- Haar detail verified live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; continue unchanged80/18; no final gain or gate qualification.", "epoch": 36, "best_epoch": 36, "best_score": 79.6643116418347, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:24:48.844609", "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.05985402328175837, "bad_epochs": 0}. SoleGPU; last printed754.0s at30 includes preprocessing/probes, not final wall time. FullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; recent improvements warrant continuation; no final gain or promotion.", "epoch": 45, "best_epoch": 44, "best_score": 79.88202580803328, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:28:11.370947", "control_same_epoch_best": 79.89736194351701, "matched_delta": -0.015336135483735802, "bad_epochs": 1}. SingleGPU, unchanged80/18; fullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; continue late-stage comparison; no demonstrated benefit, no promotion.", "epoch": 54, "best_epoch": 52, "best_score": 80.10618716475575, "observed_peak_gpu_mib": 776.27880859375, "verified_at": "2026-09-27T17:31:38.829700", "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.12082797077331975, "bad_epochs": 2}. SoleGPU; last printed elapsed1213.6s at50 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Haar detail CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 70576, "exit_code": 1, "verified_at": "2026-09-27T17:34:51.286891", "epoch": 62, "best_epoch": 52, "best_score": 80.10618716475575, "scores": {"POR": 77.01702176403668, "PERM": 89.28892837567209, "SW": 73.57130201159866, "TOTAL": 80.10618716475575}, "bad_epochs": 10, "delta_original": -0.14470661006347996, "delta_compact": -0.12082797077331975, "observed_peak_gpu_mib": 778.01318359375, "best_checkpoint_restore_verified": false, "partial_result": true, "checkpoint_files": [], "decision": "Manually stopped after62 validations, ten stale epochs and below both controls. No remainingfolds, no detail scale/layer sweep. Preserve partial evidence, do not claim final restore."}. Last printed1417.0s at60 is not final elapsed; singleGPU, preprocessing/probes included, no separate warmup. No negative checkpoint retained. FullOOF>82 unmet; next change mechanism rather than extend this branch.


- Started global head-exchange trial session46372; singleGPU admission verified no other trainers, free10.80GiB. Baseline paired posterior80, fixedfifth/80epochs/pat18, no Haar stacking. Score pending, strict>80.4 promotion, fullOOF>82 unmet. Next monitor actual validation and update atlas.


- Head exchange verified live: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Live session46372 verified; early comparison only; continue unchanged80/18.", "epoch": 12, "best_epoch": 12, "best_score": 78.43859407671914, "verified_at": "2026-09-27T17:42:26.447882", "control_same_epoch_best": 78.48230718472027, "matched_delta": -0.043713108001128376, "bad_epochs": 0, "observed_peak_gpu_mib": 773.955078125}. SingleGPU/no OOM; no final mechanism gain claimed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue80/18, intermediate evidence only.", "epoch": 21, "best_epoch": 20, "best_score": 79.07513986699209, "verified_at": "2026-09-27T17:45:58.589588", "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.016524260061416385, "bad_epochs": 1, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; fullOOF>82 unmet; no final gain claimed.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue original80/18; no final mechanism gain.", "epoch": 29, "best_epoch": 28, "best_score": 79.45907444818177, "verified_at": "2026-09-27T17:49:22.862268", "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.007652038399683647, "bad_epochs": 1, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; last printed634.4s at25 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue original80/18, no final gain established.", "epoch": 38, "best_epoch": 35, "best_score": 79.5323083696878, "verified_at": "2026-09-27T17:52:46.500151", "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.1918572954286617, "bad_epochs": 3, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; last printed876.4s at35 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; recent improvements, continue unchanged80/18; no final gain or promotion.", "epoch": 48, "best_epoch": 44, "best_score": 79.89786675506092, "verified_at": "2026-09-27T17:56:19.499196", "control_same_epoch_best": 79.93028980255141, "matched_delta": -0.032423047490496515, "bad_epochs": 4, "observed_peak_gpu_mib": 774.5}. SoleGPU, no OOM; fullOOF>82 outstanding.


- Head exchange CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 46372, "exit_code": 1, "verified_at": "2026-09-27T17:59:34.324441", "epoch": 54, "best_epoch": 44, "best_score": 79.89786675506092, "scores": {"POR": 76.5404944107509, "PERM": 89.30011532035928, "SW": 73.37336591345687, "TOTAL": 79.89786675506092}, "bad_epochs": 10, "delta_original": -0.35302701975831496, "delta_compact": -0.32914838046815476, "observed_peak_gpu_mib": 776.375, "best_checkpoint_restore_verified": false, "partial_result": true, "checkpoint_files": [], "decision": "Manually stopped after54 validations, ten stale epochs and below both controls. No promotion or head-exchange parameter sweep. Final best restoration not performed."}. SingleGPU; last printed1204.0s at50 is not final elapsed, includes preprocessing/probes, no separate warmup. No checkpoint retained. FullOOF>82 unmet; next change mechanism.


- Started overlap distribution consistency trial session57644, singleGPU admission no other trainers. Original paired model; extra16 windows/step, shift32, symmetric gradients on training-point predictions, no pseudo labels. Fixedfifth/80epochs/pat18/strict>80.4; first score pending. FullOOF>82 unmet.


- User-directed frozen80.39 numeric patch launched session64486: preserve all five experts and learned weights, train zero-init212to32to3 bounded residual MLP on training labels only. Cache construction first; initial prediction must exactly reproduce baseline80.3868919215. Up to2GPU tasks with window-consistency57644; no claim of final gain. Remainingfolds strict>80.4, fullOOF>82 unmet.


- Dual live update: window-consistency {"status": "running", "atlas_current_trial": true, "session_id": 57644, "decision": "Verified live57644; now concurrent with frozen prediction caching64486; continue80/18, no final gain.", "epoch": 26, "best_epoch": 25, "best_score": 79.19331817293609, "verified_at": "2026-09-27T18:14:50.766092", "control_same_epoch_best": 79.14794099275363, "matched_delta": 0.045377180182455845, "bad_epochs": 1, "observed_peak_gpu_mib": 907.31494140625}. Frozen80.39 numeric patch session64486 is still computing train-only four-window prediction cache; no patch score yet. At most2GPU tasks, fullOOF>82 unmet.


- Frozen80.39 numeric patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 64486, "exit_code": 0, "verified_at": "2026-09-27T18:19:26.213685", "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "best_epoch": 0, "stopped_epoch": 18, "gain": 0.0, "best_trained_epoch": 1, "best_trained_score": 80.3769038022867, "last_trained_score": 80.21606532826233, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "cache_bytes": 17365472, "decision": "No benefit: zero patch best. Close smoothL1 bounded numeric correction; preserve compact frozen prediction cache for distinct incremental mechanisms."}. Total524.6965s, includes roughly507.4s through final well cache generation; remainder also includes setup/evaluation, not pure training time. Peak1549.06MiB, concurrent with consistency training. No learned patch retained; fullOOF>82 unmet.


- Started frozen official-metric patch session30303: same numeric MLP, only loss changed from smoothL1 to weighted official clipped error; shared verified17MB cache, no frozen model re-inference. Fixedfifth,80/18, at most2GPU with consistency57644. Await score; fullOOF>82 unmet.


- Frozen metric patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 30303, "exit_code": 0, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_trained_score": 80.38603591067616, "best_trained_epoch": 1, "last_trained_score": 80.36217060265068, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "wall_seconds": 89.4641632999992, "peak_gpu_allocated_mib": 1530.28955078125, "decision": "No net gain: original zero patch best. Official loss more stable than smoothL1 but stop this bounded pointwise numeric correction family; no parameter sweep."}. Total89.46s includes preprocessing/cache checks/training/validation, concurrent with consistency57644; not pure optimization time. FullOOF>82 remains unmet.


- Started frozen80.39 profile patch session75337: change correction information to within-well9/33 raw-row frozen prediction contrasts and signed coefficients. No neighbor labels; same official loss/Q/optimizer/bounds; cache reused. Fixedfifth80/18, max2GPU with consistency57644, no result yet; fullOOF>82 unmet.


- Frozen profile patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 75337, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59455116702611, "PERM": 89.57780436763036, "SW": 73.5910143119091, "TOTAL": 80.38745188794665}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0005599664713287211, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 80.74168310000096, "peak_gpu_allocated_mib": 1563.1240234375, "checkpoint_persisted": true, "decision": "Tiny positive+0.00055997, not substantive and below80.4. Retain small positive patch as control; no window/width sweep."}. Runtime80.74s includes preprocessing/cache checks/training/validation, concurrent with consistency57644; no fullCV qualification. FullOOF>82 unmet.


- Window consistency CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 57644, "exit_code": 1, "epoch": 62, "best_epoch": 52, "best_score": 80.1021567909661, "scores": {"POR": 77.0691196166394, "PERM": 89.2051634411578, "SW": 73.59889629019732, "TOTAL": 80.1021567909661}, "bad_epochs": 10, "delta_original": -0.1487369838531265, "delta_compact": -0.1248583445629663, "best_checkpoint_restore_verified": false, "partial_result": true, "observed_peak_gpu_mib": 907.31494140625, "checkpoint_files": [], "decision": "Stopped after62 validations and10 stale epochs; no net gain, no remainingfolds or coefficient/shift sweep."}. Last printed1729.7s at60 is not final elapsed; includes preprocessing/probes and concurrent frozen cache/patch work during part of run. Additional16windows/step did not improve score. Frozen baseline error analysis now saved: train85.3684 vs validation80.3869, SW both-numeric validation zero-score fraction60.82%; diagnostic only, no validation gradients. Next target a distinct frozen80.39 repair mechanism; fullOOF>82 unmet.


- Started frozen SW action-value patch session16373, soleGPU. Freeze80.39 models/weights, preservePOR/PERM, learn66 expected candidate utilities on training labels only. Original action favored at initialization; zero-change baseline included in selection. Record changedSW fraction. Fixedfifth80/18, strict>80.4 promotion/fullOOF>82 unmet.


- Frozen SW action patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 16373, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 61.80240950000007, "peak_gpu_allocated_mib": 1530.28955078125, "checkpoint_persisted": false, "best_trained_score": 80.36705851680067, "best_trained_epoch": 2, "best_trained_SW": 73.53323004140574, "decision": "No net improvement; zero patch best. Close this SW soft-BCE action-value setup, no candidate-grid/initial-bias sweep."}. SoleGPU total61.80s includes preprocessing/cache checks/training/evaluation; no negative checkpoint retained. changed_SW_fraction logs use exact floating comparison, so include float32 representation changes and are not a semantic correction rate. Restored final prediction exactly equals baseline. FullOOF>82 unmet; no remainingfolds.


- Started private-target frozen profile patch24620, soleGPU. Same zero-initial function and sampling RNG as shared profile, independent hidden gradients for3targets; parameter count increases, not capacity-matched. Same frozen80.39/cache/Q/official loss/80epochs/pat18. Compare shared80.3874519 and baseline80.3868919; strict>80.4/fullOOF>82 unmet.


- Private profile patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 24620, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59433412427276, "PERM": 89.57789619385682, "SW": 73.59094125291199, "TOTAL": 80.38739334365091}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0005014221755885728, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 53.60223920003045, "peak_gpu_allocated_mib": 1563.1240234375, "checkpoint_persisted": true, "delta_shared_profile": -5.85442957401483e-05, "decision": "No incremental mechanism gain over shared profile; tiny baseline gain only. Close target-decoupling direction, no width/LR sweep. Keep small positive control checkpoint."}. SoleGPU total53.60s includes preprocessing/cache checks/training/validation; not a controlled speed comparison to prior concurrent runs. FullOOF>82 unmet; no remainingfolds.


- Started frozen128-block correction72985, soleGPU: same212to32to3 model as pointwise metric patch, shared input summary/correction per raw128row block. No label aggregation; loss remains train-point/Q weighted, zero initial80.3868919. Max80/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen block calibration COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 72985, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.6044252537888, "PERM": 89.57632943119414, "SW": 73.58696168941397, "TOTAL": 80.38847946834947}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0015875468741484156, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 47.49249970004894, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_profile": 0.0010275804028196944, "decision": "Small+0.00158755 baseline gain, best epoch1, later degradation; retain small best checkpoint, no80.4 promotion. No block-size sweep yet."}. SoleGPU total47.49s includes preprocessing/cache checks/training/validation; peak1540.78MiB. FullOOF>82 remains unmet.


- User authorized six penalty strengths0/.001/.003/.01/.03/.1, max2GPU. Reuse0 score80.3884794683 and.01 score80.3885472393. Batch1 launched .001 session38362/.003 session78079; batch2 .03/.1 pending. All same split/seed/model/80epochs/pat18. FullOOF>82 unmet.


- Six-strength anchor sweep COMPLETE: {"screening_fold": 5, "coefficients": [{"coefficient": 0, "TOTAL": 80.38847946834947, "gain_original": 0.0015875468741484156, "delta_no_penalty": 0.0, "best_epoch": 1, "stopped_epoch": 19, "seconds": 47.49249970004894, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": true}, {"coefficient": 0.001, "TOTAL": 80.38848480137887, "gain_original": 0.0015928799035407337, "delta_no_penalty": 5.333029392318167e-06, "best_epoch": 1, "stopped_epoch": 19, "seconds": 64.3387407999835, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.003, "TOTAL": 80.38850238097564, "gain_original": 0.0016104595003127997, "delta_no_penalty": 2.2912626164384164e-05, "best_epoch": 1, "stopped_epoch": 19, "seconds": 64.3800079000066, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.01, "TOTAL": 80.38854723925408, "gain_original": 0.001655317778755716, "delta_no_penalty": 6.777090460730051e-05, "best_epoch": 1, "stopped_epoch": 19, "seconds": 51.5190433000098, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": true}, {"coefficient": 0.03, "TOTAL": 80.38866398629858, "gain_original": 0.0017720648232568692, "delta_no_penalty": 0.00018451794910845365, "best_epoch": 1, "stopped_epoch": 19, "seconds": 50.87739680003142, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.1, "TOTAL": 80.38898528235002, "gain_original": 0.0020933608746958043, "delta_no_penalty": 0.0005058140005473888, "best_epoch": 1, "stopped_epoch": 19, "seconds": 50.15694000001531, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}], "best_coefficient": 0.1, "best_score": 80.38898528235002, "all_passed_restore": true, "all_passed_gate": false, "maximum_gpu_tasks": 2, "new_runs": 4, "reused_runs": 2, "new_process_seconds_sum": 229.75308550003683, "timing_note": "Two pairs concurrent; individual times include preprocessing/training/validation. No matched serial rerun, no speedup claim."}. Preserve small positive checkpoints and compact results; no promotion. Full originalOOF>82 unmet.


- Started well-balanced frozen block patch74759, soleGPU. Train-only inverse-well-count weights applied to data and anchor.1, fixed original validation population. Compare point-weighted80.3889853;80epochs/pat18, no coefficient expansion. FullOOF>82 unmet.


- Well-balanced block patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 74759, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.60473022199899, "PERM": 89.57744955650682, "SW": 73.58785428364669, "TOTAL": 80.38927541065343}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0023834891781007173, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 48.93912729999283, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_point_weighted_control": 0.0002901283034049129, "decision": "Tiny positive+0.00029013 over point-weighted anchor.1; no substantive gain, no80.4 gate. Preserve small best, no well-weight exponent sweep."}. SoleGPU total48.94s includes preprocessing/cache checks/training/evaluation. All predictions restored exactly; fullOOF>82 unmet.


- Started frozen block-variation features52705, soleGPU. Add3spatial std descriptors,96zero-init input weights; original inputs/init/RNG/loss/anchor.1/well-balancing unchanged. Compare80.3892754; no label-derived context.80epochs/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen block variation COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 52705, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.60444285891165, "PERM": 89.57744694086408, "SW": 73.58776066579821, "TOTAL": 80.38915552000529}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0022635985299643835, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 49.80855990003329, "peak_gpu_allocated_mib": 2143.19140625, "checkpoint_persisted": true, "delta_no_variation": -0.00011989064813633377, "decision": "No feature gain versus matched no-variation control; do not add higher moments or scan block lengths. Small positive-vs-original artifact retained as control."}. SoleGPU total49.81s includes preprocessing/cache checks/training/validation; peak2143.19MiB reflects descriptor construction too. Best epoch1/stop19, restore verified. FullOOF>82 unmet, no promotion.


- Started frozen linear block patch52216, soleGPU. Reduce MLP to212to3+tanh, zero initialization; same well-balanced block inputs/Q/official loss/anchor.1/80epochs/pat18. Structure changes RNG consumption, documented not update-identical ablation. Compare80.3892754, fullOOF>82 unmet.


- Frozen linear block patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 52216, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59361541567479, "PERM": 89.57758548553893, "SW": 73.59055551542714, "TOTAL": 80.38693397504056}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 4.205356523812043e-05, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 49.21806949999882, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_mlp_control": -0.002341435612862597, "decision": "No benefit from reduced capacity; essentially unchanged from frozen baseline, worse than MLP. Close capacity-adjustment direction, no width/LR sweep."}. SoleGPU total49.22s includes preprocessing/cache checks/training/evaluation;639parameters. Best1/stop19, restored prediction verified. FullOOF>82 unmet, no remainingfolds.


- Started frozen state-repair patch50014, soleGPU. One alternate per target from input-only33row neighbor predictions/default; trainMSE expected utility difference, inference threshold0 (no validation oracle/tuning). Start exact80.3868919;max80/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen state-repair COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 50014, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 48.7050696000224, "peak_gpu_allocated_mib": 1563.69921875, "checkpoint_persisted": false, "best_trained_score": 80.37943689195787, "best_trained_epoch": 2, "decision": "No net gain from state-repair expected-advantage patch; initial baseline best. No threshold/window scan; no checkpoint retained."}. SoleGPU total48.71s includes preprocessing/cache checks/training/validation. No trained state beats original80.3868919. FullOOF>82 unmet; no promotion.


- Started frozen smooth-score patch57430, soleGPU. Replace only training clipped kernel with Gaussian convolution sigma.25; original official validation remains. Same well-balanced block/Q/anchor.1/80epochs/pat18. Compare80.3892754, no validation gradient. FullOOF>82 unmet.


- Frozen smooth-score patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 57430, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.6042669423747, "PERM": 89.57816415742207, "SW": 73.58768332216735, "TOTAL": 80.3893267005687}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.002434779093377415, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 66.73988799995277, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_hard_kernel_control": 5.1289915276697684e-05, "decision": "Only+0.00005129 over hard-kernel control, no substantive gain; best1/stop19. No sigma sweep, retain small positive checkpoint."}. SoleGPU total66.74s includes preprocessing/cache checks/training/validation; original official scores, not smoothed proxy scores. FullOOF>82 unmet, no promotion.


Frozen latent patch COMPLETE: fifth TOTAL80.3868919215, best epoch0, natural stop18; best trained80.3759376707@1. Best prediction restoration verified. No gain and no checkpoint retained. Single GPU total66.724s including preprocessing/extraction/training/validation; extraction5.846s, peak3306.041MiB. All723556 valid rows covered once; normalization uses578891 training rows only. No remaining-fold promotion. FullOOF>82 unmet.

- Started rank8 cross-feature calibration, session77606, sole GPU. New720-parameter context-controlled multiplicative residual in calibrated member; zero output preserves initial baseline. Fixed fifth/max80/pat18/bs64, train-only preprocessing. No result yet; strict>80.4 before promotion, fullOOF>82 remains unmet. See outputs/pure_dl_paired_lowrank_calibration80_sequence/README.md.

- Live snapshot: rank8 calibration session77606 confirmed training, epoch4 best fifth76.93051414. Early result only, no promotion; no OOM, latest reported PyTorch peak776.264MiB.

- Running rank8 calibration: last epoch20, best78.91579401@20; last same-epoch delta-0.17587012. Continue because best is still improving; no final ablation gain or promotion claimed. Session77606 polled live.

- Prepared next mechanism, not launched: weak train-only cross-well hidden feature-statistics mixing (MixStyle-inspired). See outputs/pure_dl_well_style_plan/README.md for evidence, fixed starting design, padding/RNG/domain safeguards and physical-information risk. Current rank8 calibration session77606 must finish naturally under80/pat18; no manual early stop per latest user request.

- Started weak well-style training session69486 concurrently with lowrank77606 (max2). Fixed fifth,80/pat18; no extra parameters. Cross-well donor IDs from training anchor well starts, masked statistics, dedicated RNG, disabled in eval/probe. Initial score pending. Preserve lowrank natural completion. FullOOF>82 unmet.

- Live paired_lowrank_calibration80: epoch49, best79.80864783@44; session77606 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Live paired_weak_well_style80: epoch5, best77.53947410@5; session69486 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Rank8 calibration COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.72788441696548, "PERM": 89.24684557961116, "SW": 73.38376598362012, "TOTAL": 79.93907937222059}, "seconds": 1198.6207101999898, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 782.7607421875, "status": "complete", "atlas_current_trial": false, "session_id": 77606, "exit_code": 0, "screening_fold": 5, "delta_compact_control": -0.28793576330848225, "delta_original": -0.31181440259864246, "decision": "No gain. Natural patience18 stop at70; no rank/capacity sweep, no remaining folds."}. Total1198.62s includes preprocessing/training/probe/validation; partially concurrent with weak-style, no matched serial speedup claim. No checkpoint retained.

## Historical family clarification
A previous conditioned rank8 numeric-mixing trial already completed at80.1954269324 (outputs/pure_dl_paired_conditioned_mix80_sequence). That version applies mixing before the original affine calibration and has biases; this later720-parameter version adds a bounded correction after affine calibration. They are related variants, not separate unexplored mechanism families. Both below compact80.2270151; close this family, no further rank/placement sweeps. Historical PCGrad80.1582856 also failed compact control; do not relaunch gradient projection as a new idea.

- Started80.39 attribution session51546: fixed5/checkpoint hashes, single/equal/learned/leave-one-out and single-vs-four-window/coarse-vs-fine decoding; diagnostic only, no promotion. Runs alongside weak-style69486, max2GPU tasks. Results pending; models/loss attribution requires matched training controls, not historical score subtraction.

- Weak style COMPLETE: {"fold": 5, "best_epoch": 48, "stopped_epoch": 66, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 75.7366886208511, "PERM": 89.13228344192615, "SW": 73.48030727511235, "TOTAL": 79.6354133372188}, "seconds": 1278.4947431000182, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 777.9208984375, "status": "complete", "atlas_current_trial": false, "session_id": 69486, "exit_code": 0, "delta_compact": -0.591601798310279, "decision": "No gain; natural stop66, best48 restored. No augmentation strength/layer sweep."}. Partial concurrency with lowrank and source attribution; no speedup claim. User direction now prioritizes80.39 attribution, not new mechanism search.

-80.39 attribution inference completed26 arms (214.40s/1461.84MiB): strongest single80.331454/equal5 80.386864/learned5 80.386898; see outputs/pure_dl_8039_source_ablation/ANALYSIS.md. Reference score difference6.03e-6, not elementwise identical. Diagnostic dropseed07 80.403220 is NOT auto-promoted. Training ablations Q=1(session82504) and POR Gaussian(session69566) now running, max2; no results yet.

- User adopts four-source80.403220094 baseline (indices0,2,3,4). Six equal two-source subset ablations prepared and queued in observable session28718; waits for GPU capacity while Q=1 session82504 and GaussianPOR69566 run. No backbone retraining for pair study. Preserve original sources/deliveries; diagnostic selection history explicit. See outputs/pure_dl_8040_pair_ablation/README.md.

- Attribution monitoring: [{"status": "running", "atlas_current_trial": true, "session_id": 82504, "screening_fold": 5, "last_epoch": 26, "best_score": 79.15393682729156, "best_epoch": 23, "ablation": "no_quality", "note": "Partial result; no final contribution claim."}, {"status": "running", "atlas_current_trial": true, "session_id": 69566, "screening_fold": 5, "last_epoch": 22, "best_score": 79.07819692255848, "best_epoch": 20, "ablation": "gaussian_por", "note": "Partial result; no final contribution claim."}]. Pair inference28718 verified CPU-waiting while these2GPU tasks run. Historical paired module config/shared-hash audit completed without shared hash changes; source analysis updated, one-seed evidence only.

- Expanded80.4032 subset attribution per user:6pairs+4leave-one-out triples+baseline=11arms. Replaced only CPU-waiting inference28718 with94542, confirmed waiting; training82504/69566 uninterrupted. Same source inference will serve all11arms.

- Attribution runs observed live: [{"ablation": "no_quality", "epoch": 42, "best": 79.7362617128407}, {"ablation": "gaussian_por", "epoch": 40, "best": 79.57860036215014}]. Subset11arms session94542 remains queued; no thirdGPU task. Added config-based source/subset parameter counts to outputs/pure_dl_8040_pair_ablation/model_costs.json (baseline4.646M/7backbones, pairs1.987-2.658M/3-4backbones), not a measured speedup.

- Training attribution live snapshot [{"mode": "no_quality", "epoch": 56, "best": 80.09251227357417}, {"mode": "gaussian_por", "epoch": 55, "best": 80.04144251747569}]. Both still running; do not finalize Q/distribution contributions yet. Added historical attention audit (+0.2158936568 at matched single-backbone75, not current ensemble causal delta) to source ANALYSIS.md. Removed2completed-negative point-diagnostic caches44,965,265bytes with manifest outputs/cleanup_negative_calibration_style_attribution.json; code/config/validation curves/scores and all active files retained. Subset11 queue94542 unchanged.

- Training ablations COMPLETE: [{"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.94295729632633, "PERM": 89.27312796013906, "SW": 73.61151513893603, "TOTAL": 80.09251227357417}, "seconds": 1529.0026964999852, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 82504, "exit_code": 0, "delta_control": -0.1345028619548998, "ablation": "no_quality"}, {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.71596748229165, "PERM": 89.4341072589125, "SW": 73.49918494905378, "TOTAL": 80.04144251747569}, "seconds": 1362.7202495000092, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 69566, "exit_code": 0, "delta_control": -0.18557261805338499, "ablation": "gaussian_por"}]. Q and t3 retained on this single-seed evidence, no additive contribution claim. Subset11 inference94542 now admitted, soleGPU. FullOOF>82 unmet.

- All requested attribution completed:11 subset arms, baseline80.403220094 exactly reproduced. Best pairAB80.3801885; best tripleABD80.4091519 (+0.0059318, parameters3.320M versus4.646M, exploratory selection not robust improvement claim). Q-off80.0925123 and GaussianPOR80.0414425 versus matched dual80.2270151, bothbest52/stop70 restore verified. Reports outputs/pure_dl_8040_pair_ablation/SUMMARY.md and outputs/pure_dl_8039_source_ablation/ANALYSIS.md. All3sessions terminal, no new training launched; fullOOF>82 still unmet.

- Started targeted attribution follow-ups: main-only auxiliary-loss removal62826 (same original dual80/Q/t3, max80/pat18) and frozen internal expert intervention60959 (Aseed06/Bseed08 each full206/calibrated68/equal/routed). Max2GPU; not a new unstructured mechanism sweep. Four-source80.4032 baseline unchanged, results pending, fullOOF>82 unmet.

- Internal experts COMPLETE8arms, exact A reference reproduced. [{"source": "A", "equal_gain_over_best_branch": 0.3514727762118639, "router_gain_over_equal": 0.09321302034933865, "routed_gain_over_best_branch": 0.44468579656120255}, {"source": "B", "equal_gain_over_best_branch": 0.31227311776699196, "router_gain_over_equal": 0.009940617467393054, "routed_gain_over_best_branch": 0.322213735234385}]. Full206 alone stronger total; calibrated68 complements SW. This is frozen intervention, not retrained causal removal. Source report retained. Main-only62826 continues, no new GPU task.

- Main-only attribution62826 confirmed live, epoch13 best78.56925593@13, now soleGPU. Internal-expert inference60959 completed153.71s/1444.27MiB, exact reference; report ANALYSIS.md updated with branch complement and routing gains. FullOOF>82 unmet.

- Added and visually inspected attribution_summary.png/.svg (3separate comparison panels, completed results only), linked from SUMMARY.md and ANALYSIS.md. Main-only62826 continues as soleGPU, epoch25 best79.28455813@25; no final auxiliary-loss contribution claim.

- Started matched no-calibration attribution66172 alongside main-only62826, max2GPU. Original posterior network, calibration zero-output initialization frozen; 68view/Q/t3/auxiliary/router retained. Fills interaction gap beyond older no-router75/80 comparisons; no rank/width sweep. Result pending, original80.4032 four-source baseline unchanged and fullOOF>82 unmet.

- Live final attribution pair: [{"mode": "main_only", "epoch": 40, "best": 79.69091378066616}, {"mode": "no_calibration", "epoch": 2, "best": 76.68573766110542}]. Both actual train/validation/probe cycles confirmed; max2GPU and no OOM observed. Summary renderer prepared to include completed optional auxiliary/calibration ablations only, no provisional results in final comparison charts.

- Verified live attribution wait: [{"mode": "main_only", "epoch": 54, "best_epoch": 52, "best": 79.87490529017835}, {"mode": "no_calibration", "epoch": 14, "best_epoch": 12, "best": 78.49377082386238}]. Both valid tool handles active, no restart or premature final contribution claim. Main-only refreshed best52; calibration still early. No new experiments added.

- Main-only attribution COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.68933694757615, "PERM": 89.15215965438021, "SW": 73.32813807677836, "TOTAL": 79.87490529017835}, "seconds": 1235.02869440004, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.43896484375, "status": "complete", "atlas_current_trial": false, "session_id": 62826, "exit_code": 0, "delta_control": -0.352109845350725, "decision": "All3channels worse. Retain auxiliary supervision in current setup; no auxiliary-weight sweep."}. Partial concurrency; time includes preprocessing/training/probes/validation. Input-calibration66172 remains soleGPU task; fullOOF>82 unmet.

- Final main-only result integrated into SUMMARY/ANALYSIS/summary.json and regenerated chart; chart visually checked with extended delta axis (includes-0.3521). No-calibration66172 still live, epoch34 best79.63841458@28, soleGPU.


## 2026-09-27 ABD baseline and ensemble-aware module attribution
- Goal: original grouped full OOF>82 remains unmet; user adopts fixed-fifth ABD80.40915186344968 as development baseline (not independent test).
- Implemented train_abd_fused_training80.py: freeze B/D, warm-start A, train final equal-mixture likelihood with frozen train-only Q, bs64/64steps/max80/pat18, lr3e-5 cosine1e-5. Checkpoint/early stop by fused TOTAL, solo diagnostic. Session90559 started; initial readback80.4091386805 (delta -0.00001318 from historical, numerical rebatching; not elementwise exact), gradient check A-only passed. Improvement pending.
- User requests deeper attribution: evaluate_abd_backbone_modules.py started session30376, baseline+12 one-factor inference interventions in A's two members with B/D fixed. Local/global/hierarchy/pointMLP/bottleneck/skips; pending results and selected controlled retraining. Inference dependency is not causal retraining gain.
- No-calibration control completed80.0411598317@52/stop70, -0.1858553038 versus matched80.2270151; restored/coverage verified. Old policy omitted checkpoint so no fusion conclusion. New candidates retain best weights for fusion assessment.
- Updated root/repo/experiment AGENTS, README, individual exploration READMEs and atlas. Next: complete module table and compare solo/fused channel/well changes; select a focused training ablation. No remaining-fold or delivery changes yet.


## 2026-09-27 Completed ABD internal-module attribution
- Completed baseline+12 arms: 10 genuine A-module interventions and2 numerical controls (point is already EmptyPoint; independent MLP absent). B/D unchanged. Source weights preserved,144665 rows each covered4times. 419.719s total including data/cache,1705.00MiB peak, concurrent with incremental trainer.
- Full206 A branch removal deltas in ABD: local -0.140824; global +0.025934 (80.43507290); whole hierarchy -0.008891; bottleneck -0.066270; skips -0.047187. Calibrated68: local -0.065327; global -0.131623; hierarchy -0.147241; bottleneck -0.056000; skips -0.133221. All genuine removals hurt A alone. Global-full candidate gains mainly SW+0.08924,9/16 wells improve. Inference dependency only, not training causal gain.
- Original fused-loss incremental trainer finished18 epochs without gain. Bestepoch0=80.40913868, all trained epochs lower (max80.285611). Strict prediction replay assertion failed: CPU readback confirms every saved A tensor equals original A and saved predictions recompute exact saved score. Report explicitly marks exact prediction reproduction false; no false success claim. No need to retrain this negative control.
- Started single-module adapted candidate session18573: train_abd_no_global_full80.py, full206 global attention structurally removed; B/D frozen, same incremental budget/loss. Deterministic operations added to avoid repeat-index accumulation jitter, documented difference. Initial match and training results pending. User-adopted baseline remains80.40915186, not silently changed to selected candidate.
- Next: observe candidate best fused checkpoint and restoration. Random-initialization matched module retraining remains unperformed; fullOOF>82 unmet. Further modules worth later attribution: anti-aliasing downsampling and depth-only normalization; no new sweeps launched.


- Module adaptation launch18573 failed BEFORE any optimizer update: CUDA linear upsampling backward lacks deterministic implementation. Preserved source/config/epoch0 and failure record. Replaced with train_abd_no_global_full_evaldet80.py in NEW outputs/pure_dl_abd_no_global_full_evaldet80_fold5, session51252. Only validation/frozen cache deterministic; training ordinary CUDA. No second live GPU job. Source architecture/budget/loss unchanged. User-adopted ABD baseline unchanged. Compact module table published locally in docs/deep_learning/ABD_MODULE_ATTRIBUTION.md and results/deep_learning/abd_module_ablation_summary.json; no Git push requested.

- Replacement51252 verified actual training: deterministic initial fused80.4350829326; A-only gradient check passed; epoch1 fused80.343809445, best remains epoch0. Training continues max80/pat18; no trained gain claimed.


## 2026-09-27 Random-initialization module attribution
- Previous goal turn classified progress: completed13-arm module evidence, candidate deletion showed solo/fusion disagreement. Current goal remains full original OOF>82, not yet achieved.
- Verified51252 live; no-global incremental training still bestepoch0. Do not replace baseline or claim training gain.
- Implemented FrozenABDValidation and train_abd_module_retrain80.py. Two predefined matched arms: original A control vs only full206 global attention removed AFTER identical initialization. Restore original per-member/router NLL, annealed auxiliaries and full-training-point dynamicQ, not the failed incremental fused-loss objective. B/D frozen and cached; each epoch selects best A by fixed equal ABD full-distribution official score. Same seed/max80/pat18/min10/bs64/64steps/lr.0006cosine. Validation deterministic, ordinary CUDA training.
- Control started64442 while51252 active, total2GPU. Removal arm prepared, not yet started; waits for51252 natural completion. No redundant preflight or independent review, only syntax and actual runtime invariants. Actual teacher preprocessing must match training split. Source hashes frozen at launch.
- Next: launch removal once slot free; compare fused and solo curves and final restored results. Preserve both best candidates until fused assessment. No fold continuation or release replacement until clear gain.


- Incremental candidate51252 COMPLETE exit0: best80.4350829326@0, stop18. Exact restored predictions, frozenB/D unchanged and4-window coverage verified. Delta originalABD+0.0259311 comes from removing module, zero extra training gain.572.4575s including setup/cache/training/restore,1798.65MiB peak. Preserve candidate; no incremental LR sweep.
- Paired from-scratch no-global-full launched60531 after51252 terminal; control64442 already training. Two GPU tasks only. Original A losses/aux/dynamicQ retained, B/D frozen, checkpoint selected by ABD. Both80/pat18/min10 and originalseed. Final comparison pending; fullOOF>82 still unmet.

- Live paired-training snapshot (not final/restored): [{"arm": "control", "last_epoch": 8, "best_epoch": 6, "best_fused": 80.49230853582785, "solo_at_fused_best": 77.69492368909336, "delta_baseline": 0.08315667237816626}, {"arm": "no_global_full", "last_epoch": 3, "best_epoch": 3, "best_fused": 80.39633467723046, "solo_at_fused_best": 76.81323320745568, "delta_baseline": -0.012817186219223231}]. Both processes confirmed live via tool sessions64442/60531. Keep fullOOF>82 active; no early promotion before best-checkpoint restoration.


## ABD full-CV continuation prepared, not launched
- Live paired screen still running64442/60531. Original control observed80.49230854@6; no-global80.46881577@6. Not final or promoted yet. Last goal turn=progress, fullOOF>82 not proved.
- Implemented fold-aware frozen validation, single-fold B and A continuation entrypoints, promotion verifier, two-slot dependency queue and original-label fullOOF verifier. Existing D fivefold models reused only on their own fold; B must train each remainingfold before its A. Fifth reused unchanged after exact restored checkpoint/population proof. Per-fold delta is not calculated against fifth baseline.
- New outputs/pure_dl_abd_fusion_selected_fullcv README defines source roles,80/18 budget, score scope, admission, no-overwrite and failure behavior. No GPU continuation launch before completed fifth gate. New files syntax-checked only, actual other-fold runtime pending. Historical active source hashes untouched.
- Next: finish paired fifth, choose one predeclared architecture by fused score, verify it, then fill4folds with max2GPU and score full723556-row OOF. No need to ask permission: continuation is already in user goal, conditional on actual gate.


## Latest user steering: defer fullCV; compare joint training versus B-only
- User explicitly says no remaining folds for now. Promotion and queue NOT launched; no promotion.json. Root/repo/experiment AGENTS updated. FullOOF>82 remains active/unproved; this is not a pause of the overall goal.
- Both scratch attribution arms COMPLETE exit0, bestepoch6/stop24, exact restored predictions and coverage verified. Original80.4923085358 vs no-global80.4688157742; keep original. Matching delta -0.0234927616 for deletion. Rendered and visually checked results/deep_learning/abd_retrain_attribution/checkpoint_selection.png; A solo improves later while fused declines.
- Latest user asks jointly updating3models versus freeze bestA6/D and trainB. Implemented one shared trainer train_abd_joint_or_b80.py with mode joint/b_only. Both warm-start the identical80.49231 composition, same batch RNG, fused mixture NLL, 3e-5 cosine1e-5,80/pat18/min10/bs64, per-updated-model clip2. Uses frozen Q active during A best6 (probe epoch4, all1), no auxiliary restart. Explicitly incremental, not random-initialization joint training.
- Launched joint21517 and B-only38600, max2GPU. Initial cache/gradient scope/results pending. Same ordered A+(B+D) evaluation, deterministic validation. Best epoch0 remains candidate; only one ensemble_best.pt per route. B-only must verify A/D unchanged at end.
- Next: compare actual fused score gains and gradients/restore; no fullCV launch even if positive until user resumes that step. Docs/EXPLORATIONS/atlas updated, no remote push.

- New matched runs verified epoch0 EXACTLY reproduces80.49230853582785, zero differing prediction elements in both arms. Gradient masks confirmed: joint[0,1,2], B-only[1]. First epoch joint80.28915356, B-only80.36295906; both currently retain epoch0. Early observations only, no training gain yet; sessions21517/38600 remain live. First-epoch wall90.12s versus44.06s under partial concurrent setup, not an equal-workload speed benchmark.


## CPU gain decomposition while two live GPU routes continue
- Previous goal turn=progress (new matched runs launched, exact epoch0/gradient masks verified). Current sessions21517/38600 and process IDs85532/45368 confirmed live; no restart on observation timeout. No remaining-fold queue launched.
- Added analyze_abd_checkpoint_gain.py and results/deep_learning/abd_retrain_attribution/GAIN.md/json. Original ABD80.40915186 ->80.49230854: POR+0.113417, PERM+0.214180, SW-0.073804. Default contributions to TOTAL +0.150041 and nondefault -0.066885; PERM nondefault improves but POR/SW nondefault worsen. All contribution decompositions exactly sum to official delta.
- Only7/16 wells improve; well index69 contributes+0.136418 to total, greater than net+0.083157. Excluding this largest positive well diagnostically leaves -0.056845 on15wells. No per-well prediction selection or metric changes. This weakens a broad-generalization claim and supports withholding fullCV as user requested.
- Prepared distinct next route train_abd_b_scratch80.py / abd_frozen_ad_validation.py: B random seed08 initialization, original losses/aux/dynamicQ, frozen bestA6/D, fused selection. NOT launched; current matched incremental pair has priority. Not a one-factor comparison to incremental because initialization/LR/aux/Q differ; explicitly documented.
- Joint observed best trained80.377424@3 (last4); B-only best trained80.484791@5 (last8); both best saved remains epoch0 80.49230854. Continue within budget, no new score claimed.


- B-only incremental38600 COMPLETE exit0: best80.4923085358@0, stop18; trained best80.48479059@5. Exact restored prediction, A/D unchanged and coverage checks passed.923.261s including setup/cache,1806.81MiB peak. No incremental gain; do not call initialization score a training improvement.
- Deleted its duplicate epoch0 ensemble_best.pt only after all3 saved state dicts matched preserved A6/B/D exactly, freed20,066,351bytes. Manifest outputs/cleanup_abd_b_only_epoch0.json. Sources/milestones/positive controls/active outputs intact.
- Started B-from-scratch62769 after38600 terminal; total2GPU with joint21517. FrozenAD_READY confirmed; firstepoch fused80.06379145, Bsolo75.34658365. New seed08 initialization, original losses/aux/dynamicQ; not comparable to incremental as a one-factor intervention. No fullCV.
- Joint21517 remains live; last12, trained best80.39709517@5, saved best still epoch0. Respect natural pat18; source untouched. Completed CPU gain analysis confirms concentration, so no broad-generalization claim.


## Channel learning-speed hypothesis
- Joint incremental training completed18 epochs, best remains epoch0=80.49230854; no gain, exact restored predictions verified. B from scratch continues; all-three-from-scratch launch deferred in favor of user-requested channel conflict diagnostic. No remaining folds.
- New benchmark_channel_gradients.py uses512 fixed train-well windows, Q=1, saved A6/A52/B43/D52, per-channel main/auxiliary shared gradients; no optimizer updates. Session16001 started, result pending. Distinguish norm imbalance from negative cosine and from ensemble complementarity.

- Channel-gradient diagnostic16001 COMPLETE exit0:87.739s,660.494MiB,512 train-only windows, four checkpoints unchanged. A52 mean norms POR/PERM/SW4.920/3.063/1.619. B PERM-SW negative cosine7/16 batches, but no shared equal-gradient opposing any task in sampled batches. Q=1/raw gradients only; no causal or Adam-update claim.
- Same A24-epoch history: epoch6 achieves51.1%/79.0%/57.0% of observed POR/PERM/SW epoch1-to-best validation gain; first90% at17/12/20. Evidence favors faster early PERM validation learning; per-channel training-loss convergence not recorded. Report and README updated. Next: instrument channel losses/gradients, then isolated mild shared-gradient balancing with fixed B/D. No new training launched this batch.


## Correct training fusion semantics
- User clarifies main TRAINING objective must fuse A/B/D with other two frozen. Existing B scratch solo-loss run retained unchanged as control; new train_abd_b_fused_scratch80.py launched separately. A6/D52 frozen eval/no_grad, B random seed08, final mixture likelihood differentiates into B backbones+router; original B auxiliary/Q/80-epoch budget retained. Validation uses final fused official TOTAL.
- Added channel BCE/NLL per epoch and first-batch shared-gradient diagnostics; no adaptive channel weights yet. Prepared all-three-joint-scratch remains unlaunched. No remaining folds. First real backward/scores pending.

- Corrected fused-loss B session26379 reached real first backward: B experts/router nonzero gradients, A/D no gradients and tensor-equal to source. Validation pending; two-job GPU snapshot9444/12282MiB withoutOOM.


## Planned frozen backbone with independent target heads
- User proposes future POR/PERM/SW heads with independent architectures, optimizers, stopping and per-channel checkpoint selection on one frozen backbone. Plan recorded at outputs/pure_dl_frozen_backbone_channel_heads/README.md and atlas; not implemented or launched. Current fused B training unchanged.
- First isolate freezing/per-channel checkpoint selection with same head capacity; heterogeneous heads later. In ABD use final fused per-channel likelihood and scores with companions frozen. Reassemble fixed three heads and verify full population prediction; no per-row/well oracle. Existing router descriptors span all channels: frozen router parameters alone do not remove cross-head dependency; use frozen original routes or independent routes.

- Frozen-head route now implemented: frozen_channel_heads.py + train_abd_channel_heads80.py, syntax parsed; NOT launched (two liveGPU jobs). Freeze original B43 feature extractor and original routing outputs, train19350 head params, per-channel fused-score best checkpoint/earlystop, original A6/D52 frozen. Actual initialization/reassembly checks pending runtime.
- Live status refresh: solo-loss B at54, best80.484205@47 (-0.008104 vs80.492309); final-fused B at8, best79.902002@7 (-0.590306). Both processes live, no final results.


## CPU handoff and comparison snapshot
- Prior turn made progress: implemented frozen heads and real live checks. Added CPU-only run_channel_heads_after_control.py; session47285/queuePID85556 waits for controlPID47108 termination + successful report, then verifies<2GPU tasks and>=3GiB before launching heads once. Does not retry failures or run fullCV. If user stops training, also stop this waiting process.
- Added summarize_abd_b_training_routes.py and results/deep_learning/abd_b_training_routes/{snapshot.json,README.md,fused_channel_curves.png}; image visually inspected. Same epoch12 final-fused79.657911 versus solo-loss-control80.080111, delta-0.422199. Early evidence only, not final.
- Removed duplicate completed joint epoch0 ensemble_best.pt after all3states matched preserved A6/B43/D52;20,066,351bytes freed, manifest outputs/cleanup_abd_joint_epoch0.json. Active/milestone models untouched.


## User stopped and replaced routes; A6 backbone correction
- Stopped queue85556, fused-scratchB66088 and newly admitted old head trainer89900 at user request; old solo-lossB47108 had already completed65epochs with verified80.484205@47. Manifest outputs/stop_old_b_routes_20260927.json.
- User requires own-model training loss and validation-only fusion, and specifically A6 from80.4923 as frozen head backbone. Implemented separate train_a6_channel_heads_own80.py + train_b_own_finetune80.py; historical sources unchanged. A6heads use B43/D52 validation, B43warmstart uses A6/D52 validation. Both80/18,LR3e-5,normalGPU; no remaining folds.
- Both launched; first runtime/score checks pending. Head route trains19350params, independent channel stopping/selection and original frozen routing. Source/checkpoint provenance documented in separate README files.


## Unified frozen feature backbone route
- User clarified whole ABD ensemble as feature backbone, no old output averaging. Stopped B-own fine-tune PID50824 only; A6 head PID20076 preserved. Implemented abd_feature_backbone_heads.py + train_abd_feature_heads80.py:5frozen128dim branches->640features->3independent640/128/25 heads (255819 trainable). Original posterior routers not executed; original prediction tensors discarded.
- New run session10715 launched as secondGPU job, normalCUDA,80/18,bs64,LR.0006. New head initialization uses training-only priors; do not expect baseline80.4923 at epoch0. A6 control initial baseline exactly reproduced, awaiting/latest live scores tracked separately. Final head assembly and frozen source checks required; no fullCV.

- Structural feature-head run10715 passed real first backward: only255819 new-head params trainable; all5feature branches frozen, legacy voting absent. Initial random-head TOTAL68.41007972 is epoch0, not a trained result or expected80.4923 reproduction. GPU snapshot8227/12282MiB. A6-head control continues (initial exact80.49230854; per-channel best assembled candidate80.501552 byepoch11, final restore pending).


## Completed A6 target-head selection and ongoing composite backbone
- A6-head session92282 COMPLETE exit0: fixedfifth80.50155201341903 (+0.009243477591184 vs80.49230853582785). Best epochs POR/PERM/SW4/0/11, stopped22/18/29. Final assembled predictions exactly reproduced, all frozen sources unchanged,4window coverage verified.874.133s/1730.905MiB, partially concurrent. Keep positive heads_best.pt and A6/B43/D52 sources. This small development gain is not fullOOF or independent superiority; remaining folds still user-deferred.
- Whole-ABD feature-backbone session10715 confirmed live; epoch7 own-output TOTAL79.2182588, per-channel best candidate79.2587058. No legacy output voting; candidate not final reconstructed score. Continue monitoring only, do not launch further routes while user asks to watch experiments.


## Channel-wise optimization objective and first mechanism
- User explicitly adds separate optimization of POR/PERM/SW heads/methods. New statistics-readout route train_abd_feature_stats_heads80.py launched session74235 alongside existing10715; max2GPU. Frozen5branches, same base initialization/head/loss/LR/budget; per-channel zero10->128 adapter restores branch means/logstd discarded by normalization (+3840 params, total259659). Own losses and independent checkpoint selection preserved; first runtime evidence pending.
- Added fold5_screening_805.py to new trainer source hashes/config; strict>80.5, clear gain required. Existing source/config/delivery immutable, no extra folds. Compare per-channel outcomes after completion, do not require all3channels to improve simultaneously.


## Explicit overnight stop
User requests stop ALL training and continue tomorrow. All project trainers and waiting launchers terminated, process list verified empty; outputs/overnight_stop.json and per-run paused_snapshot.json record state. Best saved heads and source models retained, no final reconstruction claimed. Do not auto-resume or launch on persistent goal continuation; wait for explicit user resumption. Delivery ZIPs unchanged.


## 2026-09-28T09:16:47+08:00 双专家独立头对照核对
pure_dl_dual8222_heads_pointwise80_fold5: complete, epoch19, TOTAL=80.25333950575663, scope=完成：固定三个最佳头共同恢复; process_alive=False.
pure_dl_dual8222_heads_local580_fold5: complete, epoch19, TOTAL=80.25447676690142, scope=完成：固定三个最佳头共同恢复; process_alive=False.
仅本批两结构；旧ABD保持暂停、未补折、原交付不改。源80.221861预测已实际回放；各目标收益与最终恢复分开。完整OOF>82未完成。新记录见results/deep_learning/dual8222_head_structure_20260928/snapshot.json。


## 2026-09-28T09:34:18+08:00 双专家继续优化结果核对
|方法|状态|TOTAL|最佳轮次|
|---|---|---:|---|
|linear|最终固定三头共同恢复已验证|80.253777203|[1, 0, 0]|
|cross80|最终固定三头共同恢复已验证|80.255112897|[1, 0, 0]|
|gate_only|运行快照候选，非最终恢复|80.221861083|[0, 0, 0]|
|density_only|运行快照候选，非最终恢复|80.256678729|[1, 0, 0]|

已完成：新增线性归因/联合内部专家读出/因素分离实现与实际运行记录。
待完成：所有候选的最终恢复、逐通道配方固定后的共同推理；不存在完整OOF增益证明。
目标：当前双专家可复现优化，第五折>80.5且明确收益；完整OOF>82仍未达成。
达成后：固定方案后评估泛化，当前不自动补折、不改旧交付。
图谱在既有curation生成后运行本脚本再build_atlas；不修改此前被保护的curate.py。


## 2026-09-28T09:40:23+08:00 双专家继续优化结果核对
|方法|状态|TOTAL|最佳轮次|
|---|---|---:|---|
|linear|最终固定三头共同恢复已验证|80.253777203|[1, 0, 0]|
|cross80|最终固定三头共同恢复已验证|80.255112897|[1, 0, 0]|
|gate_only|最终固定三头共同恢复已验证|80.222820979|[0, 34, 0]|
|density_only|最终固定三头共同恢复已验证|80.256678729|[1, 0, 0]|

已完成：新增线性归因/联合内部专家读出/因素分离实现与实际运行记录。
待完成：所有候选的最终恢复、逐通道配方固定后的共同推理；不存在完整OOF增益证明。
目标：当前双专家可复现优化，第五折>80.5且明确收益；完整OOF>82仍未达成。
达成后：固定方案后评估泛化，当前不自动补折、不改旧交付。
图谱在既有curation生成后运行本脚本再build_atlas；不修改此前被保护的curate.py。


## 2026-09-28T09:40:23+08:00 四项优化与固定通道组合已完成
本轮linear/cross80/gate_only/density_only全部自然结束并共同恢复。最终固定组合TOTAL=80.257638624609，相对原80.221861=+0.035777541569；配方=[{'target': 'POR', 'arm': 'density_only', 'score': 77.39392666356925, 'epoch': 1}, {'target': 'PERM', 'arm': 'gate_only', 'score': 89.3475027344081, 'epoch': 34}, {'target': 'SW', 'arm': 'source', 'score': 73.62238476712884, 'epoch': 52}]。没有达到独立泛化目标，不启动补折或改交付。更细分布读出优化未启动。总结见results/deep_learning/dual8222_round2_20260928/FINAL.md。


## 2026-09-28T09:49:26+08:00 POR子空间结果核对
[('mean_only', 'running', 4), ('mean_scale', 'running', 4), ('density_all', 'planned', 0)]
已完成最佳={'mode': 'previous_fixed_recipe', 'TOTAL': 80.25763862460869, 'delta': 0.0}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T09:52:33+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 0)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T09:54:12+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 13)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T09:56:37+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'complete', 19)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=True。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T10:08:19+08:00 位置读出v5
已完成：代码、明确启动、实际状态记录；全部完成=False。
当前最佳={'mode': 'previous', 'TOTAL': 80.26873450885158, 'delta': 0.0}。
待办：未完成实验的最终恢复、获益读出的保存重载验证；不继续增加本批候选。
目标：第五折>80.5且明确收益，完整OOF>82未完成。达成后固定方案做泛化确认，当前无补折授权。


## 2026-09-28T10:11:25+08:00 位置读出v5
已完成：代码、明确启动、实际状态记录；全部完成=False。
当前最佳={'mode': 'point32', 'TOTAL': 80.27413533779057, 'delta': 0.005400828938988411}。
待办：未完成实验的最终恢复、获益读出的保存重载验证；不继续增加本批候选。
目标：第五折>80.5且明确收益，完整OOF>82未完成。达成后固定方案做泛化确认，当前无补折授权。


## 2026-09-28T10:15:48+08:00 位置读出v5
已完成：代码、明确启动、实际状态记录；全部完成=True，已验证保存重载=True。
当前最佳={'mode': 'point32', 'TOTAL': 80.27413533779057, 'delta': 0.005400828938988411}。
剩余：独立泛化及中断续跑等价性尚未验证；本批不再增加候选。
目标：第五折>80.5且明确收益，完整OOF>82未完成。达成后固定方案做泛化确认，当前无补折授权。


## 2026-09-28T10:34:47+08:00 复杂预测头v6核对
全部完成=False；当前保留={'mode': 'previous', 'TOTAL': 80.27413533779057}；匹配差值={}。独立记录results/deep_learning/dual8222_attention_v6_20260928；无旧路线重启/补折或交付变更。


## 2026-09-28T10:40:07+08:00 复杂预测头v6核对
全部完成=False；当前保留={'mode': 'previous', 'TOTAL': 80.27413533779057}；匹配差值={'attention_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}}。独立记录results/deep_learning/dual8222_attention_v6_20260928；无旧路线重启/补折或交付变更。


## 2026-09-28T10:41:20+08:00 复杂预测头v6核对
全部完成=True；当前保留={'mode': 'previous', 'TOTAL': 80.27413533779057}；匹配差值={'deep_mlp_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}, 'attention_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}, 'attention_minus_deep_mlp': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}}。独立记录results/deep_learning/dual8222_attention_v6_20260928；无旧路线重启/补折或交付变更。


## 2026-09-28T10:42:57+08:00 复杂预测头v6已完成
三项均best_epoch0、stop18，原80.274135保持。实际更新后最高linear80.272137140362/deepMLP80.270535660860/attention80.272549291930；没有新增收益。attention相对偏置真实更新，mask四项检查通过。当前无训练进程，无补折或交付变更；分量query/cross-attention尚未实现或启动。重载不重复，保留原v5证明。详见results/deep_learning/dual8222_attention_v6_20260928/FINAL.md。


## 2026-09-28T10:53:13+08:00 分量query v7记录
全部完成=False；保留={'mode': 'previous', 'TOTAL': 80.27413533779057}；对照=None。记录results/deep_learning/dual8222_component_query_v7_20260928；无旧任务重启/补折/交付变更。


## 2026-09-28T10:56:42+08:00 分量query v7记录
全部完成=True；保留={'mode': 'previous', 'TOTAL': 80.27413533779057}；对照={'local_minus_center_TOTAL': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0], 'same_parameters': True}。记录results/deep_learning/dual8222_component_query_v7_20260928；无旧任务重启/补折/交付变更。


## 2026-09-28T10:58:22+08:00 分量query v7完成
两组均18轮自然早停并恢复第0轮80.274135；训练后最高center80.270925285299@9、local80.271695049988@8，无新增收益。原预测器全冻结；两组14081参数/初始化一致，query真实更新，局部与mask检查通过。保留旧v5，不重复旧权重重载，无旧任务/补折/交付变更；完整分布query和损失对齐未启动。详见results/deep_learning/dual8222_component_query_v7_20260928/FINAL.md。


## 2026-09-28T11:15:47+08:00 独立通道v8
观察时间：2026-09-28T11:15:47+08:00。已完成0/6项。
已做：模型拆分和显式启动，结果如本目录progress.json。
计划：完成有限六项并固定三模型共同重载。
未做：完整OOF和独立泛化、续跑等价性。目标达成后验证固定配方，当前无补折。


## 2026-09-28T11:19:42+08:00 独立通道v8
观察时间：2026-09-28T11:19:42+08:00。已完成0/6项。
已做：模型拆分和显式启动，结果如本目录progress.json。
计划：完成有限六项并固定三模型共同重载。
未做：完整OOF和独立泛化、续跑等价性。目标达成后验证固定配方，当前无补折。


## 2026-09-28T12:02:48+08:00 独立通道v8
观察时间：2026-09-28T12:02:48+08:00。已完成6/6项。
已做：模型拆分和显式启动，结果如本目录progress.json。
计划：完成有限六项并固定三模型共同重载。
未做：完整OOF和独立泛化、续跑等价性。目标达成后验证固定配方，当前无补折。


## 2026-09-28T16:19:00.119857+08:00 v13完成
两项强SW私有双专家损失对照完成18轮，均保留第0轮73.622384767129，无新增收益；更新后最佳分别73.47330202477876/73.45786228917683。原三独立模型80.274135337791保留，无新三模型重载宣称、无补折/交付更改。完整报告results/deep_learning/independent_sw_tolerance_v13_20260928/FINAL.md。下一步分组泛化诊断尚未启动，>82目标未完成。


## 2026-09-28T16:47:32.678918+08:00 v14内层诊断完成
两项均已自然结束。pointSW=76.23493007968595，wellSW=76.165444754071；差-0.06948532561494469。这是48→16内层SW，不是原第五折；原第五折未评估，旧80.274135保持。训练与监控轨迹及诊断已保存，无自动补折。报告results/deep_learning/sw_group_generalization_v14_20260928/FINAL_inner.md。


## v15三种完整骨干筛查完成
2026-09-28T17:11:23.595216+08:00。{'adaptive': 76.43706340163116, 'latent': 76.59073085729038, 'state_space': 76.38645544042299}；全部为48/16内层SW。是否结构信号=True。未训练原第五折，旧三私有模型80.274135保持。详细研究记录results/deep_learning/architecture_zoo_v15_20260928/FINAL.md。参数匹配消融/其他通道/完整OOF>82均未完成。


## v15完成后的机制验证
2026-09-28T17:14:37.682304+08:00。潜变量/SSM完整144621点磁盘复现；自适应采样完成有限批验证。潜变量为当前内层结构候选，未改原第五折模型；匹配容量/其他目标/跨折结果待做。详见architecture_zoo_v15_20260928/FINAL.md。


## v17最终完成
2026-09-28T19:37:40.442844+08:00。两组训练/原144665点磁盘重载与三模型实际合并已完成。shared_globalTOTAL=80.219534179，local_gateTOTAL=80.105625735；无新增收益，保留80.274135。未完成：新假设/其他通道结构/完整OOF>82。无第三候选、补折或交付变更。证据results/deep_learning/latent_gate_v17_20260928/FINAL.md。


## v18 POR结构对照完成
2026-09-28T20:16:03.903565+08:00。TCN TOTAL=79.550729668，latent TOTAL=79.147915298，保留TOTAL=80.274135338。两个新POR磁盘重载精确，固定PERM/SW数组合并实测；原模型与其他折未改。后续结构/完整五折未运行，当前无新增队列。记录results/deep_learning/por_architecture_v18_20260928/FINAL.md。


## v19多尺度POR对照已完成
2026-09-28T20:35:12.522433+08:00。regional TOTAL=79.551149548，global TOTAL=79.555720337，当前保留80.274135338。新POR磁盘复现及固定其余两通道数组合并完成。尚未运行新对照/其他目标/原其他折，完整OOF>82未完成。记录por_pyramid_v19_20260928/FINAL.md。


## v20特征与融合实验完成
2026-09-28T20:56:42.218786+08:00。三项全部完成并重载，特征TOTAL差+0.152891789，晚融合TOTAL差-0.080073697；保留TOTAL=80.274135338。仅本批新POR，其他目标/折/交付不变；强骨干特征接入与完整OOF尚未做。证据feature_structure_v20_20260928/FINAL.md。


## v21冠军收益与扩容实验完成
2026-09-28T21:18:10.261733+08:00。1/2/4内部宽度三项训练/最佳恢复/实际预测组合均完成；采纳新POR=False，保留TOTAL=80.274135338。已完成收益来源/参数和开销记录；整网统一拓宽、其他目标扩容、其他折均未执行。证据champion_scaling_v21_20260928/FINAL.md。


## v22 固定原权重扩容实验已完成
2026-09-28T21:47:56.088940+08:00。2倍更新后TOTAL=80.269258294；4倍=80.263249587。均18轮停止/最佳第0轮，保留80.274135338。新通路梯度、旧参数冻结、训练后检查点全144665点磁盘重载通过。未运行其他目标/原其他折，不改交付；详见frozen_growth_v22_20260928/FINAL.md。


## v23 从头双专家扩容对照已完成
2026-09-28T22:25:40.881346+08:00。base1 TOTAL=79.548859162；ffn2 TOTAL=79.541070769；容量差-0.007788393，当前保留80.274135338。两模型从随机权重训练/最佳磁盘重载/训练与验证诊断均完成。其他目标/原其他折及原多任务完整配方未执行；无自动后续。记录scratch_scaling_v23_20260928/FINAL.md。


## v24 专家监督对照已完成
2026-09-28T22:53:42.761471+08:00。mixture TOTAL=79.492115838，independent TOTAL=79.897835986；匹配差+0.405720148，保留80.274135338。两模型梯度隔离/最佳磁盘重载/实际评分完成，未改交付或补折。详见expert_supervision_v24_20260928/FINAL.md。


## v25 冠军原权重上的位置修正已完成
2026-09-28T23:11:49.047450+08:00。amplitude=80.295581058278，shape=80.274135337791，保留研究TOTAL=80.295581058278。仅12816新参数，旧模型冻结和更新后最佳全量重载已验证。冷启动/扩容停止延伸，无其他折/新队列/交付更改。详见champion_shape_v25_20260928/FINAL.md。


## v26 三目标输出头优化完成
2026-09-28T23:33:01.038838+08:00。11项完成，实际组合TOTAL=80.300730612039，相对v25+0.005149553761；采用POR:None,PERM:gate,SW:weight。新采纳目标全144665点独立磁盘重载，旧源权重与交付不改，无其他折。证据all_heads_v26_20260928/FINAL.md。


## v27基于80.30研究完成
六项完成，TOTAL=80.300730612039，相对父配方+0.000000000000。采用POR:None,PERM:None,SW:None。原权重/交付保持，无其他折；完整记录champion_posterior_v27_20260929/FINAL.md。


## v27后验条件比较图谱
2026-09-29T09:11:55.995080+08:00；3个目标节点保留6个子实验分数与证据，单通道诊断不冒充TOTAL。调用curate.py --v27-only增量维护，无需运行旧全量摘要生成覆盖后续记录。


## Atlas maintenance
Completed via curate_champion_posterior_v27.py and build_atlas.py. Original curate.py was not modified because the editing tool rejected it; the earlier proposed --v27-only hook was NOT installed. Existing annotations retained. No active training jobs at final check. See execution_notes.json.


## v28路由对照完成
六项完成；原第五折TOTAL=80.324397278674，相对v26+0.023666666635。POR:conditional, PERM:joint, SW:conditional。原模型/交付不变，无其他折。记录champion_route_v28_20260929/FINAL.md。


## v28路由最终图谱
2026-09-29T09:37:16.181800+08:00。三个目标节点保留六项匹配结果与最终选择；实际组合TOTAL=80.324397278674，相对v26+0.023666666635；14井提高、2井下降为该次组合开发诊断，非独立泛化确认。无新训练或其他折，旧交付保留。


## v29路由解耦完成
TOTAL=80.335082954320，相对v28+0.010685675646。POR:split, PERM:coupled, SW:coupled。原模型与交付不改，无其他折。证据champion_split_route_v29_20260929/FINAL.md。


## v29最终图谱
2026-09-29T10:12:23.418778+08:00；记录六项匹配结果及实际组合80.335082954320，父模型v28保持。独立增量脚本保留历史curation，未重写原curate.py。


## v30已有模块微调完成
实际TOTAL=80.336211945752，相对v29+0.001128991432。POR:coadapt, PERM:None, SW:None。无新结构、其他折或旧队列；原交付保留。记录champion_coadapt_v30_20260929/FINAL.md。


## v30最终图谱
2026-09-29T10:28:13.522070+08:00；保留六项实测结果、v29父关系及选定组合80.336211945752。未增加结构/其他折/旧任务，旧模型保留。


## v30最终收尾
2026-09-29T10:29:50.862061+08:00。六项训练结束，新增采用POR全144665点磁盘重载精确；实际TOTAL=80.33621194575169。PERM/SW保留原v29。图谱通过独立增量脚本更新；原curate.py编辑被内容保护拒绝，未改原文件。当前训练任务数=0，训练失败数=0。没有扩大本批搜索或其他折，旧交付保留。详见execution_notes.json。


## v30后优化方向诊断（无训练）
2026-09-29T10:50:34.147292+08:00。只完成当前预测误差分解与64训练井256固定窗口路由饱和统计，当前TOTAL80.336211945752不变。诊断：POR连续路由88.47%点触及95%以上边界；SW连续数值误差为主要失分；Q50覆盖仅0.303%，深度约0.1米均匀。下一步候选为同边界重参数化、单调条件分布修正、跨曲线协同特征；均未训练。报告results/deep_learning/optimization_review_v30_20260929/README.md；不补折、不改权重和交付。


## v31有限机制验证完成
TOTAL=80.336513436816，相对v30+0.000301491065；POR:anchored, SW:None。四项结束，无其他折；证据mechanism_checks_v31_20260929/FINAL.md。


## v31最终图谱
2026-09-29T11:10:48.016430+08:00。四项完成，TOTAL=80.336513436816；保存POR极小增量和SW两项负结果，单通道分数仍为diagnostic，无其他折。


## v32跨曲线特征对照完成
TOTAL=80.336513436816，相对v31+0.000000000000；POR:None, PERM:None, SW:None。六项完成，无其他折。证据crosscurve_features_v32_20260929/FINAL.md。


## v32最终图谱
2026-09-29T11:37:31.747868+08:00。六项实际结果及父模型关系已记录，TOTAL=80.336513436816；未采用的设置标注有限负结论，不重复启动。


## v33 输入边界上下文图谱
2026-09-29T12:29:44.622942+08:00；增量条目由curate_boundary_v33.py维护，原curate.py和其他节点不覆盖。最终分数口径为单通道第五折。


## v33输入边界对照完成
四项完成，TOTAL=80.336513436816，增量=+0.000000000000；新采用=[]。输入边界与失分相关，统计截断的实际结果见boundary_context_v33_20260929/FINAL.md。无其他折、旧任务或交付覆盖。


## v33最终结论
四项已完成，当前TOTAL80.336513436816不变，无活动训练。输入近边界非默认样本很难但总体贡献有限：近±5点失分占2.08%，远>25点占87.92%。不继续硬截断统计扫描。报告results/deep_learning/boundary_context_v33_20260929/FINAL.md。未补其他折、未修改旧交付。


## v34跨目标证据图谱进行中
2026-09-29T13:29:20.861587+08:00；辅助NLL与官方SW分数分开，原curate.py及历史训练源保持。


## v34跨目标协同完成
12个井外辅助模型及3项SW头完成，TOTAL=80.336513436816，增量=+0.000000000000。采用=None；微型辅助信号，不是强模型全部OOF重训。无原其他折/交付变更。详见cross_target_v34_20260929/FINAL.md。


## v34跨目标证据图谱完成
2026-09-29T13:32:48.956573+08:00；辅助NLL与官方SW分数分开，原curate.py及历史训练源保持。


## v36检索图谱进行中
2026-09-29T14:26:47.723007+08:00。v35门控失活证据与v36有效训练分别记录，未覆盖旧来源。


## v36检索门控对照完成
两项已完成，TOTAL=80.336513436816，增量=+0.000000000000；采用=None。v35失活证据和v36有效训练区别已记录。所有旧模型/交付保留，无其他折；见retrieval_smooth_gate_v36_20260929/FINAL.md。


## v36检索图谱完成
2026-09-29T14:35:16.978459+08:00。v35门控失活证据与v36有效训练分别记录，未覆盖旧来源。


## v37 closure and storage
The four candidates finished and restored successfully; no candidate was adopted. Keep the v31 recipe at TOTAL80.336513436816. Uniform weighting has only small post-update relative gains over Q50, not an improvement over the parent. Do not expand this bounded all-factor head sweep. Next independent hypothesis to assess: train a small direct action-value readout against the original scoring kernel, with the original density decoder retained as control; this is not implemented or launched yet. Full OOF>82 remains unachieved; no other folds or old queues started.
Removed 22 non-adopted checkpoints/cache files totaling104118150 bytes across v35-v37, with SHA256/size manifests. Sources, configurations, compact predictions, reports and all baseline/deployment dependencies remain.


## v38 closure and retention
All four valid candidates stopped after18 non-improving epochs and restored epoch0 exactly. Keep baseline TOTAL80.336513436816. Original POR wrong-inverse run excluded; corrected2048-point training-kernel max error4.8336e-6, SW also checked at finalization. Original launcher now only admits SW; POR has a separate corrected launcher. No selected replacement.
15 negative/invalid checkpoints totaling30001900 bytes deleted after final verification, SHA256/size manifest saved. Compact source/config/log/score/prediction and failure evidence retained, all baseline checkpoints preserved. No live training, automatic queue, extra folds or delivery changes. FullOOF>82 remains unachieved.
Do not expand the v37/v38 frozen output-head sweeps. Next work should assess a different update location inside the retained private backbone, or measure training-vs-validation feature behavior to justify it; this is a pending hypothesis, not an implemented or launched experiment.


## v39 closure and next hypothesis
Both SW normalization scopes completed, restored baseline and passed fixed-state/prediction checks. TOTAL80.336513436816 remains current; no extra folds, active training or delivery replacement. Deleted4 redundant checkpoints totaling8015925 bytes with manifest; source/config/logs/compact predictions/score evidence and all baseline dependencies retained.
No further frozen-output-head or normalization-only LR sweep is planned. A different hypothesis for a later isolated study is training-well nuisance dependence: compare matched normalization adaptation with a training-only well discriminator gradient blocked versus reversed. Well identity would be auxiliary supervision only, never an input to the prediction, and no validation wells would train the discriminator. This proposal is not implemented or launched, and no gain is claimed. It requires separate scope/gradient/loss records if pursued. Full OOF>82 remains outstanding.


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 before reversal starts. For the prescribed active epochs4..18, blocked best73.5954025454393 and reversed best73.59465943798016 (delta-0.000743107459). Final SW difference-0.006539743169. This diagnostic does not change checkpoint selection. Final domain accuracy blocked0.165039/reversed0.162842 vs majority prior0.022486: classifier learns, but the reversed objective does not improve this SW adaptation setting.
Do not extend this adversarial coefficient/curriculum scan. Next pending direction: inspect PERM conditional-density family/loss alternatives against the current retained PERM89.396724 model; preserve original actions/scoring and distinguish an initial distribution intervention from a retraining benefit. No new PERM run implemented or launched yet. Original fullOOF>82 and fifth>80.5 remain outstanding.


## v40 retention and terminal state
Both candidates finished with no adopted model. Deleted4 redundant checkpoints totaling9012676 bytes after verified restoration and fixed3column scoring; SHA256/size manifest saved. Source/config/log/compact predictions/diagnostics retained. No live training or queued launch, no other folds and no delivery changes. Current recipe still TOTAL80.336513436816.


## v41 PERM分布族完成
Gaussian与Student-t3均18轮结束，均回退原PERM。训练后最佳89.367194812859/89.372730778423，父模型89.396724049366。实际三列TOTAL=80.336513436816不变，无其他折/旧队列；详细证据见results/deep_learning/perm_family_v41_20260929/FINAL.md。目标仍未完成，不延伸本设置分布族扫描。

## v41负权重清理
已完成恢复和实际三列评分，删除本实验负候选checkpoint共5个、7224100字节；源/配置/训练日志/紧凑预测和原冠军依赖保留。清单见perm_family_v41_20260929/deletion_manifest.json。无活动训练或自动队列。



## v42末端块原权重对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见anchored_backbone_v42_20260929/FINAL.md。

## v42归档与下一步边界
两项均18轮完成，固定组合TOTAL80.336513436816。原点约束相对free训练后最高SW+0.002969806658，18/18匹配轮次更高，中位差+0.108804776891；仅是单种子开发折内减少退化的证据，未超原SW。删除负free状态与两份第0轮重复模型共3个、4366424字节；保留anchored完整last_state约4.2MB用于局部正机制对照，不作为新交付。清单见deletion_manifest.json。无活动训练或队列；原其他折未启动。后续可在不同更新部位或目标的独立假设中携带原点约束，尚未实现或启动；不继续本dec0的系数/LR扫描。



## v43完整特征骨干对照完成
两项完成，TOTAL=80.336513436816，增量+0.000000000000，采用=None。无其他折或交付修改；见anchored_features_v43_20260929/FINAL.md。

## v43归档结论
两项18轮结束，训练后SW最高free73.505103279、anchored73.507620278，均低于原73.663082797；实际三列TOTAL80.336513436816保持。约束相对+0.002516999785，11/18对应轮次更高，中位+0.001984832480，完整骨干未证实稳定收益。关闭本配置LR/系数扫描，后续需切换机制或目标，尚无新任务。删除本批4个未采用权重共59620738字节，原冠军/v42局部对照/交付保留，源/配置/轨迹/紧凑预测保留。清单见deletion_manifest.json。第五折>80.5与完整OOF>82未完成，无其他折、活动训练或队列。



## v44默认状态消息完成
两项完成，TOTAL=80.336513436816，相对80.336513增加+0.000000000000；采用[]。完整记录state_messages_v44_20260929/FINAL.md；无其他折或交付改动。

## v44完成与归档
self/neighbor均第18轮结束，训练后最高SW73.604036960754/73.610973025844，均未超过原73.663082797066。实际TOTAL80.336513436816不变，无其他折或队列。neighbor-self局部+0.006936065090，11井升/5井降，11/18对应轮次更高，中位+0.003280926577；开发折局部证据，不称稳定新井收益。保留两个第15轮model_best_updated小头作为相对机制与匹配对照，未采用其预测。删除完整末态与两个第0轮重复权重共3个、8262546字节；原模型/交付不变，清单见deletion_manifest.json。当前消息步数/耦合界不扫描；可将局部邻点机制带入新独立设计，尚未实施或启动。目标第五>80.5及完整OOF>82未完成。



## v45两条新架构完成
SW长井段与POR九曲线轴向完成，实际TOTAL=80.336966691471，增量+0.000453254655，采用[('SW', 'long_section')]。详见divergent_models_v45_20260929/FINAL.md；未补其他折，旧交付不变。

## v45最终候选状态与清理
新SW683033参数，第12轮73.664377810365，30轮早停；磁盘重载全144665点一致。与原POR/PERM实际组合TOTAL80.336966691471，比原80.336513仅+0.000453254655。8井升/8井降，未调整选择的井级TOTAL区间[-0.3040,0.2735]跨零，因此仅保留数值候选及新结构起点，不宣称明确泛化收益，不启动补折。POR九曲线轴向506145参数，第37轮最好73.269695792712，第55轮自然patience18结束，低原4.282241；提前停止请求到达时已结束，保留late_stop_request_record.json，不冒称人工终止。两项总墙钟350.63913秒、max2GPU，单项196.48/309.45秒，峰值1184.93/1149.31MiB。删除POR负权重2个、12341946字节；SW新网络/原冠军/旧交付保留。下一步应从新结构或井尺度条件建模出发，不退回冻结小头连续微调；未新增任务/队列，完整OOF>82仍未完成。



## v46井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用well，训练后well-reference=+0.147097342423。详见well_conditioned_v46_20260929/FINAL.md；无其他折或交付修改。

## v46最终候选状态与清理
实际组合TOTAL80.342771761689，相对v45增加0.005805070217；SW73.680963725271，第9轮最好、第27轮patience18结束。当前井条件与固定参考同容量/同初始化；固定参考训练后最高73.533866382849，第18轮早停回退起点73.664377810365。训练后最佳差+0.147097342423，但相对可保留的起点仅+0.016585914907，不能混淆两种收益。
16验证井7升9降；SW默认值部分-0.928308，连续数值部分+2.186920，抵消后仅微增；未调整开发选择的井级TOTAL区间[-0.139812,0.154816]跨零。只保留局部数值候选和已证实的匹配对照差，不宣称稳定泛化收益；严格>80.5及完整OOF>82仍未完成。
全144665点磁盘重建与存档预测一致，实际三列评分完成。两任务总墙钟244.526723秒（含加载、预热、输入库构建及启动间隔），单项156.79/209.25秒，峰值各1252.30MiB、最大并发2；编写与最终重载另计。负参考权重已按deletion_manifest.json删除，保留当前井模型、完整恢复状态、父SW/原冠军及交付。
本批训练全部结束，无自动队列、其他折或交付覆盖。下一批可沿井条件机制设计新的独立结构对照，重点解决默认状态与连续值的损益抵消；不直接密集扫描点数/调制界，不把待实施设想写成完成事实。


## v47任务查询对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后section-global=-0.003543385011。详见well_memory_v47_20260929/FINAL.md；无其他折或交付修改。

## v47最终收尾与有效训练证据
两组均第18轮自然patience18停止，最佳恢复第0轮，不能把回退后的73.680964写成训练后新成绩。真正训练后最高均第9轮：global_query SW73.476844096201（低父0.204119629070）；section_query SW73.473300711190（低父0.207663014081），局部查询较窗口平均-0.003543385011。采用新模型=False，原组合TOTAL80.342771761689保持，未达80.5/完整OOF82。
两组完整最终状态显示记忆编码器、key/value、两组query及投影均实际更新；训练损失约0.04465降至0.01725。memory_training_evidence.json记录各参数变化。最佳回退的零投影不代表训练支路失活；本轮负结果可据此关闭当前配置，停止扫点数/头数/查询维数，不泛化到所有记忆模型。
同初始化、初始父函数、最佳恢复与144665点覆盖通过；原POR/PERM与保留SW实际三列评分完成。单项212.33/210.04秒，双并发总墙钟230.554407秒，含加载/预热/输入库/启动间隔，最大并发2，显存峰值各1397.69MiB。源编写与最终收尾另计。
已删除两负候选的best/last四个权重文件，具体字节/哈希见deletion_manifest.json；保留源码、配置、曲线、预测、紧凑诊断、当前v46和父v45权重及原交付。当前无训练进程/自动队列/补折。下一批应换训练表征或输出分布机制，当前注意力记忆路线不继续；尚未实施的新设想不计收益。


## v48任务查询对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后rqs-linear=+1.292082008243。详见quantile_density_v48_20260929/FINAL.md；无其他折或交付修改。

## v48最终收尾：保留有对照收益的平滑密度
两组新密度初始SW均72.346441547110，非父模型73.680963725271。linear训练后最高第7轮72.136427521820，第18轮早停、回退自身第0轮；rqs训练后最高第4轮73.428509530063，第22轮早停。RQS较linear含初始最佳+1.082067982952SW（等价TOTAL+0.378723794033），仅比较训练后最高差+1.292082008243；这两个差值不能混用。
同容量/同初始化对照13井升3井降，选择未校正的探索性井级TOTAL区间[0.163806,0.602874]。因此保留RQS模型作为有局部机制收益的候选；不因绝对低于当前SOTA而删除。RQS实际替换组合TOTAL80.254412793365，仍低当前80.342771761689（SW差-0.252454195209），不采用进当前配方、不补折。完整OOF>82仍未完成。
新分布解析密度在8个实际训练锚点输出上与CDF自动微分相符，最大相对误差linear5.94e-7/rqs8.38e-7，全实数CDF端点精确0/1；这是新数学实现的单次最终检查，不是重复预检。RQS最佳权重磁盘重建全144665点与存档完全一致，证据retained_rqs_verification.json。原最终验证已恢复各自最佳；原三目标保留组合实际评分完成。
单项158.51/183.54秒，双并发总墙钟213.127631秒，含加载/预热/输入库/启动间隔；峰值各1261.56MiB、max2GPU。代码编写与最终RQS重载检查2.21秒另计。两项已结束，无自动队列。删除linear两个负权重，保留RQS最佳/完整状态、v46正模型、父模型和旧交付；哈希/字节见deletion_manifest.json。
不继续当前线性密度、分位点数或尾概率扫描。后续可带着已证实的平滑插值机制，独立研究分位位置的监督或分布训练方法；这只是下一步方向，尚未训练，不宣称已取得新收益。目标、剩余工作及完成后独立井确认/新目录交付继续维护。


## v49任务查询对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后aux-likelihood=+0.007017394414。详见quantile_supervision_v49_20260929/FINAL.md；无其他折或交付修改。

## v49最终结果与清理
likelihood训练后最佳第1轮SW73.191339565226；quantile_aux训练后最佳第1轮73.198356959640。训练后辅助项较对照仅+0.007017394414，但两者都低于同起点73.428509530063，均第18轮patience18停止、最佳选择回退第0轮。按预定义包含起点的选择规则，辅助监督最终收益=0；不能把最佳回退或训练后微小差写成新有效模型。
当前原三列组合TOTAL80.342771761689保持，新采用=False，第五严格>80.5及完整OOF>82均未完成。原v48平滑插值对照收益仍有效，其RQS权重继续保留；本轮关闭当前辅助分位项系数/节点数扫描，不否定其他分位训练方法。
初始完整144665点与v48源预测一致、两个初始状态哈希相同、结束最佳恢复一致，来源哈希与实际三列评分已核验；真实训练中骨干、FiLM、密度与默认头梯度均非零。新候选没有被保留，因此不重复执行原样条密度公式或原父模型磁盘验证。
单项166.8964/167.2358秒，两项总墙钟193.43774秒，max2GPU，各峰值1261.56MiB；含加载/预热/输入库及启动间隔，编写与最终收尾另计。已删除两项best/last四个冗余权重，保留源码/配置/曲线/预测/成绩与删除哈希清单。v48 RQS和当前v46、所有父依赖及旧交付保留。
本批已全部结束，无训练进程、队列或其他折。下一批需切换当前辅助损失的思路，不继续0.5附近扫描；下一项尚未实施，不能算作进展或收益。独立新井确认/完整OOF和新目录交付仍未完成。


## v50井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后transfer-scratch=+0.239511576802。详见perm_transfer_v50_20260929/FINAL.md；无其他折或交付修改。

## v50最终结果：保留训练井内迁移候选
两组初始PERM先验68.383772436967，完整初始预测/输出头哈希一致，骨干哈希按设计不同。scratch第19轮最佳88.811295997433，第37轮patience18结束；sw_transfer第21轮最佳89.050807574234，第39轮结束。迁移较随机+0.239511576802PERM（等价TOTAL+0.083829051881），但较当前PERM89.396724049366仍低0.345916475131。
匹配差9井升7井降，选择未校正的探索性井级TOTAL区间[-0.081486,0.252971]跨零；仅保留局部对照收益，不宣称稳定新井收益。迁移实际替换组合TOTAL80.221700995393，未采用；当前CURRENT_RECIPE三列仍为TOTAL80.342771761689。严格第五>80.5与完整OOF>82仍未完成，无其他折。
迁移最佳模型磁盘重建全144665点与存档完全相同，实际三列评分一致，证据retained_transfer_verification.json。初始化/最佳恢复、源码哈希、相同输出头及数据来源已核验；训练骨干/头/FiLM均有真实梯度。汇总器曾把NPZ井编号字段误替换成模式名，已修复并记录finalization_failure_fixed.json；没有修改训练源码、权重或重训。
单项327.53/331.33秒，总墙钟355.129723秒，含加载/预热/输入库和启动间隔；max2GPU、各1252.30MiB。代码编写与最终磁盘重建1.59秒另计。两项全部结束无队列，删除scratch两个负权重，保留迁移最佳/完整状态、当前三目标、v48平滑候选及所有父依赖/交付。哈希/字节见deletion_manifest.json。
本次没有回到SW辅助损失扫描；后续可携带迁移表征设计新的PERM结构/训练方案，当前只是方向，未实施。随机冷启动当前配置不延伸，不能把初始化到88分的学习当成超越当前模型。独立新井确认、完整五折和新目录交付仍未完成。


## v51井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后interwell-plain=+0.031618982727。详见perm_moment_mix_v51_20260929/FINAL.md；无其他折或交付修改。

## v51最终结果与实际增强证据
plain训练后最好第3轮PERM88.890099775317，interwell最好第5轮88.921718758043。训练后增强较对照+0.031618982727，但两者均低于同起点89.050807574234，均第18轮patience18结束并恢复第0轮；按预定义包含起点的最终选择，收益=0，不产生新有效模型。
增强组真实训练1152批，累计36928窗口应用增强，异井配对数同为36928；对照增强数0。最佳恢复后计数0是回退起点，不是模块失活，实际末态证据保存在report的last_training_augmentation和逐轮历史。验证始终关闭增强，无新增井ID特征/验证统计拟合。
两组初始参数、输出头和144665点预测哈希一致，初始精确复现v50，结束最佳恢复一致，源哈希及原三列实际评分已完成。当前TOTAL80.342771761689保持；原PERM89.396724、v50迁移候选和v48样条候选都保留。原第五严格>80.5与完整OOF>82未完成，不启动其他折。
单项146.9093/149.3306秒，总墙钟179.067863秒，max2GPU，各峰值1252.30/1252.41MiB，含加载/输入库/预热/启动间隔，代码与收尾另计。两项全部结束无队列；删除本轮两项best/last四个冗余权重，保留来源和旧交付，哈希/字节见deletion_manifest.json。原CDF/父模型检查未重复运行。
本轮停止此概率/Beta/层位置扫描，只限当前暖启动PERM与此增强位置。训练后减轻退化的微差不能升级为验证达标或稳定泛化改善；下一批需换机制，尚未启动。独立井确认/完整OOF/新目录交付仍未完成。


## v52井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后content-position=-0.026171081763。详见perm_token_pool_v52_20260929/FINAL.md；无其他折或交付修改。

## v52最终结果与清理
position第22轮最好PERM88.894217488422，第40轮patience18结束；content第24轮最好88.868046406659，第42轮结束。内容-位置=-0.026171081763，无匹配收益。两者也低于历史v50普通平均汇聚89.050807574234（分别-0.156590085813/-0.182761167576），以及当前PERM89.396724049366。历史比较与本批同容量比较分开，不能把初始先验68.383772到88分的学习当超越当前模型。
两项初始状态/输出头/144665点先验预测哈希一致；真实打分器梯度非零，选定权重范数position0.253729/content0.350035，内容权重熵训练中已偏离均匀值，非失活试验。最大池化权重可受边界仅少数有效点影响，不将单个max数值当全窗口尖锐性证据。
最佳恢复全144665点一致、来源哈希与真实三列组合评分完成。没有采用新模型，CURRENT_RECIPE仍TOTAL80.342771761689；严格第五>80.5及完整OOF>82未完成，无其他折。负候选不追加重复磁盘/原密度验证。
单项379.7283/389.8115秒，总墙钟414.115675秒，max2GPU，峰值1319.99/1291.71MiB；含加载/预热/输入库/启动间隔，代码与最终收尾另计。全部结束无训练队列；删除四个本轮best/last负权重，保留代码/配置/日志/预测/梯度与紧凑报告。v50迁移、v48样条、当前三目标及源依赖/旧交付不改，哈希/字节见deletion_manifest.json。
关闭当前汇聚组数/步长/温度扫描，不否定所有自适应token化。下一批需换机制，尚未实施，不能算作收益。独立井确认、完整五折及新目录交付仍未完成。


## v53井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后transfer-scratch=+0.248076966158。详见por_transfer_v53_20260929/FINAL.md；无其他折或交付修改。


## v53最终结果与保留边界
scratch：POR 73.640524506112，最佳第32轮，停止第50轮（patience18）。
sw_transfer：POR 73.888601472270，最佳第62轮，停止第80轮（patience18）。
迁移-随机POR=+0.248076966158；当前POR=77.551936801885，实际组合TOTAL=80.342771761689。新采用=False。
迁移有匹配局部收益，保留迁移最佳及完整状态；不将其低于现有POR的绝对成绩写成新冠军。该收益来自反复使用的开发折，未完成独立井确认。
两组同836921参数、相同初始POR头/144665点先验、不同初始骨干；全骨干只用POR标签与Student-t3损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=True。不重复未改密度公式检查。
单项耗时分别[466.646, 623.297]秒；双并发总墙钟642.362秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1252.51, 1252.51]，代码与最终重建另计。
已删除2个本轮负检查点，共19048838字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v54井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后stratified-uniform=+0.001729531416。详见sw_stratified_v54_20260929/FINAL.md；无其他折或交付修改。


## v54最终结果与保留边界
uniform：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
stratified：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
分层-随机SW=+0.000000000000；当前SW=73.680963725271，实际组合TOTAL=80.342771761689。新采用=False。
分层无最终匹配收益，停止当前32/32比例/LR扫描；不把训练后两个退化成绩间的小差写成有效模型。
两组同836921参数、同v46SW完整起点/144665点预测；全骨干只用SW标签与Gaussian损失。两组共同窗口均值风险，分层32/32且按训练状态比例校正，未改变状态风险目标；原Q保持。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[165.044, 161.629]秒；双并发总墙钟186.892秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1257.47, 1257.47]，代码与最终重建另计。
已删除4个本轮负检查点，共37953164字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v55井条件对照完成
两项完成，实际TOTAL=80.342771761689，采用None，训练后sam-adamw=+0.104051059886。详见sw_sharpness_v55_20260929/FINAL.md；无其他折或交付修改。


## v55最终结果与保留边界
adamw：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
sam：SW 73.680963725271，最佳第0轮，停止第18轮（patience18）。
SAM-AdamWSW=+0.000000000000；当前SW=73.680963725271，实际组合TOTAL=80.342771761689。新采用=False。
SAM无最终匹配收益，停止当前rho/优化器组合扫描；不把训练后两个退化成绩间的小差写成有效模型。
两组同836921参数、同v46SW完整起点/144665点预测；全骨干只用SW标签与Gaussian损失。两组共同原批次点均值风险；SAM同批两遍、固定rho.05、精确恢复扰动权重后更新，原Q保持，非等计算量。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[170.059, 246.178]秒；双并发总墙钟263.480秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1249.71, 1256.12]，代码与最终重建另计。
已删除4个本轮负检查点，共37951500字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v56井条件对照完成
两项完成，实际TOTAL=80.354325927816，采用additive，训练后multiplicative-additive=-0.191968804219。详见sw_interaction_v56_20260929/FINAL.md；无其他折或交付修改。


## v56最终结果与保留边界
additive：SW 73.713975628494，最佳第11轮，停止第29轮（patience18）。
multiplicative：SW 73.522006824275，最佳第3轮，停止第21轮（patience18）。
乘法-加法SW=-0.191968804219；当前SW=73.680963725271，实际组合TOTAL=80.354325927816。新采用=True。
乘性交互无匹配收益，停止当前分支宽度/层数扫描；只限当前模块和训练配置，不泛化所有门控网络无效。
两组同839481参数、相同完整初始参数与144665点先验；四个内部FF只差加法或乘法，同公共LayerNorm，SW输出全重置；全网络只用SW标签与Gaussian损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=True。不重复未改密度公式检查。
单项耗时分别[270.365, 223.589]秒；双并发总墙钟274.873秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1266.92, 1276.45]，代码与最终重建另计。
已删除2个本轮负检查点，共19051526字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v56当前配方与归因限制
当前采用加法组第11轮，SW73.713975628494（相对v46+0.033011903223），原第五实际组合TOTAL80.354325927816（+0.011554166128）。乘法组第3轮73.522006824275，相对加法-0.191968804219；此项配对不支持乘性交互。加法相对历史源的微增涉及新FF/公共归一化/重置输出和重新训练的整体配置，未单独证明其中一个因素因果增益，也未完成独立井确认。完整144665点新SW磁盘重建和固定三列实际评分已完成；清理阶段复用未改权重的核验凭据，不重复同一推理。
原206已有井内秩，初步秩特征构想未训练，不能计作新实验收益。保留新加法best/last和旧v46父模型、v53/v50迁移及v48样条正候选，旧交付不改。乘法负权重已删。第五严格>80.5与完整OOF>82仍未完成，不补其他折；下一项尚未启动。


## v57井条件对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后grow128-base=-0.001879626891。详见sw_ffgrowth_v57_20260929/FINAL.md；无其他折或交付修改。


## v57最终结果与保留边界
base：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
grow128：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
扩容-基准SW=+0.000000000000；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
并行扩容无匹配收益，停止当前并行分支宽度/层数扫描；只限此初始化和共同学习率，不泛化所有大模型无效。
base839481/grow128989369参数，公共核心及完整144665点初始预测相同，新增四个128并行FF零输出；原输出头不重置，全网络只用SW标签与Gaussian损失，同lr3e-5。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[166.372, 179.907]秒；双并发总墙钟202.907秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1266.92, 1291.83]，代码与最终重建另计。
已删除4个本轮负检查点，共41123212字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v57实际末态与结论边界
{"post_update_best": {"base": {"epoch": 1, "score": 73.51945824737717}, "grow128": {"epoch": 1, "score": 73.51757862048666}}, "extra_training": {"last_epoch": 18, "selected_epoch": 0, "last_extra_output_L2": 0.25923800468444824, "scope": "last actual trained state; selected epoch0 has zero extra outputs"}}
新增分支首个实际第二步梯度非零、末态输出权重L2为0.259238，说明并行模块参与了训练；最佳恢复后为零是回退第0轮，不是模块失活。完整公共核心/初始函数相同已核验，参数量不同为预定容量变量。两组原18轮patience结束，训练后最佳扩容较基准-0.001880且均低起点，没有扩容收益；保留v56当前配方，停止此并行宽度/层数扫描。未增加其他折或修改交付。


## v58井条件对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后headstart-joint=+0.120651423172。详见sw_rqs_staging_v58_20260929/FINAL.md；无其他折或交付修改。


## v58最终结果与保留边界
joint：SW 73.587566682801，最佳第4轮，停止第22轮（patience18）。
headstart：SW 73.708218105973，最佳第2轮，停止第20轮（patience18）。
先适配-共同SW=+0.120651423172；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
先适配有匹配局部收益，保留其最佳及完整状态；不将其低于现有SW的绝对成绩写成新冠军。该收益来自反复使用的开发折，未完成独立井确认。
两组同840771参数及初始RQS函数，v56加法骨干和默认头保留；新密度训练侧先验。headstart前5轮仅4386密度参数，第6轮全解冻，其余相同；全程只用SW标签及原Q/RQS解析损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=True。不重复未改密度公式检查。
单项耗时分别[222.114, 191.98]秒；双并发总墙钟226.230秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1277.44, 1278.72]，代码与最终重建另计。
已删除2个本轮负检查点，共19070070字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v58真实阶段与绝对边界
{"joint": {"stopped_epoch": 22, "best_epoch": 4, "density_only_epochs": 0, "backbone_update_steps": 1408, "selected_body_parameter_change_L2": 1.4109308771569136}, "headstart": {"stopped_epoch": 20, "best_epoch": 2, "density_only_epochs": 5, "backbone_update_steps": 960, "selected_body_parameter_change_L2": 0.0}}
先适配最佳第2轮发生在仅密度更新阶段，选定骨干变化L2为0；第6轮确实解冻，所有组梯度非零，但之后未超过第2轮。不能把此次收益解释为解冻后全网络提升。两组初始SW72.415319274399相同，先适配最终73.708218105973较共同73.587566682801高0.120651423172，但比当前73.713975628494低0.005757522521。先适配实际替换TOTAL80.352310794934，当前组合80.354325927816保持。
保留先适配best/last与完整144665点磁盘重建凭据；共同训练负权重已删除，原RQS/v56及其他正候选、交付保留。分位点/尾质量/公式均未改，旧数值验证不重复。复用的通用配置描述提到linear/RQS是实现支持范围，本批density_mode固定rqs，两项只差阶段；未训练linear。独立井确认、第五严格>80.5及完整OOF>82仍未完成。


## v59最终结果
实际TOTAL=80.354325927816，SiLU-affine SW=+0.000000000000，采用None，保留[]。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，失败和耗时如实记录。详见sw_cached_readout_v59_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。


## v60最终结果
实际TOTAL=80.354325927816，Brier-likelihood SW=+0.000000000000，采用None，保留[]。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，耗时如实记录。详见sw_brier_v60_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。


## v61深度还原对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后local-coarse=-0.007137240669。详见sw_upsample_v61_20260929/FINAL.md；无其他折或交付修改。


## v61最终结果与保留边界
coarse_query：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
local_query：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
局部-粗查询SW=+0.000000000000；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
精细局部查询无匹配收益，停止当前插值rank/邻居数/温度扫描；只限当前双邻居结构，不泛化所有神经上采样。
同842585参数、完整参数与144665点初始预测相同，新增rank16双邻居插值模块3104参数，query零初始；原输出头不重置，全网络只用SW标签与Gaussian损失，同lr3e-5。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[179.664, 178.092]秒；双并发总墙钟187.754秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB=[1359.76, 1359.92]，代码与最终重建另计。
已删除4个本轮负检查点，共38219084字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v62密度私有骨干对照完成
两项完成，实际TOTAL=80.354325927816，采用None，训练后numeric-aux=-0.013721007637。详见sw_density_private_v62_20260929/FINAL.md；无其他折或交付修改。


## v62最终结果与保留边界
gate_aux：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
numeric_only：SW 73.713975628494，最佳第0轮，停止第18轮（patience18）。
仅数值-辅助SW=+0.000000000000；当前SW=73.713975628494，实际组合TOTAL=80.354325927816。新采用=False。
仅数值监督无匹配收益，停止本辅助系数和支路数扫描；只限当前固定默认/私有密度结构，不泛化全部任务解耦。
同1678962总/839481优化器参数、完整初始状态及144665点预测相同；默认分支冻结，私有密度骨干全训练，自己的gate仅作可选辅助，numeric_only额外129个gate参数无梯度且不用于推理。原SW/Q/Gaussian损失、lr3e-5保持。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建=False。不重复未改密度公式检查。
单项耗时分别[188.23, 186.628]秒；初始并发失败加串行恢复的总墙钟447.500秒，包含失败尝试/恢复间隔/加载/输入库/预热；报告内seconds不含未保存的失败轮工作，attempts记录其墙钟；每任务峰值MiB=[1282.68, 1282.68]，代码与最终重建另计。
已删除4个本轮负检查点，共58450492字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。
第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。


## v63最终结果
两项串行完成，TOTAL=80.354325927816，local-point POR=+0.000000000000（训练后差+0.004661976642）。采用=None，保留[]，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见por_internal_adapter_v63_20260929/FINAL.md。


## v64最终结果
一项新训练完成，复用v63 point对照。TOTAL=80.354325927816；interval-control POR=+0.000000000000（训练后差-0.006178360861）；采用=False，保留=False。负权重清理，来源/交付不改；无其他折或队列，目标未完成，见por_interval_v64_20260929/FINAL.md。


## v65最终结果
两项串行完成，TOTAL=80.354325927816；joint-detached SW=+0.000000000000，训练后差=-0.017791342092；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_transition_v65_20260929/FINAL.md。


## v66最终结果
两项完成，TOTAL=80.354325927816；ordered-permuted SW=+0.000000000000，训练后差=+0.007728273109；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_well_order_v66_20260929/FINAL.md。


## v67最终结果
两项完成，TOTAL=80.354325927816；rqs-linear PERM=+0.221226055194，训练后差=+0.221226055194；采用=None，保留=['rqs']。负权重清理，源/交付不改，无其他折；目标未完成，见perm_quantile_v67_20260929/FINAL.md。


## v68最终结果
两项完成，TOTAL=80.354325927816；project-sum PERM=+0.000000000000，训练后差=+0.006571455767；采用=None，保留=[]。负权重清理，源/交付不改，无其他折；目标未完成，见perm_gradient_v68_20260929/FINAL.md。


## v69最终结果
一个新候选完成，复用历史sum控制，TOTAL=80.354325927816；curvature-sum PERM=+0.000000000000，训练后差=+0.011725166945；采用=None，保留=[]。负权重清理，源/交付不改，无其他折；目标未完成，见perm_curvature_v69_20260929/FINAL.md。


## v70最终结果
两项完成，TOTAL=80.354325927816，nonlinear-affine PERM=+0.000000000000（训练后差+0.000421751979）。采用=None，保留[]，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见perm_cdfwarp_v70_20260929/FINAL.md。


## v71最终结果
两项完成，TOTAL=80.354325927816；directional-symmetric SW=+0.000000000000，训练后差=-0.018606844709；采用=None。负权重清理，源/交付不改，无其他折；目标未完成，见sw_depth_attention_v71_20260929/FINAL.md。


## v72最终结果
两项完成，TOTAL=80.354325927816；interval-point PERM=+0.000000000000，训练后差=-0.006504305345；采用=None，保留=[]。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见perm_observation_v72_20260929/FINAL.md。


## v73最终结果
两项完成，TOTAL=80.354325927816；ordered-permuted SW=-0.010634580411，训练后差=-0.010634580411；采用=None，保留=[]。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_patch_moments_v73_20260929/FINAL.md。


## v73 文档编码修复
v72/v73启动说明经PowerShell管道传输曾损坏中文，已按各自PLAN.md修复研究/输出README、PROJECTS、AGENTS及入口登记；不改已训练的模型源码、配置、哈希、分数或时间。v73初始化scope中的继承静态random body文案已单独澄清：实际载入v56骨干，只有输出重置。


## v74最终结果
两项完成，TOTAL=80.354325927816；dropkey-control SW=+0.120506927208，训练后差=+0.120506927208；采用=None，保留=['dropkey']。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_key_dropout_v74_20260929/FINAL.md。


## v74收益来源与后续依据
默认SW子群改善+0.709667913805，数值SW子群变化-1.232741305323；16井中13胜/3负。总体SW配对+0.120506927208成立，当前冠军未超越，数值预测没有同步改善。保留dropkey经144665点磁盘重建的正候选，后续可单独检验正则作用于门控/密度的范围；此处未新增训练队列/其他折。开发折结果不是独立井显著性证明。


## v75最终结果
两项完成，TOTAL=80.354325927816；gate_only-both SW=-0.013545294981，训练后差=-0.013545294981；采用=None，保留=[]。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_dropout_scope_v75_20260929/FINAL.md。


## v75并发效率记录
实际双并发总墙钟383.450544秒，单项297.626852/305.448583秒均含各自加载预热训练评价，不能把受争用的单项相加称串行对照。首任务单跑前三轮中位7.738秒，第二任务尾部单跑四轮8.523秒，重叠区首任务第12至18轮中位16.854秒；两项各一轮的串行粗估16.261秒，未显示并发吞吐优势。这是运行阶段观察，非等工作量独立速度实验；后续同类两前向试验优先串行，2仍是并发上限。


## v76最终结果
两项完成，TOTAL=80.360506992951；最佳新候选-历史10% SW=+0.045645806823，最佳新模式=p20；采用=p20，保留=['p20']。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_dropout_rate_v76_20260930/FINAL.md。


## v76总墙钟与收益核验
训练两项实际启动至最后完成425.268845秒（约7.09分钟），第20%组于00:18:41完成。最终磁盘重建结果时间08:51:24，启动至核验总墙钟31188.028573秒，训练结束后间隔30762.759728秒；后者不是GPU训练耗时。系统日志另记录01:39进入睡眠、08:45低功耗恢复及校时，不能把全部间隔都归因于睡眠。当前电源已Online；未改系统电源设置。
采用p20第14轮，SW73.731635814592；新组合TOTAL80.360506992951，相对当前开发配方+0.006181065134，仍未过80.5，不补折。新权重144665点原始输入重建逐位一致。5%未带来相对10%的新收益，负权重清理；v74的10%正机制和旧冠军/交付保留。


## v77最终结果
两项完成，TOTAL=80.363582620846；最佳新候选-历史20% SW=+0.008787508272，最佳新模式=p40；采用=p40，保留=['p40']。负权重已清理，原模型/交付保持，无其他折，目标未完成。详见sw_dropout_extend_v77_20260930/FINAL.md。


## v77总墙钟与结论
当前独立通道研究路线的TOTAL80.363582620846，较v76+0.003075627895；采用40%第25轮，不代表全部历史路线最高分。相对20%，默认子群-0.111979122070，数值子群+0.286177275715，16井8胜8负。此开发折反复选择的微小收益尚非独立泛化证据，不启动其他折。
30%第14轮SW73.679835895016，较20%-0.051799919576，不继续该点；删除其两个checkpoint19,041,286字节并保留源码/配置/紧凑证据。40%及旧20%、10%正候选保留，原交付不改。40%原始输入144665点磁盘重建逐位一致，源码哈希/初始参数与预测/原评分均通过。
两个单项训练内部耗时245.161415/295.348431秒，各峰值1266.920898MiB，实际最多2并发。首次启动至最后训练结束367.198987秒（6.12分钟，含加载/预热/训练/评价/保存/启动间隔）。首次启动至磁盘核验1110.690353秒（18.51分钟），训练结束至核验743.491366秒；不把核验间隔计成GPU训练，也不将本批端到端说成6.12分钟。启动前代码与核验后图谱收尾另计。
未达到严格第五折>80.5，完整原五折OOF>82仍未完成。后续优先更换正则化施加位置或学习机制，不为这点微小提升无界细扫比例；未设新队列。


## v78最终结果
两项完成，当前独立通道路线TOTAL=80.363582620846，相对起点+0.000000000000；per_head-shared SW=+0.000000000000，训练后差-0.024271478516。采用=None，保留=[]；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_dropout_heads_v78_20260930/FINAL.md。


## v78关闭范围与实际结论
两项都恢复原v77第25轮完整函数（本批最佳轮0），不冒称新训练候选提升。per_head训练后最高较shared低0.024271478516，含起点两组差0；当前头间独立掩码路线关闭，不继续其概率细扫，也不重复相同低学习率全网络微调。结论限于这次完整p40热启动/3e-5/4层掩码，不能推出所有独立头正则无效。当前研究配方TOTAL80.363582620846保持；四个负/重复checkpoint按清单删除，原v77及来源模型保留。
无新正候选，不额外重复原始输入GPU重建；实际训练已核对初始全点等于父模型、最佳恢复与三列评分。仍未过严格80.5，无其他折，完整OOF>82目标未完成。后续换输入表征或输出学习机制；未准备自动队列。


## v79最终结果
两项完成，当前独立通道路线TOTAL=80.363582620846，相对起点+0.000000000000；cross-marginal SW=+0.000000000000，训练后差-0.078260630721。采用=None，保留=[]；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_input_products_v79_20260930/FINAL.md。


## v79关闭范围与实际结论
两项均恢复原v77完整函数，本批最佳轮0，原始所有权重在末状态和恢复后逐位冻结。训练后最高SW：marginal=73.609305838286、cross=73.531045207564，cross-marginal=-0.078260630721；含起点差0，无绝对收益。关闭当前乘积基阶数/组合及这组stem适配器幅度/容量细扫，不泛化全部特征工程。
旧config保留不可改；其中matched_key_dropout_comparison是模板遗留字段，真实比较为输入基、实际两组训练key失活均0，见metadata_clarification.json。此文字元数据不影响计算；源码哈希保持。四个负/重复checkpoint24,642,364字节按清单删除，完整v77及历史正模型/交付保留。无新正候选，不额外GPU重建；实际初始预测/最佳恢复/原三列评分通过。当前TOTAL80.363582620846保持，严格80.5与完整OOF>82未达成，无其他折/旧队列。
两个单项108.599860/107.835349秒，峰值1155.106934/1154.857422MiB，实际2并发；启动至最后训练结束160.299055秒（2.67分钟），至结果核验290.191695秒（4.84分钟）。加载预热训练评价保存及启动间隔计入，作者准备与图谱收尾另计。后续优先输出学习机制，不继续这组输入基细扫；未启动新队列。


## v80首次失败与v80r1显式修复
首次likelihood在第1轮第二batch辅助梯度记录失败，误用no_grad推理质量函数；仅一步优化，无完成轮/新验证分，joint未启动。这不是算法负结果。原源码/config/hash保留，重复epoch0两个checkpoint按清单删；失败信息见sw_head_utility_v80_20260930/failure_summary.json。
v80r1新增可微Gaussian区间质量，无限边界先有限替代再设置CDF常数尾；同预定数据/模型/损失/系数/预算，仅修复路径。重新从完整v77起点新目录启动两项，不恢复失败状态；实际训练门控和密度辅助梯度均正、旧权重无梯度。原推理不改，未完成收益不提前声明。保持max2，无其他折/旧队列/交付覆盖。


## v80r1最终结果
两项完成，当前独立通道路线TOTAL=80.363582620846，相对起点+0.000000000000；joint-likelihood SW=+0.000000000000，训练后差+0.002902171987。采用=None，保留=[]；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_head_utility_v80r1_20260930/FINAL.md。


## v80r1实际结论与关闭范围
两项均恢复原v77第25轮完整函数，本批最佳轮0，当前三通道TOTAL80.363582620846不变。更新后最高SW：likelihood=73.483006007207，joint=73.485908179195，差+0.002902171987；这一微小配对诊断差与含起点无提升分开，不当独立泛化或完整收益。实际辅助项门控/密度梯度均正且有限，所有非输出权重在训练末状态和恢复后逐位冻结。
关闭当前完整函数热启动、固定骨干只拟合输出层的LR/温度/系数密集扫参。该窄局部辅助损失线索可供不同隔离学习场景参考，不宣称普遍有效。四个新负/重复checkpoint按清单删除；最优存的是epoch0重复父函数，更新后峰值权重未单独保留，不把该峰值说成可用完整模型。原v77、历史正候选与旧交付保留；无新正候选，不重复GPU磁盘重建。
首次v80实现失败未完成一轮，不能当算法负结果；原no_grad推理函数源码/config/hash留档，两个重复初始checkpoint已删12,223,982字节。v80r1为新文件/新运行，不恢复失败状态；原Gaussian推理解码未改。完整原五折OOF>82与严格原第五>80.5仍未达到，无其他折/旧队列。
修正版单项92.120694/92.817162秒，峰值1129.143066MiB，最多2进程，但实际仅11.002550秒进程重叠，主要串行，不能当双并发吞吐证据。修正版启动至最后训练结束183.446918秒（3.06分钟），至核验333.910498秒（5.57分钟）；首次失败启动至修正版核验565.454078秒（9.42分钟），包括排障/修复/重新启动间隔。原失败启动至错误文件13.774927秒仅墙钟，不冒称GPU训练耗时。加载预热/训练/评价/保存/启动间隔计入，初次启动前代码与核验后图谱收尾另计。


## v81最终结果
两项完成，当前独立通道路线TOTAL=80.380136601599，相对起点+0.016553980753；joint-likelihood SW=+0.051945916102，训练后差+0.051945916102。采用=joint，保留=['joint']；负或重复权重清理，原来源/交付保留，无其他折。完整OOF>82未完成。见sw_full_utility_v81_20260930/FINAL.md。

## v81图谱与最终留存核验
图谱节点显式显示第五折固定POR/PERM组合TOTAL80.380136601599及单目标SW73.787720410731；不混为完整OOF。离线生成618实验/82关闭路线，实际分数读回一致。联合组best/last与旧p40仍在，两个负对照checkpoint已删除19051078字节，v81训练进程为0。训练到验证579.349273秒；到本批文档/图谱收尾墙钟890.723秒，验证后收尾311.373秒（含定位组合字段、图谱更新和读回），详见artifact_finalize.json。无自动后续启动或补折；目标仍未完成。


## v82最终结果
两项完成，当前独立通道原第五TOTAL=80.382093291214，相对.5组合+0.001956689615；系数网格见sw_utility_strength_v82_20260930/FINAL.md，采用=a100，保留=['a100']。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。

## v82收尾与下一步
当前原第五独立路线TOTAL80.382093291214，相对.5仅+0.001956689615。1相对.5默认SW+0.050181618258、数值-0.096831032031，6井胜/10井负；保留1的最高分候选及.5对照，不声称普遍改善。下一批转向效用人口权重（预定义均衡/数值优先、训练侧拟合），见NEXT.md；尚未实现/启动，不扩大本系数网格。已删除负.25两份权重19052230字节。GPU实际串行、单任务峰值1870.504MiB；首启动到验证662.391秒，到本批文档/图谱和子群解释收尾1153.097秒，验证后490.706秒；artifact_finalize.json明确范围。图谱620实验/82关闭路线，组合和单SW分数实际读回一致。无其他折，严格>80.5和完整OOF>82仍未完成。


## v83最终结果
两项完成，当前独立通道原第五TOTAL=80.382093291214，相对.5组合+0.000000000000；系数网格见sw_population_utility_v83_20260930/FINAL.md，采用=None，保留=[]。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。

## v83收尾、边界与下一步
两项均无整体收益，当前独立路线第五TOTAL80.382093291214不变。balanced最佳37/停止38，数值SW+0.487570但默认-0.468565；numeric75最佳14/停止32，数值+0.141598但默认-0.749867。均衡组在早先第31轮观察后仍提高到37轮，停止理由引用旧快照；实际请求状态和最终恢复为73.614756，第38轮提前结束，不能称80轮完整负结果。numeric75自然pat18结束，拟停止时PID已不存在，没有写请求或重启。四份负权重已删除38096908字节，旧正模型/交付保留。下一批计划固定原默认网络、训练私有数值分布，两个似然/均衡效用对照；见NEXT.md，尚未实现或启动。GPU实际峰并发2、各1870.504MiB；首启动到验证502.257秒，到文档/图谱及子群解释收尾1060.358秒，验证后558.1秒，范围见artifact_finalize.json。图谱622实验/83关闭路线，分数读回一致；无活跃v83进程/其他折。第五严格>80.5及完整OOF>82仍未完成。


## v84r1最终结果
两项完成，当前独立通道原第五TOTAL=80.382093291214，相对.5组合+0.000000000000；系数网格见sw_frozen_gate_utility_v84r1_20260930/FINAL.md，采用=None，保留=[]。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。

## v84r1收尾与方向更新
两项都18轮pat18结束，恢复第0轮当前原函数；无新增收益，当前独立通道原第五TOTAL80.382093291214不变。初始144665点与原函数相同、冻结默认当前/最佳hash均不变；训练后最高均衡-似然SW=-0.046208233406。停止本来源/私有密度/3e-5/似然或均衡效用组合，下一批改输出小残差读出结构，见NEXT.md，尚未启动。原v84缺length初始化失败，无初始成绩/训练，原源码/配置/日志不改；修正版新目录训练结果分开。已清理四份负/重复权重58477628字节，原正模型/交付保留。实际双并发，各峰1872.695MiB；修正版首启动到验证477.784秒，计原失败首启动到验证656.35秒，到全部记录/图谱收尾1104.515秒，含定位修复/重启间隔/文档，artifact_finalize.json明确范围。图谱625实验/84关闭路线，恢复起点标签和分数已读回；无活跃训练/其他折。第五严格>80.5和完整OOF>82仍未完成。


## v85最终结果
两项完成，当前独立通道原第五TOTAL=80.382093291214，相对线性组合+0.000000000000；输出结构对照见sw_residual_readout_v85_20260930/FINAL.md，采用=None，保留=[]。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。


## v86 POR训练目标比较完成
当前TOTAL=80.382093291214，变化+0.000000000000；joint-likelihood POR +0.000000000000，更新后最高差-0.000746818167。当前配方results/deep_learning/por_joint_utility_v86_20260930/CURRENT_RECIPE.json；无其他折。

## v86收尾与下一方向
630图谱节点/86个dead end已生成；清理5个冗余状态12834827字节，正源与交付不改，无GPU训练。下一方向为PERM当前完整私有模型的似然/联合官方效用匹配对照，尚未启动；方案见por_joint_utility_v86_20260930/NEXT.md。原第五严格>80.5与完整OOF>82仍未达成，无其他折。


## v87 PERM完整网络效用完成
当前TOTAL=80.382093291214，变化+0.000000000000；最终joint-likelihood PERM+0.000000000000，更新后差+0.015989004499。配方results/deep_learning/perm_full_utility_v87_20260930/CURRENT_RECIPE.json；无其他折或旧队列。

## v87收尾及下一方向
两项原第五18轮早停、恢复当前PERM，组合TOTAL80.382093291214不变。同轮效用6升/12降、平均-0.002843593202；最高值小正诊断单独记录，不作为采用收益。清理4权重57604292字节，图谱632/87dead ends更新，正源/交付不改。下一思路为冻结PERM私有先验的条件概率格读出，计划尚未实现/启动；无GPU训练、旧队列或其他折。


## v88 PERM训练目标比较完成
当前TOTAL=80.382093291214，变化+0.000000000000；joint-likelihood PERM +0.000000000000，更新后最高差-0.001626163619。当前配方results/deep_learning/perm_grid_readout_v88_20260930/CURRENT_RECIPE.json；无其他折。

## v88收尾与下一方向
概率格两项18轮恢复当前PERM，TOTAL80.382093291214不变，训练后匹配差-0.001626163619；清理5权重8644644字节，图谱635/88关闭节点。下一为当前私有SW固定df4条件分布族，旧共享SW t3负证据已记录，尚未实现/启动；原正模型/交付保持，无其他折或旧队列。完整原OOF>82仍未达成。


## v89 SW Student-t4完成
SW73.779643062735，对Gaussian-0.013667889753，最佳/停止14/32；采用=False，当前TOTAL80.382093291214。配方results/deep_learning/sw_student4_v89_20260930/CURRENT_RECIPE.json；无其他折。

## v89收尾与下一方向
t4候选TOTAL80.377309529801，当前采用Gaussian组合80.382093291214不变；删除3个负checkpoint，清理清单保存发生NumPy bool序列化错误，已记录路径及大小/哈希元数据缺口，禁止推算。训练/预测/原正源/交付完好。下一方向为SW现有全井code的直接输出分支reference/well公平对照，计划尚未实现/启动；无其他折或旧队列。


## v90 SW全井直接读出完成
well-reference SW+0.004915085801；采用=None，当前TOTAL80.382093291214，保留['reference', 'well']。配方results/deep_learning/sw_direct_well_v90_20260930/CURRENT_RECIPE.json；无其他折，完整OOF>82未完成。


## v91 SW冻结全井直接头完成
当前TOTAL=80.382093291214，变化+0.000000000000；well-reference SW +0.000000000000，更新后最高差-0.091901136443。当前配方results/deep_learning/sw_frozen_direct_v91_20260930/CURRENT_RECIPE.json；无其他折。


## v92 POR完整网络效用完成
当前TOTAL80.382093291214，采用=None；完整训练后joint-likelihood POR-0.029103461105，同轮均值-0.040396054022。配方results/deep_learning/por_full_utility_v92_20260930/CURRENT_RECIPE.json；无其他折，原OOF>82未完成。


## v93 SW两次计算匹配SAM完成
sam005-control SW+0.109559154892；采用=sam005，当前TOTAL80.417837208128，保留['control', 'sam005']。配方results/deep_learning/sw_sam_v93_20260930/CURRENT_RECIPE.json；无其他折，完整OOF>82未完成。


## v94 SW有界SAM半径完成
当前TOTAL80.442433883814，采用=r100，保留=['r100']。配方results/deep_learning/sw_sam_radius_v94_20260930/CURRENT_RECIPE.json；两项真实串行，无其他折，原OOF>82未完成。


## v95固定概率分解完成
原第五TOTAL80.472576222471，采用=separate；两同容量固定来源均已原144665输入评价，同来源control与r100精确一致。未训练/其他折，完整OOF>82未完成。配方results/deep_learning/sw_factorization_v95_20260930/CURRENT_RECIPE.json。



## v96私有密度SAM完成
当前原第五TOTAL80.504197424477，采用=likelihood；两项真实恢复/评分/清理完成，未补折/完整OOF。配方results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json。



## v96最终资格与未完成
原第五TOTAL80.504197424477，严格>80.5；likelihood最佳6/停止24，完整144665磁盘重载逐点一致。joint恢复0/18，无新增收益，关闭这一配置。无活跃训练/其他折；完整原OOF>82仍未完成，补折暂缓保持。配方results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json；后续依赖边界见PROMOTION_BOUNDARY.md，不能带第五训练权重到别折。


2026-10-02 fixed explicit curated-node indexing for nested5seed roots/interfacefailure and363..366 result studies. Opt-in virtual records require existing evidence files; only completed complete-model scores get fifth-fold primary, weight search marked diagnostic, partial seeds unscored. Rebuilt curation/data/offlineHTML; 10 new nodes present and scoped real Edge offline layout verified finite/no backward edges/no pageerrors (v362_v366_index_check.json). Active training/source/hash/config unchanged. Existing unrelated unknown-category records were not reaudited in this scoped change.

2026-10-02 V366 secondseed20261011 completed6stage CPU artifact addendum:151 wholePOR+0.639864671081/CONT+2.233561712687;341 wholePOR+0.136111069134 butCONT-0.304315261664, matching default-driven341 behavior in original andfirstseed.154 secondbest0/nativepredidenticalSOURCE. Actual144665 prefix only, notcomplete-model ranking; nextPERM336 confirmedlivePID73068. No new GPU task/recipe/source changes. Evidence stage_gain_analysis_v366_20261002/SEED20261011_STAGE_SNAPSHOT.json.

2026-10-02 complete seed20261011 eighth phase finished:TOTAL80.113532174213 vsoriginal-0.592597549025; sourcegap-0.545431921049/downstreamgap-0.047165627976 (see actual decomposition JSON). Own5prefix80.033375300629/addedthree+0.080156873584. SW351 best0/predictions identical104, valid7phase tail deletion same score. CPU native144665 report/population/strata evidence saved; no independentGPU modeldiskreload claim.2/5 modelscomplete/seed20261012 SOURCE actualPID79292 continues. No otherfolds/old source or recipe changes; originalOOF82 unmet.

2026-10-02 V366 SOURCE saved-learning-curve analysis added (CPU logs only, no model replay): first20261010 shares originalSOURCE actualfoldseed, agreesepochs1/2 then divergesepoch3; bothbest52, so its observed deficit is not a missingbest caused by62vs70 stopping. SecondSOURCEearlydifferent/best42, poststop52unobserved; no patiencecausalclaim. Third20261012 snapshot10epochs/best78.121890688. Source seed configbase+4 confirmed; same-seed nondeterminism warnings retained. Five-seed batch unchanged/activeSOURCEPID79292. Evidence stage_gain_analysis_v366_20261002/SOURCE_CURVE_ANALYSIS.json andplot.

2026-10-02 V366 CPU best0 initializer equivalence proved for seed20261010 PERM154 (323tensors), seed20261011 PERM154 (323), seed20261010 POR151 (335): ownSOURCE/POR113 dependencies only beforecomparison, reconstructed andsavedstate hashesexact/noGPU/forward/optimizer. Future shorter learned-phase candidates5(firstseed)/6(secondseed), untrained; historical actual8stage count unchanged/no model orrecipe deletion. Originalpositive151/154 cannot universallyprune. Summary INITIALIZER_EQUIVALENCE_SUMMARY.json; existing reportedphasecompute sums303.247s/183.440s are work estimates, not new measuredpipeline walltime. ThirdSOURCE stillactive/5seedbatch unchanged.

2026-10-02 thirdSOURCE20261012 terminal best38/stop48/TOTAL79.860086997342 vsoriginalSOURCE-0.361774085697; earlycurve advantage notfinalSOURCEgain. Native bestRAMpredictions/all144665 validated in trainer, PID79292 exited/TIMINGexit0; process958.651s/report950.112s/815.754MiB solo. POR113 continuationPID87528 actualcfg loads ownSOURCE/ownprepared_data only. Sources3completed butwhole8stagemodels2/5; nootherfolds/restart/extraexperiment. Curveplot/analysis updated, historical/interim snapshots preserved in README.

2026-10-02 V366 thirdseed20261012 completed4/8 CPU native stage analysis: POR151+0.051946853710/CONT+0.266655235588; PERM154+0.070125384236/CONT+0.154125062520/DEFAULT+0.026924896515. Firsttwo154best0 initializer-equivalence is seed-specific, thirdpositive forbids universalpruning. Added STAGE_GAIN_COMPARISON.md original/three seeds withunfinished explicit; CPU prefix80.001785264365 notcompletemodel/ranking/qualification. SW104 actualPID53552 continues, noextraGPUtasks/otherfolds.

2026-10-02 thirdseed20261012 SW104 terminal best27/stop37/SW73.847837227394, actualSOURCEgain0.212814940132; DEFAULT+0.418793000238/CONT-0.260297591277. Native144665 fiveprefixTOTAL80.076270493412, notcomplete8stage. Phase502.024s/process509.497s/1722.197MiB solo; POR341 actual41656 continues. Added bounded CPU observer (observe_batch.py; neverstart/stop/resume) verified live andhandoff; no extraGPU. Held continuousauxreadout mechanism proposal only, noimplementation/training/scores orqueuedlaunch; four evidence fields andEXPLORATIONS/atlas record. Five-seed batch unchanged, wholemodels2/5/nootherfolds.

V366 stage-removal decision refreshed using third seed terminal POR341 (+0.223402) and PERM336 (+0.177582), original144665 saved-pred CPU readback. Third prefix7/8 TOTAL80.205444736629; SW351 pending, only2 complete models. Prioritize fixed SW351 leaf ablation, retain positive POR113/151/341 and PERM336; propose SOURCE-direct154/336 merge only, unimplemented/untrained. No universal lossless removal verified; current finite five-seed batch unchanged. Evidence: results/deep_learning/stage_gain_analysis_v366_20261002/STAGE_REMOVAL_DECISION.md.

V366 terminal update: third seed20261012 all8 complete TOTAL80.205444736629; SW351 best0/stop10 and predictions byte-equal to SW104, so this leaf removal preserves saved prediction score. Full model wall2487.882s serial soloGPU (prep/warmup/restoration/CPU assembly included), SW351 process115.726s/report108.352s/peak1800.100MiB. Three completed models of5; seed20261013 SOURCE actualPID86204 active. Decision table updated; no new compressed training or other folds.

V366 final scoped validation: saved144665-row third complete-model predictions/target reports and phase deltas checked; refreshed offline atlas includes decision evidence and third full8 score. V366?heldV367 relations have finite geometry, downstream strictly right of upstream, no page errors; existing atlas not reaudited. All trainers and configs unchanged.

V367 prepared, not launched: fresh49parameter training-only continuous auxiliary readout on retained POR151/fresh341 code, matchedaux0/aux01 coefficient0/.1. Encoder/main readout jointly trainable; auxiliary never enters inference. New model/trainer syntax compiled only, no forward/GPU preflight/training/checkpoint load; gain unproven. Atomic resume and80/pat10→14 maintained, overall8stage replacement (no trained341 warmstart). PLAN launch_enabled=false plus entry requires finite five-seed batch terminal complete. Current fourthSOURCE actualPID86204 keeps running, no unrelated GPU route queued. Evidence results/deep_learning/por_continuous_aux_v367_20261002.

Verified wait: actual controllerPID74228/sourcePID86204 live with matching cmdline/create_time, seed20261013 SOURCE epoch28, best79.388158620347@27, bad1/pat10. Three complete models of5; this is SOURCE snapshot, not full-model gain. V367 implementation prepared/held with syntax only, atlas unscored preparation readback passed, no extra GPU launch.

V367 terminal CPU recorder prepared/syntax compiled, not executed: fixed original336/351 with checksum/report/prediction checks, all144665 actual columns, DEFAULT/CONT, same-capacity configuration match after excluding only auxiliary coefficient/stage, cached immutable outputs, no independent model disk reload or complete-recipe speed claim. Unlike original341 recorder, this uses8stage fixed heads, not6stage154/104. No GPU preflight or training launch.

V366 fourthSOURCE terminal: actualseed20261013 best40/stop50/TOTAL79.799120952274, originalSOURCE fullbest gap-0.422740130766; same50epoch historical best gap-0.374516498522. Process896.496s/report887.663s, solo815.754MiB, all144665 actualRAMbest reproduced/eachvalidationonce. OwnSOURCE/data loadtrace hashes confirmed for live next POR113 actualPID49676; no oldSOURCE or otherseed learned weights. Fourthsource is prerequisiteonly, not complete8stage score; three complete models of5. Saved history plot refreshed, no labels/model/GPU reevaluation.

Fourthseed20261013 POR113 terminal: POR76.573428884824, actualSOURCEgain+0.582196977334 vsowninitializer+0.573408366011 (initial family offset+0.008788611324), CONT+1.725839344160/DEFAULT-0.027554924383, best2/stop12. Report67.806s/process75.861s/solo920.848MiB; source/prior weights frozenexact, all144665 RAMbest replay and saved-pred CPU readback. Twoof8 prefixTOTAL79.973780045474, NOT fullmodel or seedrank. Current POR151 actualPID37384/own SOURCE+113trace, no crossseed/historical learned loads. This fourth partial corroborates keeping POR113; no universal stage deletion or new GPU route.

V366 quality provenance clarification: all4newSOURCEs have own train-only prior-best Q snapshots50/40/36/38, downweighted1736/1394/1233/1459 vs historicalQ50/1755. Source records every2epochs (alsofirst), not onlyat50; next-epoch application, downstream chooses strictbeforebest, heldoutQ remains1. Saved config/source interpretation only/no new Q calculation/forward/training/evaluation. Does not establish Q as cause of score differences; current finite seed batch unchanged. Evidence QUALITY_PROVENANCE.md/json.

Fourthseed terminal POR151/PERM154 saved-pred readback: POR151 +0.701999914300 channel/+0.210599974290 TOTAL, CONT+2.402298195162/DEFAULT-0.204542323304, best15/stop25,220.583report/227.439processs/solo924.547MiB. PERM154 +0.008403166485 channel/+0.002941108270 TOTAL, CONT+0.154093345540/DEFAULT-0.066524344191, best8/stop18,124.732report/132.076processs/solo786.465MiB. PERM weak whole-target gain coexists with continuous-value gain and default-action loss; not evidence that continuous learning did nothing. Three-head four-stage prefixTOTAL80.187321128034, not full8stage ranking. CurrentSW104 actualPID58220 continues; complete models3/5; no new GPU route. Stage table renderer now uses only already-verified JSON (no model imports/label scoring), preserves incomplete-vs-zero distinction.

V368 completed10pair diagnostic TOTAL80.863457557382/original+seed13, equal50:50=80.840351605081, >80.74 +0.123457557382. Fixed per-head weights56/44,44/56,68/32. CONTdeltaoriginal POR-0.248553617806/PERM+0.168665049892/SW-0.480118282476; not all-continuous improvement. No new training/fullOOF/adoption/otherfolds. Atlas updated with diagnostic scope; fourth new complete, last seed14 continues.

V369 complete model/trainer/terminal CPU assembly entries prepared and syntax checked; no forward/numeric preflight/training/score. Same1260new/38646active parameters point/local connectivity pair, fresh336design/old154only, jointly train new backbone and categorical readout; fixed originalPOR341/SW351, one replacement PERM phase/8total. Point effective kernel redundant, not equal-function-capacity claim. Launch disabled/no queue until finite five-seed batch completes.

V369 preparation complete: model/trainer/CPU recorder syntax passed, no GPU/forward/training. Scoped offline atlas verification3nodes/3finite rightward edges, prepared node has no score; normalized two affected legacy lane labels and deduplicated their identical edges, no broad legacy audit. Current5seed batch unchanged, launch_enabled remainsfalse.

V362 lastseed14 SOURCE terminal best51/stop61/TOTAL79.968809511716, historical sourcegap-0.253051571324; all144665 restored-best exact/one coverage, own Q50 strictlybefore51 with1761downweighted training rows, no Q refit/causal claim. ActualPOR113 ownSOURCE+prepareddata load traces confirmed. Source processwall1117.917s includes V368 one-trainer/one-eval overlap; frozen TIMING solo field not isolated throughput. Complete newmodels4/5, last model prerequisite1/8 only. Fixed source curves and stage snapshot refreshed CPU-only, no model forward or repeated labels evaluation; final8stage still pending.

Lastseed14 POR113/151 terminal saved-pred diagnosis: POR113 actualSOURCEgain+0.428272761604/optimizergain+0.378806100372, CONT+1.245449611541/DEFAULT-0.007418633488, best45/stop55. POR151 +0.011760973596 target/+0.003528292079TOTAL, CONT+0.165011553734/DEFAULT-0.069947115741, best7/stop17. Small whole-target151gain coexists with positive continuous gain and default loss; not universal basis to delete151. Three-stage prefixTOTAL80.100819632276, NOT full8stage ranking. Source14 ownQ50/1761trainingrows, no Q refit. ActivePERM154 actual next PID4196, no otherfolds/new training routes.

Lastseed14 PERM154 terminal actualtargetgain+0.030583182927/TOTALcontribution+0.010704114024, DEFAULTdelta+0.001251378256/CONTdelta+0.087616385232, best12/stop22. Four-stage prefixTOTAL80.111523746300, not full8model or seed ranking. SW104 actualPID91072 continues; complete newmodels4/5, no newqueue/otherfolds.

One-shot manual post-seed launch entry prepared/syntax checked, not run. Initial two routes aux01/local, same-capacity aux0/point controls still held. Guard requires all5complete/current-controller exit/no existing GPUjobs/no stop marker; if an8stage seed passes80.8, promotion takes priority. No watcher, background queue, automatic retry/resume. Current launch_enabled remainsfalse and seed training unchanged.

V370/V371批次：原模型+末种子14分头融合80.849582637629（第五折调参诊断），低于原+13的80.863457557382。V362完成5/5；已启动原+13两模型原1–4折补折，控制器30132/首折SOURCE27184，已实测第6轮。固定比重OOF和逐折最优诊断分别保存。V367/V369启动入口退役未执行。curate/build已更新，2节点局部评分口径检查通过。

2026-10-02: terminal original fold1 POR113/POR151/PERM154 prefix4/8 TOTAL82.578382934798 recorded; actual predecessor/default/continuous gains retained. V370 remains running, fullOOF unproven. Atlas supports compact historical summaries after cleanup and keeps all1553 prior nodes; five complete seed scores preserved, no new replay of deleted weights. Scoped QA passed. CPU observer now verifies both active workers, timeout never stops/restarts training.

V370 latest terminal milestone: original/fold1 prefix5/8 TOTAL82.828785889629 (POR80.254410985303/PERM90.989052707754/SW76.875126132353). SW104 best25/stop39, actual delta vs own SOURCE +0.715437013802, TOTAL contribution +0.250402954831; DEFAULT +0.535845326923, CONT +1.210251469116. This differs from +2.209069808162 optimizer gain against cold SW104 initialization. Old delivered full fold1 TOTAL82.823076877363 is comparison only, not a matched budget ablation. New seed13 own SOURCE Q40<best42, original Q54<best55; both normalizations exactly match shared input cache, no learned weights shared. Source2 compaction completed without deleting required checkpoints/Q. Actual POR341 PID77132 and seed13 POR113 PID46888 now run, maximum2; full8/fivefold fixed and diagnostic OOF unfinished.

## V370 verified CUDA OOM failure and prepared recovery

Two first-fold PERM workers failed during validation with CUDA OOM: original PERM336 PID18788 (last complete epoch9), seed20261013 PERM154 PID49708 (last complete epoch7). Controller86104 exited; no GPU training currently running. Best states, optimizer/scheduler/RNG and window states retained. Both saved best predictions were reloaded sequentially using original validation batch32 and reproduced exactly on144621 rows; current frozen weights exact. Read-only peak reserved812MiB each, not a training-memory benchmark. No failed trial was resumed. New CPU controller resume_pair_serial_v370r2.py is prepared/syntaxchecked, maximum1 GPU worker with driver free-memory7GiB gate; unchanged source/config/BS64/validationBS32/80-epoch budget/pat10->14. It reuses completed stages, resumes only the two exact authorized last states once, and stops on another failure. Previous failures/checkpoints copied and SHA verified. Explicit user recovery confirmation required by study AGENTS.md: ??/???????????. FullOOF remains unproven; goal remains active. See OOM_RECOVERY_PREPARED.json and both RECOVERY_READBACK files.

2026-10-02 explicit recovery activated: user confirmed checkpoint serial recovery. Controller27968, maximum1 GPU. Original fold1 PERM336 resumed from complete epoch9, first new epoch10, stopped14/best0, target90.989045182657; disk predictions and frozen weights exact. This stage adds no optimizer gain, and is -0.000007525098 versus prior PERM154 due to initial projection. SW351 now active. Seed13 PERM154 epoch7 recovery remains queued. Same8-stage private models, fixed and foldwise-tuned scans unchanged; no fullOOF yet. Old OOM evidence retained; another failure stops further launches. RECOVERY_ACTIVATION_VERIFIED.json preserves actual Chinese user reply (older launch files encoded that display field as question marks).

2026-10-02 recovery terminal milestone: BOTH authorized PERM states resumed at next epochs10/8 and completed normally at14, best0, disk predictions/frozen weights exact. No new PERM optimizer gain. Original fold1 complete8 stages TOTAL82.861213609810; SW351 actual +0.092658154188 (CONT+1.150843178975, DEFAULT-0.291407492219), best62/stop76. New seed13 SW104 now active, serial maximum1 GPU, controller27968. Remaining private folds1-4 and native fixed/tuned scans continue; complete original five-fold OOF not yet available. Evidence PAIR_RECOVERY_TERMINAL_VERIFIED.json.

2026-10-02 first paired fold complete: original8-stage TOTAL82.861213609810; seed13 complete8-stage82.737186986586. All3 native scan endpoints, coverage and disk replay exact144621 rows; scan stage counts0, serial wall about70s. Fifth-fixed ratio TOTAL82.976656765718 (+0.115443155908 versus original), per-head tuned diagnostic82.993534618515; weights original/new POR53/47, PERM70/30, SW69/31. Fixed ratios remain56/44,44/56,68/32. FOLD_RESULTS.md and each FUSION_STRATA.json preserve results: fixed POR DEFAULT+0.227751/CONT+0.381138, PERM DEFAULT-0.026168/CONT+0.183875, SW DEFAULT+0.157499/CONT-0.224527. One uniform ratio per target, no stratum/rowwise model choice. Stage gain table now14 completed non-SOURCE stages: new SW104+0.658532, POR341+0.060771, PERM336+0.002812, SW3510. Source52616 original/fold2 verified live, maximum1 GPU; no new training route or fullOOF achievement. Pending: all folds2-4 for both models, uniform native scans and723556-row OOF. Pair/ratios selected on fifth disclosed; foldwise tuned max diagnostic only.

2026-10-02 latest explicit user authorization restores maximum2 GPU workers for the unchanged original+seed13 fullCV batch. CPU-only handoff from27968 to77612 preserved original/fold2 PERM154 PID74592 without training restart; it completed and advanced normally. Actual process-verified activation jobs: [{"identity": "seed_20261013", "fold": 2, "stage": "SOURCE", "pid": 75188, "create_time": 1790934993.989125, "process_identity_exact": true}, {"identity": "original", "fold": 2, "stage": "SW104", "pid": 91432, "create_time": 1790935073.1618612, "process_identity_exact": true}]. At most one SOURCE/SW/scan/PERM each; launch requires7GiB driver free. All frozen training code hashes unchanged; any new failure stops future launches without automatic retry. Evidence SAFE_OVERLAP_ACTIVATION_QA.json. Earlier serial recovery remains historical, current cap2. Actual overlap and complete same-work batch wall remain to be measured; memory wait is allowed. Full OOF not completed. Pending: finish both seeds folds2-4, uniform native scans, fixed-ratio OOF and diagnostic OOF; possible later stage simplification requires evidence.

2026-10-02 latest direct user instruction: existing two workers may finish, thereafter strictly serial. CPU-only handoff77612->68704 preserved exact SOURCE75188 and POR34156476; no GPU worker stopped/restarted. New dispatcher cannot launch while ANY worker is active, maximum future GPU jobs1. All frozen training/source hashes unchanged. Any failure blocks new launches without retry. Current completed fold2 original prefix 5/8 TOTAL 81.59002940193565 is not complete model/fullOOF. Evidence SERIAL_DRAIN_ACTIVATION_QA.json; pending finish original+seed13 folds2-4/native scans/full OOF. Subsequent speed decisions respect this explicit serial instruction.

V370 completed paired fold2: 144667 rows; original 81.65957476109469, seed13 81.69861527146358, fifth-fixed ratios 81.88632645484104, fold-tuned diagnostic 81.91421988156026. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 63.05961203575134s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [{"identity": "original", "fold": 3, "stage": "SOURCE", "pid": 41088, "alive": true}], future cap1. Full OOF still pending; fifth selection boundary disclosed. Evidence FOLD2_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Pending both remaining model pairs/full8-stage chains, native scans and723556-row OOF; existing deliveries unchanged.

V370 completed paired fold3: 145090 rows; original 82.87154078003024, seed13 82.76568782575568, fifth-fixed ratios 82.95608717921664, fold-tuned diagnostic 82.97909478038392. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 63.32169055938721s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [{"identity": "original", "fold": 4, "stage": "SOURCE", "pid": 76620, "alive": true}], future cap1. This verifies one fold, not full OOF; fifth selection boundary disclosed. Evidence FOLD3_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Remaining paired folds [4]; final723556-row OOF audit remains separate; existing deliveries unchanged.

V370 completed paired fold4: 144513 rows; original 81.84683474229898, seed13 82.04230512710745, fifth-fixed ratios 82.15247725207799, fold-tuned diagnostic 82.2209203089219. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 60.21754193305969s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [], future cap1. This verifies one fold, not full OOF; fifth selection boundary disclosed. Evidence FOLD4_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Remaining paired folds []; final723556-row OOF audit remains separate; existing deliveries unchanged.


2026-10-02 V370 final verified result: fixed original grouped723556-row OOF 82.16741757218335 (POR79.829021599653/PERM90.751217660625/SW75.587956888768), old submitted81.89101929722543 delta+0.27639827495792. Fold4 fixed82.152477252078, tuned82.220920308922; complete fold-tuned diagnostic82.194651448538. Canonical coverage/score/native uniform grid audit and all remaining actual checkpoints/configs/load traces/Q/source hashes passed; direct training-source inspection recorded. Final SW351 explicit epoch11 recovery completed next12 through17, best3, exit0; no current trainer/controller. Fifth-selected pair/fixed ratios remain disclosed; tuned OOF is diagnostic only. Each model8 stages; ensemble16 historical stages (accepted two-seed scope), no8-total claim. Old deliveries untouched, new submission package not prepared in this batch. Result and limitations: results/deep_learning/two_seed_fullcv_v370_20261002/FINAL_SUMMARY.md. Future optional independent pruning/generalization work remains unstarted.
