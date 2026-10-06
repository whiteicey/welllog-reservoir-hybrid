# v22进度

目标：在保护80.274135原权重的条件下检验内部容量扩展。

已做：本轮重新核对本地v21；它已完成1/2/4三档全量微调，均回退第0轮。当前无运行训练。已确认本地v21同时扩容内部FFN和位置头，与前一轮附件中仅扩容FFN的包不同，以下使用已训练的本地实现作匹配对照。

计划：实现仅新增参数可训练的v22，显式启动ffn2/ffn4，检查初始函数、冻结范围、真实新增梯度及最终预测。

未做：v22真实训练结果、容量净收益、其他通道扩容或全OOF验证。

完成后：只有真实总分超过当前冠军才记录新开发配方；无收益则保留80.274135，不自动追加实验或补折。


## 2026-09-28T21:41:47.854539+08:00 v22 explicit launch
[{"mode": "ffn2", "pid": 50536, "started_at": "2026-09-28T21:41:47.842778+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_frozen_growth_v22.py", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_frozen_growth_v22_ffn2_fold5"]}, {"mode": "ffn4", "pid": 49624, "started_at": "2026-09-28T21:41:47.853151+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_frozen_growth_v22.py", "--mode", "ffn4", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_frozen_growth_v22_ffn4_fold5"]}]
Only the two predefined jobs; no queue, other folds or prior-source modifications.


## v22 固定原权重扩容实验已完成
2026-09-28T21:47:56.088940+08:00。2倍更新后TOTAL=80.269258294；4倍=80.263249587。均18轮停止/最佳第0轮，保留80.274135338。新通路梯度、旧参数冻结、训练后检查点全144665点磁盘重载通过。未运行其他目标/原其他折，不改交付；详见frozen_growth_v22_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
