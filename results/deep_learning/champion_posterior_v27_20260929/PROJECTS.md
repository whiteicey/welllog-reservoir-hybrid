# v27已完成

目标：在80.30原模型上检验三个有效头的跨专家后验条件。

已完成：六项训练、当前基线一致性、冻结范围、最终恢复及必要的新目标磁盘重载和实际组合。

## v27基于80.30研究完成
六项完成，TOTAL=80.300730612039，相对父配方+0.000000000000。采用POR:None,PERM:None,SW:None。原权重/交付保持，无其他折；完整记录champion_posterior_v27_20260929/FINAL.md。

未完成：独立种子/数据划分和其他四折确认；不自动延伸本批负候选。

达成后：固定有效配方，再按用户授权讨论完整OOF；原交付保留。


## Atlas maintenance
Completed via curate_champion_posterior_v27.py and build_atlas.py. Original curate.py was not modified because the editing tool rejected it; the earlier proposed --v27-only hook was NOT installed. Existing annotations retained. No active training jobs at final check. See execution_notes.json.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
