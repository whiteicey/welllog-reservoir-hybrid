# 两种子补折

- 最新用户明确授权：V362 当前五种子训练结束后，沿用八阶段结构/损失/解码，补原模型与 seed20261013 的原始折1–4；第五折直接复用已验证模型。
- 每折每头扫描0:0.01:1的概率分布融合比例，一个头全折只用一个比例。保存逐折最优诊断和固定第五折比例OOF，禁止逐行/按真值类型选择模型。
- 原模型基础种子20261006，新模型基础种子20261009；各折实际种子=基础+零基折号，第五折分别20261010和20261013。每模型8阶段，双模型历史总计16阶段。
- 后续训练max80/min10/pat10，严格突破现有各头参考分后pat14。历史第五折原模型部分阶段的较旧耐心配置保留，不能声称五折调度完全一致。
- normal GPU、最多2任务；先串行，实测支持才增加并发。九曲线206维主输入、63维原幅值侧视图、原GroupKFold/行序/官方评分不变。每折预处理和幅值统计仅拟合64训练井，Q仅取本种子本折训练观测。
- SOURCE和各头均从本折来源训练，禁止旧18阶段头或其他种子权重；只复用同折种子无关的输入缓存，Q独立。
- V367/V369 暂停，新启动入口已退役；不覆盖任何旧交付。维护PROJECTS.md、README.md四项探索记录、EXPLORATIONS.md与DL图谱。
- 固定OOF的配对和权重曾使用第五折选择；逐折最优又在各折上调权，两者均须说明选择边界，后者不能作为无偏超过82的证据。
- 中止/失败不得自动重试或恢复；有限授权批次依赖启动可继续，用户STOP须立即阻止后续启动。

- 2026-10-02 用户明确确认“从断点串行恢复（推荐）”：仅恢复原模型第一折 PERM336 第9轮和 seed20261013 第一折 PERM154 第7轮完整状态，保持配方和批量，继续既定两种子补折及融合扫描；本批后续最多1个GPU任务。新失败仍须停止，不能把此次授权当作无限重试许可。

- 2026-10-02 用户随后明确“可以加入另外一个并发线程”，覆盖上一条后续最多1任务：本批可恢复最多2个GPU任务。只切换CPU调度，保留实际GPU进程及已有配方；同时至多1个SOURCE、1个SW、1个PERM和1个扫描，新增worker前driver空闲至少7GiB。遇新失败仍停止后续启动，不自动重试；记录实际重叠和同等工作量总墙钟。

## Latest user instruction: drain current overlap, then serial
Existing SOURCE75188 and POR34156476 may finish without restart. No new GPU worker may launch until all existing workers are terminal; thereafter maximum1. Preserve model recipe, frozen hashes, explicit failure-recovery boundary and native fixed/diagnostic OOF separation. Maintain PROJECTS after batches. Evidence SERIAL_DRAIN_AUTHORIZATION.json and SERIAL_DRAIN_ACTIVATION_QA.json.
