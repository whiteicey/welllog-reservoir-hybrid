# SOURCE与后续阶段Q的来源

## 探索重点

核对完整新模型为什么与历史模型并非只差一个随机种子，尤其确认较早停止的SOURCE是否仍有自己的训练侧Q。只读取已保存配置和冻结训练代码，不读取新的标签、不重新计算Q、不改变训练。

## 已探索的内容与证据

| 模型 | SOURCE最佳轮 | 后续固定Q的观察轮 | 降权训练点数 |
|---|---:|---:|---:|
| 历史模型 | 52 | 50 | 1755 |
| 新20261010 | 52 | 50 | 1736 |
| 新20261011 | 42 | 40 | 1394 |
| 新20261012 | 38 | 36 | 1233 |
| 新20261013 | 40 | 38 | 1459 |

来源：各POR151/config.json的quality字段，历史341/config.json，及SOURCE报告；机器记录见QUALITY_PROVENANCE.json。第四个151尚在训练，但配置在启动时已保存，不将其过程分数当最终收益。

冻结源码train_complete_seed_source_v362.py在第1轮及后续每个偶数轮观察全部有效训练点的质量冲突，新的训练权重从下一轮生效。不是到了第50轮才开始Q。fullcv_context_v97.bind_fold_quality选择严格早于SOURCE最佳轮的最后一次观察，并校验Q原始行号正好是该折训练行集合。没有观察才使用Q=1。

因此较早停止的第二、第三、第四SOURCE也有自己的Q40/Q36/Q38；“没到50轮，所以没有Q”不符合实现。各配置的Q文件SHA不同，后续阶段通过自己SOURCE/prepared_data.pt继承，不从其他种子或历史模型替换Q。记录中的“original training-only source Q”指原训练规则，不能理解为统一使用历史模型Q50。

## 实际收益与解释边界

这是来源澄清，没有新增模型、训练或收益。原与新模型除早停预算、来源权重外，各自Q的观察轮和内容也不同；完整配方的比较仍有效，但分数差不能只归因于随机种子、耐心或Q中的单项。没有同权重、只变Q的固定消融，不能声称Q差异解释了分数缺口。

下游全部使用训练侧生成的固定Q，验证行Q始终为1；SOURCE代码还断言验证原始行的监督权重不变。没有为本澄清运行额外GPU、forward、Q预检或数值评分。

## 不值得继续的点与待做

不修改当前五种子批次的Q；不把新模型统一换成历史Q50再称其为原配方完整种子对照。先完成五个完整模型，保留真实质量和来源记录。未来V367两组从同一原151父函数与同一训练侧Q开始，主要判断匹配辅助项增益；它仍不能单凭一个对照证明所有种子均有收益。

V362 lastseed14 SOURCE terminal best51/stop61/TOTAL79.968809511716, historical sourcegap-0.253051571324; all144665 restored-best exact/one coverage, own Q50 strictlybefore51 with1761downweighted training rows, no Q refit/causal claim. ActualPOR113 ownSOURCE+prepareddata load traces confirmed. Source processwall1117.917s includes V368 one-trainer/one-eval overlap; frozen TIMING solo field not isolated throughput. Complete newmodels4/5, last model prerequisite1/8 only. Fixed source curves and stage snapshot refreshed CPU-only, no model forward or repeated labels evaluation; final8stage still pending.
