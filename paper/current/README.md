# 当前论文与构建

唯一现役稿件为：

- `main.tex` / `main.pdf`：英文主文，23 页；
- `supplement.tex` / `supplement.pdf`：英文补充材料，7 页；
- `refs.bib`：参考文献；
- `public_table.tex`：公开 Volve 表格；
- `figures/`：主文和补充材料引用的图件。

题目：*Tolerance-Aware Reservoir Property Prediction from Wireline Logs: Representation Complementarity, Decision Sensitivity, and Offline Transfer*。

作者：Minjie Chen、Jiawei Xin、Chengfeng Wu；Minjie Chen 与 Jiawei Xin 共同一作。

## 构建

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

`manifest.json` 记录本目录所有交付文件的实际 SHA256，旧版本不作为当前稿引用。

## 证据边界

- 公开 Volve/PDDA 测试井在早期开发中使用过，因此属于公共复核，不是全新未触碰盲测。
- 内部 80 井交叉验证、内部 10 井历史盲榜和公共复核是不同证据层，不混合解释为统一排名。
- 不包含内部原始测井曲线、标签、模型权重或大型 OOF 数组。
