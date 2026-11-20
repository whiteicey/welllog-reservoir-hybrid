# 发布审计记录（2026-11-19）

## 当前版本

- 以用户调整后的 `final_submission` 为基准，保留三位作者、共同一作、单位信息和扩展后的正文。
- 当前英文主文 24 页，补充材料 7 页；已同步收录中文对照版本（20 页主文 + 7 页补充材料）。唯一英文官方入口为 `paper/current/`，中文版本入口为 `paper/chinese/`。
- 扩充了薄互层（thin beds）、陡变岩性过渡带（lithologic transitions）与储层流体界面（reservoir boundaries）处树模型与时序模型的岩石物理失效/优势机理分析。
- 对 PERM 效用公式、POR/SW 数值稳定项、独立种子方差、公开测试井语义和过强因果表述做了一致性校正。
- 当前主文和补充材料均无未解析引用或 `??`。

## 数据与凭据审计

- 内部原始测井曲线、训练/测试标签、模型权重和大型 OOF 数组未加入发布目录。
- 当前 Git 历史的大文件主要为模型预测结果，不是隐藏真值。
- 对当前及历史文本执行了 GitHub token、私钥、常见云平台凭据和密码模式扫描，未发现命中。
- 公开前对当前工作树执行 `tools/sanitize_before_public.py --apply`，累计替换 35 个文件中的 60 处主机路径/别名。
- 未改写 Git 历史；历史服务器路径和开发记录保留作审计轨迹。未发现历史中的真实服务器 IP，只有发布清单中的匹配模式和普通数值。
- `docs/history/`、旧论文和旧竞赛叙述均为历史材料，不是当前论文入口。

## 验证

- `test_audit_v35.py`：15 项测试全部通过。
- 公共 Volve 适配器导入通过；`reproduce_public_validation.py` 导入通过。
- 新增实验脚本通过 AST 语法检查。
- 七组论文图件从发布目录的聚合 JSON 重新生成成功。
- 主文和补充材料在服务器使用 `latexmk -pdf -interaction=nonstopmode -halt-on-error` 构建成功。
- `paper/current/manifest.json` 记录当前论文实际文件哈希；`RELEASE_MANIFEST.json` 记录发布文件哈希。

## 未执行项与限制

- 公开 CSV 按 `.gitignore` 排除，因此本次只验证公开复核入口导入，未在发布目录重新下载数据并完整重训。
- 内部受控实验需要用户合法持有的内部数据、缓存和模型参数；公开仓库不包含这些输入。
- 当前 LaTeX 仍有少量 Overfull/Underfull 和模板目的地址警告，已渲染全页联系图检查，未发现内容裁切；具体数值见 `paper/current/manifest.json`。
- DOI 和正式出版信息尚未分配，`CITATION.cff` 未虚构 DOI。
