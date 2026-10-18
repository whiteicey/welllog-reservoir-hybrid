# 开源前发布清单（repo → public，比赛结束 + 论文接收后执行）

> 当前仓库为 **private**，比赛数据不在库内。以下清单只在准备公开时逐项执行。

## 1. 合规（必须先过）

- [ ] **竞赛数据零入库**：`data/`（train/test 原始曲线文件与标签）不得出现在仓库或 Release；确认 `git log --all --stat | grep -iE 'data/(train|test)'` 无命中
- [ ] **提交预测件（submissions/*/result.zip）**：赛题条款若禁止公开预测结果，比赛结束后仍建议保留（官方允许公开后再放开）；保守做法：移入 Release 附件或私有归档
- [ ] 竞赛平台/主办方名称与赛题描述引用方式确认（是否允许在论文/仓库中提赛题名称）
- [ ] 论文接收后再公开 repo 与数据可用性声明一致（否则先只公开代码与文档）

## 2. 去敏感化（工具已备）

```bash
python tools/sanitize_before_public.py           # 先 dry-run 看清单
python tools/sanitize_before_public.py --apply   # 再改写（检查 diff 后提交）
```

替换规则：`/home/ubuntu/...` → `<PROJECT_ROOT>`；服务器别名 → `<SERVER_ALIAS>`；IP → `<HOST_IP>`；端口 → `<PORT>`；localhost 端口 → `<LOCALHOST_PORT>`。
预计命中：约 30 个文件、70+ 处（docs/ 历史报告、src/queue*.sh、submissions metadata、ROADMAP 等）。

- [ ] `tools/sanitize_before_public.py --apply` 执行并人工复核 diff
- [ ] `git grep -nE '120\.53\.|22023|ubuntu-server|/home/ubuntu'` 应无命中（除补丁脚本自身）
- [ ] 检查 git 历史中的服务器路径：可选 `git filter-repo --path-glob` 历史注入清理（如需彻底）

## 3. 内容整理

- [ ] README 顶部更新为最终战报（A 最佳 83.9361、14 次提交终审、论文 31 页 14 图）
- [ ] CHANGELOG 已修复编码（2026-10-18 UTF-8 修复 + 最终战役条目）
- [ ] `docs/paper/main_v2.pdf` 为最新编译版本（31 页、0 警告）
- [ ] 论文占位符填完：作者块、数据可用性 URL/DOI、致谢
- [ ] 清理工作区临时脚本（均为 `_*.py`，已在 .gitignore 或删除）
- [ ] `requirements.txt` 覆盖服务器 venv 关键依赖（torch/numpy/pandas/sklearn/catboost/lightgbm）
- [ ] 增加 `CITATION.cff`（论文正式发表后补 DOI）

## 4. 复现性

- [ ] README「快速开始」：数据放置方式、环境安装、一键复现论文图/表脚本
- [ ] 论文图脚本：`docs/paper/make_fig14.py` 等；旧 1–13 图由历史脚本生成（保留 CSV 快照）
- [ ] 确定性声明：推理 SHA256 manifest 与固定种子说明
- [ ] 论文 §实验 每个数字对应脚本/JSON 路径核对表（至少覆盖主表）

## 5. 发布动作

- [ ] 公开前最后的 `git status` 干净、无未跟踪大文件
- [ ] 打 tag：`v-final`（对应 A 榜最终工件 pl_pw 的 manifest）
- [ ] Release 附件：模型权重包（DL runtime）与可复现提交包（如允许公开）
- [ ] 在 README 放置论文链接（出版后 DOI）与 citation
