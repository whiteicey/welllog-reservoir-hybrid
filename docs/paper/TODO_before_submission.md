# 论文投前待办（docs/paper/main_v2.tex）

> 服务器 TeX Live 编译通过：`latexmk -pdf main_v2.tex` → 31 页、0 警告、14 图。
> 编译方法：把 `main_v2.tex` + `refs.bib`（在仓库 `paper/refs.bib`）+ `figures/` 复制到临时目录后编译。

## 必须由用户提供/确认

- [ ] **作者块**（`\author` / `\affiliation` / `\cortext`）：
  - 姓名、单位（中英文）、城市、邮编、国家
  - 通讯作者邮箱
  - 作者顺序
  - （当前为占位：First/Second/Third Author、first.author@example.org、Department, University）
- [ ] 是否需要致谢（竞赛主办方/平台/资助）
- [ ] 竞赛平台 URL 已填入 `https://comp.kunlungpt.cn`——如规则不允许公开提及赛题名称/平台，请告知改为匿名描述
- [ ] Mendeley Data DOI：投稿时若尚未产生，可先写 "DOI assigned upon acceptance"（已按此措辞）

## 我方可独立完成（待办）

- [ ] `[verify DOI]` 引用核验：refs.bib 目前 13 条均为真实文献（无占位），但仍建议 CrossRef 核对：
  - archie1942 `10.2118/942054-G`
  - phoon2024trustworthy `10.1016/j.geoai.2024.100008`
  - fu2024contest `10.30632/PJV65N1-2024a6`
  - yu2021sonic `10.30632/PJV62N4-2021a4`
  - berger1985 `10.1007/978-1-4757-4286-2`
  - rabiner1989hmm `10.1109/5.18626`
- [ ] 图表编号一致性最终检查（正文引用 fig1–14 / tab1–8，编译无 undefined reference ✓）
- [ ] §5.9 新增段落的数字与 `docs/*.md` 台账逐条核对（已核对：gate 83.9203 / distill2 83.7918 / distill3 83.7676 / offsets +1.76/+1.58/+1.50）
- [ ] 摘要长度：当前摘要约 430 词（较长）；若期刊限制 250 词需压缩（原 v2 稿已超，待投稿系统确认）
- [ ] Highlights：5 条（Elsevier 上限 5 条）✓
- [ ] Cover letter 草稿（投稿时）
- [ ] 作者贡献声明（CRediT）、利益冲突（已有）、生成式 AI 使用声明（已有）
- [ ] 关键词核对（当前 9 个，期刊通常 3–6 个）

## 数据可用性一致性

- [ ] 论文声称 "code repository will be made public upon acceptance" → 与 `docs/release_checklist.md` 的公开计划一致
- [ ] B 榜（10 井）赛后结果若公开，可在 arXiv/期刊最终版补充一句 B 榜终审说明（可选）
