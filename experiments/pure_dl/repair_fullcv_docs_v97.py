"""Repair only newly generated notes affected by PowerShell stdin encoding."""
from pathlib import Path
from fullcv_context_v97 import ROOT, STUDY

note='''\n\n## 补折训练入口完成
已固定18有效训练阶段和1个折内来源骨干，按依赖顺序串行补全原1~4折；通道各自损失与验证最佳checkpoint，原第五预测复用。v8私有提取第0轮逐权重完全一致，固定为身份阶段。通用decoder实际第五三通道144665点逐元素复现；完整源码哈希见training_entry_manifest.json。授权后的controller逐项执行，失败停止无重试；不重启其他旧路线。
未完成：其余折实际最终三通道、完整OOF>82、新目录交付。
'''
for name in ('README.md','PLAN.md'):
    path=STUDY/name;s=path.read_text(encoding='utf-8');i=s.rfind('\n\n## ')
    if i>=0 and s[i+6:].startswith('?'):s=s[:i]
    path.write_text(s+note,encoding='utf-8')
text='''# v97折内来源训练
当前原第一折local-context80多通道预训练，全部统计/Q/标签只用原64训练井。是固定三通道配方依赖，完整OOF尚未完成。80/18/b64/64steps，原子完整last，恢复显式；维护PROJECTS，每批更新，旧文件与交付保留。
'''
for name in ('AGENTS.md','PROJECTS.md','README.md'):
    (ROOT/'outputs/pure_dl_fullcv_v97/root'/name).write_text(text,encoding='utf-8')
p=ROOT/'EXPLORATIONS.md';s=p.read_text(encoding='utf-8')
lines=[line for line in s.splitlines() if not ('fullcv_v97_20260930/PROGRESS.md' in line and '?' in line)]
lines.append('- [v97 已授权其余四折训练](results/deep_learning/fullcv_v97_20260930/PROGRESS.md)：固定v96第五80.504197，18有效阶段加折内来源，原行序实际OOF待完成。')
p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
