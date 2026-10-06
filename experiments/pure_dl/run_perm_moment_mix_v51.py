"""Explicit two-arm train-only moment-mixing experiment."""
from datetime import datetime
import argparse,json,subprocess,sys
from perm_moment_mix_v51 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append

TEXT='''# v51 PERM跨井隐藏矩混合

## 探索重点与假设
v50迁移PERM89.050807574234较随机+0.239512，但低当前89.396724049366；本批携带迁移表征，检验训练分布增强。plain/interwell同一个v50第21轮完整模型、同836921学习参数/初始化/损失/窗口/预算，无新增学习模块，所有骨干/FiLM/输出继续训练。
算法参考[MixStyle官方实现](https://github.com/KaiyangZhou/mixstyle-release)。本项目适配为深度有效掩码上的均值/标准差与不同训练井配对；不加载外部模型。位置在stem经过整井FiLM后、第一层局部TCN前；计算每个窗口每个隐藏通道的有效深度均值和标准差，统计量detach。以0.5批次概率，用Beta(.1,.1)系数将本窗口与另一训练井窗口的矩做凸组合，再复原标准化局部形状。
这是假设井间矩差异有可作为扰动的部分，不把它当已知物理不变量。标签不混合，也不改输入曲线及其206衍生特征。对照plain同样消耗配对/概率随机数但不应用变换；interwell必须不同井，某窗口若批中无异井伙伴则不增强。两组验证/推理都完全关闭增强，网络不使用井编号作为特征。

## 数据与初始化
所有伙伴来自当步原64训练井锚点；16验证井从不参与增强、梯度、拟合或伙伴统计。分组井ID仅在训练时构建排除同井的布尔关系，不进入任何Linear/Attention/FiLM；输入井库仍只有曲线特征。源迁移模型及其SW前身都仅在原64井训练，沿用开发折选择的局限如实保留。
初始完整144665点精确复现v50 PERM89.050808；当前全局配方仍PERM89.396724/TOTAL80.342771761689。不能把比低分父模型提高写成超过当前最优。原9曲线206、固定第五/seed20261010、Q50、1024上下文/中心128、原Gaussian8密度/默认门控/官方256支持点和800动作保持。

## 训练、已做与待做
已实现两项训练变换与计数证据；开始前无新成绩。共同新AdamWlr3e-5/wd.01/clip2、cosine80*64至1e-5，max80/pat18/min10/b64/64步；新调度并非v50优化器轨迹续跑。完整原子last_state含模型计数、优化器、调度、RNG和早停，显式恢复。最多2GPU、无自动队列，若明显失败可基于真实证据提前结束。
直接实际训练，记录初始与最佳恢复、真实梯度及增强应用数/异井配对数。结束按官方PERM最佳实际三列评分，有保留价值候选磁盘重建；不重复旧密度数值检查。CUDA注意力/插值反向warn_only，不声称重新训练逐位确定。
待做两项训练、完成后的匹配差/父差/当前最优差、增强非失活核验、负权重清理及PROJECTS/README/EXPLORATIONS/atlas更新。单项与并发总墙钟分开，含加载/输入库/预热/启动间隔，代码/最终重载另计。
若本批增强没有最终匹配收益，不扫描概率、Beta参数或层位置；结论只限当前配置，不能否定所有跨井增强。原第五严格>80.5且明确收益才考虑其他折；完整OOF>82仍未完成。无其他折/旧任务/交付覆盖，目标达成后的独立新井确认与新目录交付尚未开始。
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if len(jobs())>=2:raise RuntimeError('Maximum2GPU processes')
    out=ROOT/f'outputs/pure_dl_perm_moment_mix_v51_PERM_{a.mode}_fold5'
    if out.exists():raise FileExistsError(out)
    STUDY.mkdir(exist_ok=True)
    for n,txt in [('AGENTS.md','# v51规则\n仅plain/interwell两项v50完整同起点PERM对照，跨井隐藏矩增强仅训练64井。原数据/Q/评分/80-18/max2保持，无验证增强或井ID特征。维护PROJECTS目标/已做/未做/后续，无补折或交付覆盖。\n'),('README.md',TEXT),('PROJECTS.md',TEXT)]:
        if not (STUDY/n).exists():(STUDY/n).write_text(txt,encoding='utf-8')
    out.mkdir()
    for n in ('AGENTS.md','README.md','PROJECTS.md'):(out/n).write_text((STUDY/n).read_text(encoding='utf-8'),encoding='utf-8')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_perm_moment_mix_v51.py'),'--mode',a.mode,'--out',str(out)]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(pid=p.pid,mode=a.mode,started_at=datetime.now().astimezone().isoformat(),command=cmd,automatic_queue=False)
    (out/'launch.json').write_text(json.dumps(launch,indent=2),encoding='utf-8')
    marker='## v51 '+a.mode+'显式启动'
    for path in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,marker+'\n'+json.dumps(launch,ensure_ascii=False))
    append(ROOT/'EXPLORATIONS.md','perm_moment_mix_v51_20260929','- [v51 PERM跨井隐藏矩混合](results/deep_learning/perm_moment_mix_v51_20260929/README.md)：同v50全模型，训练异井均值/尺度增强与无增强对照；开始前无成绩。')
    append(ROOT.parent/'AGENTS.md','## v51 PERM跨井矩混合','## v51 PERM跨井矩混合\n本批plain/interwell两项，同v50迁移PERM原函数，全部参数训练；异井隐藏均值/尺度增强仅训练，井ID只约束配对，验证关闭。原数据/Q/评分/80-18/max2，无其他折/旧队列/交付覆盖；维护perm_moment_mix_v51_20260929。')
    print(json.dumps(launch),flush=True)

if __name__=='__main__':main()
