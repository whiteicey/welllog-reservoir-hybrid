"""Launch exactly the two predeclared cold-network POR capacity trials."""
from pathlib import Path
from datetime import datetime
import ast,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/scratch_scaling_v23_20260928'
MODES=('base1','ffn2')
METHOD='''# v23：相同双专家从随机权重开始的内部容量比较

## 问题
v21/v22末期扩容均未超过80.274135。本轮让新容量从表征形成开始参与学习，保持两个容量的共同条件一致。只运行base1和ffn2，不自动展开全尺寸/种子/目标网格。

## 网络与容量
完全保留私有POR双专家结构：206维完整视图和68维校准视图、多尺度卷积和局部点/全局patch注意力、9点输入路由；8分量Student-t3/专家及默认门控，两专家经原路由形成最终分布。每位专家位置头含128→32→8的小残差。
base1有1290490参数。ffn2在18个残差块加入等于原宽度的内部单元，并把位置残差32→64，合计1981498参数（1.535×整网参数）。阶段外部宽度、注意力、专家数量和路由不改；两个扩容位置同时变化，不能区分其各自贡献。
原参数按同一seed20261010随机初始化，没有导入旧网络state_dict。扩容使用v21同一代码，新投影输出零初始化；两组共同参数逐元素相同，初始全验证预测需核对。非目标packed输出只是随机的不可训练缓冲行，不进入预测与损失。

## 数据与损失
复用v21准确训练侧缓存、原Q50、原64/16井及行序。POR分量初始位置只由64训练井POR非默认标准化标签的分位数产生。网络权重从头训练，但预处理/支持格/Q50仍使用之前训练侧产物；不称整个pipeline完全冷启动。
只优化最终两专家混合后的POR BCE+Student-t NLL，原总体分母/3。路由和两专家共同由此损失更新。无PERM/SW训练损失，两个其他目标保留各自既有私有模型。
重要区别：80.22原训练用了多目标、深层辅助头、每专家自身损失及独立路由损失、动态Q。这里采用当前v21/v22私有模型的单任务最终混合损失，固定Q50；本轮不是复现那套历史预训练。两种新容量之间可比较，和80.27差异不可唯一归因于容量。

## 训练和评分
从头学习使用共同AdamW6e-4、wd.01、cosine80×64至1e-5、clip2，max80/pat18/min10、bs64、每轮64步。相对末期微调必须记录学习率变化；本轮两组学习率相同。固定抽样生成器，使两组相同轮次看到相同训练窗口。
原256点窗口、中央128监督、原256支持格/800候选动作/单覆盖解码不变。原第五折POR选模，最后实际合并三列计算TOTAL。该折已反复用于开发，没有独立盲测或完整Scaling Law结论。
每轮保存原子可恢复current/optimizer/scheduler/RNG/early-stop和最佳模型。只显式启动/恢复，最多2GPU。选中模型实际恢复预测；仅最终新模型重建，旧PERM/SW复用已验证数组。

## 成功与停止
同时报告相对base1的容量增益和相对80.274135的绝对差。跑满80轮与patience停止分开标注。若无收益保持旧模型/交付，不自动补其他折。第五折严格>80.5和全OOF>82目标未完成。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv)!=1:raise RuntimeError('This launcher runs exactly base1 and ffn2')
    if active_jobs():raise RuntimeError('Existing trainer found; inspect before starting')
    for mode in MODES:
        if (ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5').exists():raise RuntimeError('Output already exists')
    for name in ('scratch_scaling_v23.py','train_scratch_scaling_v23.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    (STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
    rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
    marker='## v23 从头训练的同架构容量对照'
    append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n当前继续授权用于base1/ffn2两个POR私有双专家从随机网络训练；保留原视图、Q50、目标损失和解码。两组统一lr6e-4、80/18/max2；原尺寸1290490参数，扩容1981498。不是原多任务预训练复跑，旧网络/交付/其他折不变。每批维护scratch_scaling_v23_20260928/PROJECTS.md。')
    launches=[]
    for mode in MODES:
        out=ROOT/'outputs'/f'pure_dl_scratch_scaling_v23_{mode}_fold5';out.mkdir()
        (out/'AGENTS.md').write_text(rules,encoding='utf-8')
        (out/'README.md').write_text(METHOD+'\n当前模式：'+mode+'；启动时尚无精度结论。\n',encoding='utf-8')
        (out/'PROJECTS.md').write_text('# 进度\n目标：'+mode+'从头学习的容量验证。\n已做：代码、数据来源与协议。\n计划：训练和恢复最佳、与同批比较。\n未做：最终收益及验证。\n完成后：有实测收益再记录采用，不自动追加或补折。\n',encoding='utf-8')
        cmd=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('train_scratch_scaling_v23.py')),'--mode',mode,'--out',str(out)]
        with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        launch=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=cmd)
        (out/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2),encoding='utf-8');launches.append(launch)
        append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v23 {mode}](outputs/{out.name}/README.md)：原POR双专家从随机权重训练，容量对照；启动时收益未证实。')
        print('LAUNCHED',json.dumps(launch,ensure_ascii=False),flush=True)
    entry='## '+datetime.now().astimezone().isoformat()+' v23显式启动\n'+json.dumps(launches,ensure_ascii=False)+'\n仅这两个预定义任务，无其他折或自动追加。'
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')


if __name__=='__main__':main()
