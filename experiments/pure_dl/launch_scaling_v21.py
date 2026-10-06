"""Explicit three-size champion scaling study; no implicit training queue."""
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,subprocess,sys
from launch_latent_gate_v17 import active_jobs,append_once
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_scaling_v21_20260928'
MODES=('base1','ffn2','ffn4')
METHOD='''# v21 扩容实验协议

收益归因见GAIN_ANALYSIS.md及gain_attribution.json。当前冠军为POR私有双专家77.448916、PERM89.347503、SW73.622385，TOTAL80.274135。

## 预先固定的三个规模
base1：原POR私有双专家，作为同起点同预算继续训练对照。
ffn2、ffn4：在两个专家共18个UngatedBlock内，将非线性隐藏宽度分别扩到原来的2/4倍。块的外部通道仍64/96/128/192，层数、卷积核、注意力与视图不变。两专家POR位置小残差的内部宽度同步从32扩为64/128。

旧块计算y=x+gain*W2[0.5*GELU(W1*z+b1)]。扩容添加W2new[0.5*GELU(W1new*z+b1new)]，新输出权重W2new=0。旧计算保持独立求值再相加，避免重排已有乘加；新增神经元输入不同，因此训练后可以提供新增表示。数学上等价于内部隐藏单元增加。此处参考Net2Net的函数保持扩容思想（arXiv:1511.05641），没有声称完整复现其复制单元算法。

所有初始验证点必须逐元素重现旧POR，否则本次训练直接中止并记录原因。所有旧/新增POR可学习参数允许训练，PERM/SW保持私有固定模型。骨干lr1e-5，头lr3e-5，新增块参数跟随其所属组，不额外提高学习率；wd0.01/clip2/cosine到1e-6、80轮/patience18/min10/bs64/64steps相同。

## 隔离与评价
原九曲线/206特征/相对深度、64训练井/16原第五折、原Q50、own POR NLL/3、原256格/800动作/单覆盖。完整原子模型/优化器/调度器/RNG恢复状态。每次只显式启动指定的预定义项，最大2GPU，不补折或覆盖旧交付。

报告每个档位实际参数量、显存、轮耗时、停止轮次、训练后最佳及含第0轮的最佳。主比较是扩容相对同批base1，另与80.274135对比。不能将扩容的回退第0轮当作新收益；只能选择完整目标模型，不能逐行/逐井选优。第五折反复用于开发，一次实验不能推出普遍scaling law。
'''

def replace(text,old,new):
 if text.count(old)!=1:raise RuntimeError('Frozen trainer fragment mismatch: '+old[:100])
 return text.replace(old,new,1)

def build_trainer():
 parent=ROOT/'experiments/pure_dl/train_independent_channels_v8.py'
 expected='7c566491e9be82e6ba74c64aa3623aae72e2ffb3ac88fa3ae970db0565650a78'
 if hashlib.sha256(parent.read_bytes()).hexdigest()!=expected:raise RuntimeError('Inspect changed v8 trainer')
 text=parent.read_text(encoding='utf-8-sig')
 edits=[
 ('from independent_channels_v8 import Network,TARGETS,ARCHITECTURES','from independent_channels_v8 import TARGETS,ARCHITECTURES\nfrom champion_scaling_v21 import make_model,load_data,MODES'),
 ("ap.add_argument('--resume',action='store_true');a=ap.parse_args()", "ap.add_argument('--mode',choices=tuple(MODES),required=True)\n    ap.add_argument('--resume',action='store_true');a=ap.parse_args()\n    assert a.target=='POR' and a.architecture=='dual'"),
 ('data=prepare_data(master)','data,_cached_master=load_data();del _cached_master'),
 ('model=Network(master,previous,a.target,a.architecture).cuda()','model=make_model(a.mode,master).cuda()'),
 ("'model_module':'independent_channels_v8',","'model_module':'champion_scaling_v21','mode':a.mode,'scaling':model.scaling_manifest,"),
 ("'dropout_during_finetuning':True,","'dropout_during_finetuning':'actual unchanged module behavior; see scaling manifest',"),
 ("Path(__file__).with_name('independent_channels_v8.py')]", "Path(__file__).with_name('independent_channels_v8.py'),Path(__file__).with_name('champion_scaling_v21.py')]"),
 ("'private_process_pid':os.getpid(),'other_target_losses_computed':False}", "'private_process_pid':os.getpid(),'other_target_losses_computed':False,\n                     'extra_output_gradient_norm':float(sum(p.grad.detach().float().square().sum() for n,p in model.named_parameters() if '.extra_' in n and p.grad is not None).sqrt()) if a.mode!='base1' else 0.}"),
 ("'scope':'one target only; independent model from raw input; no shared backbone'", "'scope':'POR champion function-preserving internal-width scaling; PERM/SW frozen'"),
 ("'scores':{a.target:score},'target_score':score,", "'scores':{a.target:score},'target_score':score,'mode':a.mode,'scaling':model.scaling_manifest,\n         'best_after_update':max(history,key=lambda r:r['target_score']) if history else None,"),
 ("'architecture':a.architecture,'last_completed_epoch':epoch,", "'architecture':a.architecture,'mode':a.mode,'last_completed_epoch':epoch,")]
 for old,new in edits:text=replace(text,old,new)
 # The original data-provenance setup is retained, but prepare_data is performed
 # once in prepare_scaling_v21. Exact copied evaluator/loss remain unchanged.
 dest=ROOT/'experiments/pure_dl/train_scaling_v21.py';ast.parse(text)
 if dest.exists():
  if dest.read_text(encoding='utf-8')!=text:raise RuntimeError('Existing v21 trainer changed')
 else:dest.write_text(text,encoding='utf-8')
 adaptation=STUDY/'trainer_adaptation.json'
 if not adaptation.exists():adaptation.write_text(json.dumps(dict(parent=str(parent.relative_to(ROOT)),sha256=expected,
  changes=[dict(old=o,new=n) for o,n in edits],old_source_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')
 return dest

def main():
 sys.stdout.reconfigure(encoding='utf-8')
 ap=argparse.ArgumentParser();ap.add_argument('--launch',choices=MODES,nargs='+',required=True);a=ap.parse_args()
 if len(set(a.launch))!=len(a.launch):raise ValueError('Duplicate mode')
 if len(active_jobs())+len(a.launch)>2:raise RuntimeError('Maximum two project GPU jobs')
 if not (STUDY/'data_manifest.json').exists():raise RuntimeError('Prepare exact data first')
 for mode in a.launch:
  if (ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5').exists():raise RuntimeError('Output exists; inspect before resuming')
 trainer=build_trainer();rules=(STUDY/'AGENTS.md').read_text(encoding='utf-8')
 if not (STUDY/'METHOD.md').exists():(STUDY/'METHOD.md').write_text(METHOD,encoding='utf-8')
 marker='## v21用户要求冠军收益分析与scaling up'
 append_once(ROOT.parent/'AGENTS.md',marker,marker+'\n本批预定义base1/ffn2/ffn4三个POR私有双专家热启动对照，函数保持扩容18块内部非线性与小位置头；原64/16数据、Q和评分、80/18/max2保持。完整输入输出通道和注意力不扩宽。固定PERM/SW，显式分批，无补折或旧交付覆盖。每批维护champion_scaling_v21_20260928/PROJECTS.md。')
 records=[]
 for mode in a.launch:
  out=ROOT/'outputs'/f'pure_dl_champion_scaling_v21_{mode}_fold5';out.mkdir()
  (out/'AGENTS.md').write_text(rules,encoding='utf-8')
  (out/'README.md').write_text(METHOD+'\n本目录模式：'+mode+'。启动时未有扩容训练收益。\n',encoding='utf-8')
  (out/'PROJECTS.md').write_text('# 扩容实验进度\n\n目标：'+mode+'是否改善冠军POR。\n已做：协议、代码、原数据与原模型来源核对。\n计划：训练并恢复最佳，比较更新后最佳。\n未做：训练收益和最终组合评价。\n完成后：保留有效候选，无补折或覆盖交付。\n',encoding='utf-8')
  command=[sys.executable,'-X','utf8','-u',str(trainer),'--target','POR','--architecture','dual','--mode',mode,'--out',str(out)]
  with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
   proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,close_fds=True,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
  item=dict(mode=mode,pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),command=command)
  (out/'launch.json').write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding='utf-8');records.append(item)
  append_once(ROOT/'EXPLORATIONS.md',out.name,f'- [v21 {mode}：冠军内部宽度扩容](outputs/{out.name}/README.md)，原POR功能起点保持，新增收益待实际训练。')
  print('LAUNCHED',json.dumps(item,ensure_ascii=False),flush=True)
 entry='## '+datetime.now().astimezone().isoformat()+' v21显式启动\n'+json.dumps(records,ensure_ascii=False)+'\n仅预定义3规模之一或两项，最多2GPU，其他尚未启动项不自动排队。'
 for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md'):
  with p.open('a',encoding='utf-8') as f:f.write('\n\n'+entry+'\n')

if __name__=='__main__':main()
