"""Close the bounded inner study; do not launch any further training."""
from pathlib import Path
from datetime import datetime
import json,subprocess,sys
import psutil
from prepare_sw_group_v14 import ROOT,STUDY

def main():
    progress=json.loads((STUDY/'progress_inner.json').read_text(encoding='utf-8'))
    assert progress['all_complete'] and not progress['original_fifth_evaluated']
    rows={r['mode']:r for r in progress['rows']};cmp=progress['comparison']
    p=rows['point']['report'];w=rows['well']['report'];pd=rows['point']['diagnostic'];wd=rows['well']['diagnostic']
    text=f'''# SW内层跨井泛化诊断 v14

## 本轮结论
本轮完成两项从零训练的私有SW TCN。仅使用原64训练井内部的48/16划分，原第五折未用于本轮训练、早停、评分或候选比较。两项都自然结束，最佳监控预测已共同按固定最佳参数重推理复现。

按井均衡没有超过按点监督：内层SW差{cmp['SW_delta']:+.9f}，{cmp['wins']}/16井提高、{cmp['losses']}井下降，未达到预设+0.05及至少9井提高的进一步试验门槛。因此未启动原64井重新训练，不改变旧三模型80.274135337791。

## 真实隔离与对照
48训练井有434270个有效点，16监控井有144621个有效点。插补统计、206维特征中位数/IQR、SW目标标准化、256格支持、初始标签先验全部仅由48训练井拟合。没有加载旧模型或旧Q，Q恒为1；原始9曲线与206维特征语义不变。
模型均82113参数，输入为B×256×207，207含206特征和1相对深度；初始权重完全相同。两个候选训练窗口与优化参数相同，唯一变化为学习权重。
对照w_i=1，均衡w_i=N/(W*n_w)，N是训练有效点总数，W=48是训练井数，n_w是点i所属井的有效点数，均为标量。L=mean_i[w_i*(BCE(g_i,d_i)+(1-d_i)*CE(p_i,z_i))]/3，g_i为默认概率标量、d_i为0/1标签、p_i为256维连续概率、z_i为变换后的SW标量。窗口边缘与重叠让这个式子近似井均风险，评估依旧按点官方SW。
训练井最短4379点、最长12319点，权重0.734418—2.066063，均值1。它调节各井训练贡献，不判定标签好坏。

## 最佳参数下结果（全部是内层SW）
|方法|训练SW|监控SW|训练非默认段SW|监控非默认段SW|最佳/停止轮次|
|---|---:|---:|---:|---:|---|
|按点|{p['fit']['SW']:.6f}|{p['monitor']['SW']:.6f}|{p['fit']['numeric_SW']:.6f}|{p['monitor']['numeric_SW']:.6f}|{p['best_epoch']}/{p['stopped_epoch']}|
|按井均衡|{w['fit']['SW']:.6f}|{w['monitor']['SW']:.6f}|{w['fit']['numeric_SW']:.6f}|{w['monitor']['numeric_SW']:.6f}|{w['best_epoch']}/{w['stopped_epoch']}|

这些分数不是原第五折分数，不能和原SW73.622385直接比较，也不能拼成新的80.x总分。监控集参与早停，当前只做一次内层切分，并非完整嵌套CV或独立盲测。

## 定位到的现象
按点组在最佳第18轮后继续18轮，训练SW上升{pd['after_best']['fit_SW_change']:+.6f}，监控SW变化{pd['after_best']['monitor_SW_change']:+.6f}。训练连续交叉熵继续下降，监控连续交叉熵反而上升{pd['after_best']['monitor_CE_change']:+.6f}。同一组模型轨迹说明后期拟合改善没有传到未见井，支持后期过拟合这一解释，但不证明唯一根因。

总体分差会被样本组成遮住：训练默认率{100*pd['fit_default_rate']:.4f}%，监控默认率{100*pd['monitor_default_rate']:.4f}%。最佳时训练总分甚至略低于监控；把训练默认/非默认子集分数按监控默认率重新加权后，训练仍高{pd['default_prior_adjusted_fit_minus_monitor']:.6f}分。原始非默认子集差为{pd['numeric_fit_minus_monitor']:.6f}分。这个标准化仅是离线描述性诊断，没有修改模型或原评分。

均衡组监控默认段比对照高{cmp['default_delta']:+.6f}，非默认段变化{cmp['numeric_delta']:+.6f}，净收益为负。探索性整井重采样95%区间{cmp['exploratory_selection_unadjusted_95']}跨零；不证明井均衡在所有设置无效。

## 后续边界
停止本批井权重指数/学习率扩展，不增加外层训练或原其他折。下一项未实施：在当前严格内层切分上检验跨井局部参考对非默认数值预测的增量，先固定参考库仅来自48训练井，避免本井近邻或旧模型见过监控井造成偏乐观；有证据再考虑原第五折。没有声称该机制已有效。
继续保留三个通道各自的私有模型，未改已交付文件。原第五折>80.5和完整OOF>82仍未完成。

## 运行与可追溯性
两组并发，实际从显式启动到两组结束墙钟{progress['timing']['launch_to_last_finish_wall_seconds']:.3f}秒；单项记录按点{p['seconds']:.3f}秒、按井{w['seconds']:.3f}秒。两组各峰值显存约{p['peak_allocated_MiB']:.3f}MiB。数据准备另计{progress['timing']['preparation_seconds']:.3f}秒；均不含编码和报告工作。不可将两个并发单项耗时相加解释总墙钟。
保存了当前/最佳权重、优化器/调度器/RNG、历史轨迹、数据缓存和拟合范围清单。最终做的是当前运行内最佳参数恢复核对，未另做磁盘重建推理或不中断/恢复等价性测试。
图谱由report_sw_group_v14.py的独立人工元数据扩展维护，再执行build_atlas.py；不重写会抹去后续元数据的旧主curate.py。

## 证据
- data_inner_manifest.json：井划分、训练范围、权重统计。
- progress_inner.json：匹配配置、实际分数、逐井差、默认率标准化与轨迹分解。
- outputs/pure_dl_sw_group_v14_inner_point80/final_report.json。
- outputs/pure_dl_sw_group_v14_inner_well80/final_report.json。
'''
    (STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    previous=ROOT/'results/deep_learning/independent_sw_tolerance_v13_20260928/CURRENT_RECIPE.json'
    recipe=json.loads(previous.read_text(encoding='utf-8'))
    recipe.update(v14_original_fifth_evaluated=False,v14_models_selected=False,v14_scope='inner48to16 diagnostic only',
                  prior_recipe=str(previous.relative_to(ROOT)),new_three_model_gpu_replay_this_turn=False)
    (STUDY/'CURRENT_RECIPE.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    curpath=ROOT/'docs/deep_learning/atlas/curation.json';cur=json.loads(curpath.read_text(encoding='utf-8'))
    if not cmp['predeclared_inner_screen_passed']:
        ident='dead:sw_group_v14_inner_balance';dead=cur.setdefault('deadends',[])
        dead[:]=[x for x in dead if x.get('id')!=ident]
        dead.append(dict(id=ident,title='按井均衡·本内层切分未达继续门槛',stage='backbone',
            parents=[r['id'] for r in progress['rows']],summary=f"同TCN/初始化，内层SW差{cmp['SW_delta']:+.6f}，{cmp['wins']}井升/{cmp['losses']}井降。",
            reason='仅关闭本次井数加权附近的扩展；单内层切分非独立盲测，不否定一般分组学习。'))
        curpath.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    note='\n\n## '+datetime.now().astimezone().isoformat()+' v14最终收尾\n完成严格48/16内层实验，原第五折未评估。按井均衡未通过预设内部门槛，不启动额外原64井训练。发现默认率差可掩盖总体泛化差，后期训练分升而监控分降；具体见results/deep_learning/sw_group_generalization_v14_20260928/FINAL.md。后续跨井局部参考未实现，旧三私有模型80.274135保持，原其他折与交付不改。\n'
    for pth in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
        with pth.open('a',encoding='utf-8') as f:f.write(note)
    with (ROOT.parent/'AGENTS.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## v14本轮实现与验证范围\n在继续优化授权下，本轮执行原64训练井内部48/16分组诊断，两组新TCN只差井学习权重，全部统计重新拟合且Q=1。原第五折未评估；当前内层结果不可写成第五折TOTAL，不自动补原其他折。真实结果和后续待做以sw_group_generalization_v14_20260928/FINAL.md与PROJECTS.md为准。\n')
    run=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'docs/deep_learning/atlas/build_atlas.py')],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
    print('ATLAS',run.returncode,run.stdout,run.stderr)
    if run.returncode:raise RuntimeError('Atlas build failed')
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=proc.info;argv=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and any('pure_dl/' in x.replace('\\','/') and Path(x.replace('\\','/')).name.startswith(('train_','benchmark_')) for x in argv):active.append(d['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps(dict(report=str(STUDY/'FINAL.md'),active_training_pids=active,inner_delta=cmp['SW_delta'],original_fifth_evaluated=False),ensure_ascii=False))

if __name__=='__main__':main()
