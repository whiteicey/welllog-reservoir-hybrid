"""Finalize the completed complex-head experiment. Never starts training."""
from __future__ import annotations
import json,sys
from datetime import datetime
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_attention_v6_20260928'
ATLAS=ROOT/'docs/deep_learning/atlas'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    progress=json.loads((STUDY/'progress.json').read_text(encoding='utf-8'))
    diagnostic=json.loads((STUDY/'attention_implementation_diagnostic.json').read_text(encoding='utf-8'))
    assert progress['all_complete']
    details={r['mode']:r for r in diagnostic['runs']}
    assert set(details)=={'linear','deep_mlp','attention'}
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            d=p.info;args=d.get('cmdline') or []
            if 'python' in (d.get('name') or '').lower() and 'pure_dl' in ' '.join(args) and any(Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in args):active.append(d)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,'Do not finalize active training'
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    baseline=progress['baseline_scores']['TOTAL']
    assert all(r['final_report']['best_epoch']==0 for r in progress['runs']), 'Decision must be revised if any candidate has a gain'
    old=json.loads((ROOT/'results/deep_learning/dual8222_location_v5_20260928/CURRENT_RECIPE.json').read_text(encoding='utf-8'))
    recipe={**old,'observed_at':now,'new_v6_model_adopted':False,
            'decision':'Retain previously verified v5point32; all3v6 trials restore epoch0, no incremental gain.',
            'new_checkpoint_reload':'Not repeated: no improved candidate; original v5 reload proof remains valid.'}
    (STUDY/'CURRENT_RECIPE.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 复杂预测头与注意力：最终结论','',now,'',
           '**本批三个实验均已完成。更深MLP和两层局部多头注意力都没有超过原80.274135337791；保留原v5模型，不替换交付。**','',
           '## 实际训练结果','',
           '|方案|可训练参数|更新参数后的最高TOTAL（不含第0轮）|对应轮次|相对原模型|最终选定轮次|停止轮次|',
           '|---|---:|---:|---:|---:|---:|---:|']
    names={'linear':'线性位置继续训练','deep_mlp':'两层深MLP','attention':'两层四头局部注意力'}
    for row in progress['runs']:
        d=details[row['mode']]
        lines.append(f"|{names[row['mode']]}|{row['trainable_parameters']}|{d['best_trained_nonzero_epoch_TOTAL']:.9f}|{d['best_trained_nonzero_epoch']}|{d['best_trained_nonzero_epoch_TOTAL']-baseline:+.9f}|{d['selected_epoch']}|{d['stopped_epoch']}|")
    lines+=['','三个最终报告的TOTAL都是80.274135337791，是因为各自恢复第0轮原函数；不能把这个数字当新增结构的训练成果。恢复预测逐元素一致，因此最终预测差值与重采样区间机械地为零，这不是三种训练机制等效的统计证明。','',
            '## 已完成的结构','深头先拼接同井两个专家的128维隐藏特征，256->64投影后通过两层残差块。deep_mlp每块包含两个64->128->64逐点网络；attention将第一个网络替换为4头、单头16维、前后16点的局部注意力及相对位置偏置，保留相同FFN和宽度。',
            '二者参数量相差392（约0.45%）。新增输出是两位专家各8个POR分量的位置修正；原门控、混合权重、尺度、PERM/SW、骨干与v5残差均固定。完整公式和维度见METHOD.md。',
            '共同线性位置lr3e-5，新增随机深头lr3e-4；两种复杂头条件一致。其他原始九曲线/206特征、原第五折、Q50、80/18/min10、每轮64步与单覆盖256格800动作解码不变。','',
            '## 注意力是否真的训练','attention两层相对位置偏置由零变为非零，最后L2范数分别为'+str(details['attention']['last_relative_bias_L2'])+'；新增输出投影也已更新。首批反向范围确认只有POR读出可训练。',
            'CPU检查通过：全padding输出有限且为零、有效输出不受padding数值影响、单层只读取半径16范围。实现检查覆盖这些边界，不等于对模型正确性或泛化的全面证明。',
            'attention日志中的连续NLL从第一轮'+str(details['attention']['first_epoch_training_NLL'])+'变为最后一轮'+str(details['attention']['last_epoch_training_NLL'])+'，同时验证分未超过起点。两个NLL来自不同随机批次的轮均值；只能描述训练记录，不能当同一固定训练人口的严格对照。','',
            '## 结论适用范围与下一步','本次未支持“在冻结双专家上追加局部attention并仅调整POR分量位置”获得新的开发集收益；没有试验整类attention、完整输出参数或其他目标的所有设计。',
            '候选方向（未启动）：用8个分量查询分别从局部隐藏序列读取证据的cross-attention读出，与同容量无序列查询控制比较；或对完整分布输出与训练目标做独立对照。需先锁定结构/目标范围，不能把本轮失败包装成支持扩大同类宽度或层数。',
            '当前仍保留原80.274135。第五折严格>80.5与完整OOF>82未达成，完整OOF已知最好81.713233；本批无补折、无旧ABD恢复、无交付替换。三项训练全部结束。无改进候选，因此未重复为原模型做新的磁盘重载；已有v5重载证明保持。','',
            '## 已保存证据','- progress.json：正式报告、初始/最终预测核对、冻结范围和按井分解。',
            '- attention_implementation_diagnostic.json：各组实际训练最高分、注意力更新与mask边界检查。',
            '- trainer_derivation.json：新训练器改动，旧训练器未改。',
            '- CURRENT_RECIPE.json：继续引用已验证v5权重，new_v6_model_adopted=false。',
            '- 各outputs/pure_dl_dual8222_v6_*目录：README、config、history、heads_best、last_state和最终report。',
            '- METHOD.md：结构、形状、公式和比较边界。','',
            '## 时间口径',json.dumps(progress['timing'],ensure_ascii=False,indent=2),
            '最多2训练进程并发。时间范围含显式分批启动间隔，不当作同工作量的部署速度比较。']
    final='\n'.join(lines)+'\n'
    (STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    note='\n\n## '+now+' 复杂预测头v6已完成\n三项均best_epoch0、stop18，原80.274135保持。实际更新后最高linear80.272137140362/deepMLP80.270535660860/attention80.272549291930；没有新增收益。attention相对偏置真实更新，mask四项检查通过。当前无训练进程，无补折或交付变更；分量query/cross-attention尚未实现或启动。重载不重复，保留原v5证明。详见results/deep_learning/dual8222_attention_v6_20260928/FINAL.md。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ATLAS/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    for row in progress['runs']:
        folder=ROOT/'outputs'/row['id']
        for fn in ['README.md','PROJECTS.md']:
            with (folder/fn).open('a',encoding='utf-8') as f:f.write(note)
    cur=json.loads((ATLAS/'curation.json').read_text(encoding='utf-8'))
    for row in progress['runs']:
        d=details[row['mode']]
        cur['nodes'][row['id']]['conclusion']=f"完成18轮，最佳为第0轮80.274135；更新后的最高{d['best_trained_nonzero_epoch_TOTAL']:.9f}@{d['best_trained_nonzero_epoch']}。未取得新增收益，原模型保持；未补折。"
    dead_id='dead:dual8222_v6_complex_location_readouts'
    cur['deadends']=[d for d in cur.get('deadends',[]) if d['id']!=dead_id]
    cur['deadends'].append({'id':dead_id,'title':'冻结位置读出：本轮深头无增益','stage':'loss',
        'parents':[r['id'] for r in progress['runs']],
        'summary':'原80.274135同起点的线性/两层深MLP/两层局部attention，POR位置更新，三组都恢复第0轮。',
        'reason':'只关闭本次结构+训练设置附近的追加宽度/层数扫描；不推及所有attention或完整分布头。实际attention梯度与mask已检查，无独立泛化收益。'})
    (ATLAS/'curation.json').write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'all_complete':True,'current_scores':recipe['scores'],'new_model_adopted':False,
                      'trained_peak_scores':{k:v['best_trained_nonzero_epoch_TOTAL'] for k,v in details.items()},
                      'active_trainers':active,'summary':str(STUDY/'FINAL.md')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
