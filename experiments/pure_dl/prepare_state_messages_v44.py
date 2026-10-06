"""Training-well-only state-transition diagnosis before the bounded v44 study."""
from pathlib import Path
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/state_messages_v44_20260929'

def main():
    STUDY.mkdir(exist_ok=True)
    (STUDY/'AGENTS.md').write_text('# v44约束\n只用训练井状态诊断，SW self/neighbor两个神经消息头固定对照；旧预测器固定，原数据/评分/Q50/80-18/max2不变。维护PROJECTS.md目标/已做/计划/未做/后续；无其他折或旧交付覆盖。\n',encoding='utf-8')
    (STUDY/'PROJECTS.md').write_text('# v44进度\n目标：检查状态邻接是否值得建模，再做SW有限对照。已做：训练侧诊断代码；收益未证实。待做：诊断、真实训练、恢复与三列评分。未做：其他折、新井复现。目标完成后可做独立验证，当前不启动。\n',encoding='utf-8')
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    rows=cache['train_raw'].numpy();train=np.zeros(len(cache['depth']),bool);train[rows]=True
    default=cache['default'].numpy();lower=cache['lower'].numpy()
    pair=train[:-1]&train[1:]&(lower[:-1]==lower[1:])
    results={}
    for c,target in enumerate(('POR','PERM','SW')):
        a=default[:-1,c][pair].astype(int);b=default[1:,c][pair].astype(int)
        counts=np.bincount(a*2+b,minlength=4).reshape(2,2)
        results[target]=dict(adjacent_counts=counts.tolist(),transition_probability=(counts/counts.sum(1,keepdims=True)).tolist(),
            training_default_fraction=float(default[rows,c].mean()),state_change_fraction=float((a!=b).mean()))
    report=dict(scope='training64 wells only; adjacent original raw rows with both labels valid; no validation labels inspected',
                training_rows=len(rows),adjacent_pairs=int(pair.sum()),results=results,
                interpretation='Label persistence motivates a test, not evidence of prediction improvement; marginal state inference uses input and frozen model only.')
    (STUDY/'training_state_diagnostic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
