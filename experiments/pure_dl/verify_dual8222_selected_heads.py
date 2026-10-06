"""Select one fixed completed readout per target and replay a shared dual backbone.

Selection is across whole-target validation scores, never rows/wells. This is
explicitly tuned development evidence, not an independent evaluation. No training.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import psutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_dual8222_heads80 import SourceEvaluator, prepare_data, atomic_json, atomic_torch, atomic_array
from paired_local_context_router_sequence import Network as Original, retained_indices, calibration_indices
from paired_detached_router_sequence import routed_merge
from dual8222_independent_heads import ResidualReadout
from dual8222_head_structure_v2 import TargetReadout
from gpu_job_guard_v2 import check_slot
from analyze_dual8222_round2 import ARMS, TARGETS

ROOT=Path(__file__).resolve().parents[2]
SOURCE_DIR=ROOT/'outputs/pure_dl_paired_local_context_router80_sequence'
DEST=ROOT/'outputs/pure_dl_dual8222_selected_round2_fold5'


class FixedSelectedHeads(nn.Module):
    def __init__(self, original, choices, checkpoints):
        super().__init__()
        self.backbone=original.eval().requires_grad_(False)
        self.kinds=[];groups=[]
        for c,choice in enumerate(choices):
            arm=choice['arm']
            if arm in {'pointwise','local5'}:
                group=nn.ModuleList([ResidualReadout(member.main_head,c,arm) for member in original.members])
                kind='separate_residual'
            else:
                mode='cross80' if arm=='cross80' else 'linear'
                group=TargetReadout(original.members,c,mode)
                kind='target_group'
            if arm!='source':group.load_state_dict(checkpoints[arm]['head_state_dicts'][c],strict=True)
            groups.append(group);self.kinds.append(kind)
        self.heads=nn.ModuleList(groups).eval().requires_grad_(False)
        self.captured={}
        self.handles=[member.main_head.register_forward_pre_hook(self._capture(e)) for e,member in enumerate(original.members)]

    def _capture(self,e):
        def hook(_module,inputs):self.captured[e]=inputs[0].detach()
        return hook

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        if auxiliary:raise ValueError('Inference only')
        self.captured.clear()
        for member in self.backbone.members:
            member(features,relative_depth,valid,absolute_depth=absolute_depth)
        logits=self.backbone.route_logits(features,valid)
        combined=torch.cat([self.captured[0],self.captured[1]],dim=1)
        corrections=[]
        for group,kind in zip(self.heads,self.kinds):
            if kind=='separate_residual':
                corrections.append([group[e].residual(self.captured[e],valid) for e in range(2)])
            else:
                value=group.corrections(combined,valid)
                corrections.append([None,None] if value is None else [value[:,:25],value[:,25:]])
        outputs=[]
        for e in range(2):
            h=self.captured[e]
            heads=[group[e] if kind=='separate_residual' else group.branches[e] for group,kind in zip(self.heads,self.kinds)]
            gate=F.conv1d(h,torch.cat([head.gate_weight for head in heads]),torch.cat([head.gate_bias for head in heads]))
            raw=F.conv1d(h,torch.cat([head.density_weight for head in heads]),torch.cat([head.density_bias for head in heads]))
            # For targets with no residual, adding exact zero leaves their baseline unchanged.
            pieces=[torch.zeros_like(gate[:,:1]).expand(-1,25,-1) if corrections[c][e] is None else corrections[c][e] for c in range(3)]
            gate=gate+torch.cat([piece[:,:1] for piece in pieces],dim=1)
            raw=raw+torch.cat([piece[:,1:] for piece in pieces],dim=1)
            outputs.append((gate.transpose(1,2),raw.transpose(1,2).reshape(len(h),-1,3,3,8)))
        self.captured.clear()
        return routed_merge(outputs,logits)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if DEST.exists():raise RuntimeError('Fixed selection output already exists; no overwrite')
    parent=json.loads((SOURCE_DIR/'fold_5_report.json').read_text(encoding='utf-8'))
    reports={}
    for arm,name in ARMS.items():
        path=ROOT/'outputs'/name/'fold_5_report.json'
        if not path.exists():raise RuntimeError('All planned candidates must finish before selection: '+arm)
        report=json.loads(path.read_text(encoding='utf-8'))
        assert report['assembled_heads_exactly_reproduce_selected_predictions']
        reports[arm]=report
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=p.info;argv=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and 'pure_dl' in ' '.join(argv) and any(Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in argv):
                raise RuntimeError('Finish project trainers before final readout replay')
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    choices=[]
    for c,target in enumerate(TARGETS):
        best={'target':target,'arm':'source','score':parent['scores'][target],'epoch':52}
        # Deterministic tie rule: retain source if equally good; else fewer parameters first.
        order=sorted(ARMS,key=lambda arm:(reports[arm]['trainable_parameters'],arm))
        for arm in order:
            if reports[arm]['scores'][target]>best['score']+1e-10:
                best={'target':target,'arm':arm,'score':reports[arm]['scores'][target],'epoch':reports[arm]['best_epochs'][c]}
        choices.append(best)
    print('FIXED_CHOICES',json.dumps(choices,ensure_ascii=False),flush=True)
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    master=torch.load(SOURCE_DIR/'fold_5.pt',map_location='cpu',weights_only=False)
    data=prepare_data(master)
    numeric,conditioning=calibration_indices(master['selected_feature_names'])
    original=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=conditioning)
    original.load_state_dict(master['state_dict'],strict=True)
    original.cuda().eval().requires_grad_(False)
    donor_arms={choice['arm'] for choice in choices if choice['arm']!='source'}
    checkpoints={a:torch.load(ROOT/'outputs'/ARMS[a]/'heads_best.pt',map_location='cpu',weights_only=False) for a in donor_arms}
    model=FixedSelectedHeads(original,choices,checkpoints).cuda().eval().requires_grad_(False)
    evaluator=SourceEvaluator(data,master)
    with torch.no_grad():prediction,scores=evaluator(model)
    source_pred=np.load(SOURCE_DIR/'fold_5_pred.npy')
    expected=np.empty_like(source_pred)
    for c,choice in enumerate(choices):
        donor=source_pred if choice['arm']=='source' else np.load(ROOT/'outputs'/ARMS[choice['arm']]/'fold_5_pred.npy')
        expected[:,c]=donor[:,c]
    assert np.array_equal(prediction,expected),'Fixed shared-backbone readout assembly differs from chosen predictions'
    assert all(abs(scores[c]-choices[j]['score'])<1e-10 for j,c in enumerate(TARGETS))
    assert all(torch.equal(value.detach().cpu(),master['state_dict'][key]) for key,value in model.backbone.state_dict().items())
    DEST.mkdir()
    atomic_array(DEST/'fold_5_pred.npy',prediction)
    manifest={'source_checkpoint':str((SOURCE_DIR/'fold_5.pt').relative_to(ROOT)),
        'source_sha256':hashlib.sha256((SOURCE_DIR/'fold_5.pt').read_bytes()).hexdigest(),
        'choices':choices,'source_kinds':model.kinds,'selection_scope':'whole-target fold5 development selection; fixed across all rows/wells',
        'head_state_dicts':[{k:v.detach().cpu() for k,v in group.state_dict().items()} for group in model.heads]}
    atomic_torch(DEST/'selected_heads.pt',manifest)
    report={'fold':5,'screening_fold':5,'scores':scores,'choices':choices,'best_epochs':[x['epoch'] for x in choices],
        'delta_source':scores['TOTAL']-parent['scores']['TOTAL'],
        'delta_previous_local5':scores['TOTAL']-reports['local5']['scores']['TOTAL'],
        'assembled_heads_exactly_reproduce_selected_predictions':True,'frozen_source_unchanged':True,
        'all_validation_rows_covered_once':True,'backbone_branches_evaluated_once_per_window':2,
        'complete_five_folds':False,'screening_threshold':80.5,'screening_passed':scores['TOTAL']>80.5,
        'remaining_folds_started':False,'development_selection':True,'observed_at':datetime.now().astimezone().isoformat(),
        'source_readouts':{a:ARMS[a] for a in donor_arms},'implemented_new_training_this_script':False}
    atomic_json(DEST/'fold_5_report.json',report)
    atomic_json(DEST/'runtime_status.json',{'status':'complete','atlas_current_trial':False})
    description='# 双专家固定通道读出组合\n\n只选择整条目标通道的已完成读出，所有行/井使用同一配方；两源骨干每窗口只计算一次。原单覆盖/256格/800动作不变。\n\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n\n这是同一第五折反复开发后的固定组合，不是新的独立泛化证据。没有训练、补折或修改已交付模型。\n'
    (DEST/'README.md').write_text(description,encoding='utf-8')
    (DEST/'AGENTS.md').write_text('# 约束\n维护PROJECTS.md，记录目标、已做、未做、计划和完成后工作。固定组合不可覆盖；不按行/井选择不同模型，不自动补折，旧交付不改。\n',encoding='utf-8')
    (DEST/'PROJECTS.md').write_text('# 进度\n已完成固定通道选择与实际共同恢复。待独立泛化验证，完整OOF>82未完成；下一步按收益定位输出子结构，补折仍暂缓。\n',encoding='utf-8')
    with (ROOT/'EXPLORATIONS.md').open('a',encoding='utf-8') as f:
        f.write('\n- [双专家第二轮固定通道组合](outputs/'+DEST.name+'/README.md)：完整第五折共同回放已验证，开发集选模，未补折。\n')
    print('ASSEMBLY_VERIFIED',json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
