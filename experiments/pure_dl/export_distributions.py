"""Export neural held-out posteriors using saved fold-only preprocessing."""
from pathlib import Path
import argparse,sys,json
import numpy as np
import torch
from train import ROOT,Network,OFF,TS
from src.data import load_split
from src.feature import FeatureBuilder
from src.model import prep_well

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'outputs/pure_dl_v1');a=ap.parse_args()
torch.set_num_threads(4);rec=load_split(ROOT.parent/'data','train');o=np.load(a.run/'oof.npz');fold=o['fold'];well=o['well']
for f in range(5):
    dest=a.run/f'fold_{f+1}_distributions.npz'
    if dest.exists():continue
    ck=torch.load(a.run/f'fold_{f+1}.pt',map_location='cpu',weights_only=False)
    net=Network().cuda();net.load_state_dict(ck['state_dict']);net.eval()
    builder=FeatureBuilder();builder.medians_=ck['medians'];fm=np.asarray(ck['feature_median'],np.float32);fs=np.asarray(ck['feature_scale'],np.float32)
    states=[];probs={t:[] for t in TS};indices=np.flatnonzero(fold==f)
    for wi in sorted(set(well[indices])):
        frame=rec[wi][1];v=frame[TS].notna().all(1).to_numpy()
        base=builder._transform_frames(frame)[0].to_numpy(np.float32)
        x=np.concatenate([prep_well(frame,ck['gstats']),np.clip((base-fm)/fs,-8,8).T.astype(np.float32)])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):s,z=net(torch.from_numpy(x[None]).cuda())
        states.append(s.float().softmax(1)[0].T.cpu().numpy()[v].astype(np.float16))
        for j,t in enumerate(TS):probs[t].append(z.float()[0,OFF[j]:OFF[j+1]].softmax(0).T.cpu().numpy()[v].astype(np.float16))
    payload={'indices':indices,'state':np.concatenate(states)}
    for j,t in enumerate(TS):payload[t+'_prob']=np.concatenate(probs[t]);payload[t+'_centers']=np.array(ck['centers'][j])
    np.savez_compressed(dest,**payload);print(dest,'rows',len(indices),flush=True)
    del net;torch.cuda.empty_cache()
