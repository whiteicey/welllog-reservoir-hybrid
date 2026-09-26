"""Independent reconstruction of input-only external summaries from raw curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_invariant as base


def verify(out):
    ck=torch.load(out/'fold_5.pt',map_location='cpu',weights_only=False)
    bank=np.load(out/'fold_5_external_bank.npz')
    rec=base.load_split(base.ROOT.parent/'data','train')
    names=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']
    selected=[ck['selected_feature_names'].index(x) for x in names]
    fm=np.asarray(ck['feature_median'])[selected];fs=np.asarray(ck['feature_scale'])[selected]
    offset=0;checked=0;max_error=0.
    for w,(_,frame) in enumerate(rec):
        raw=frame[names].to_numpy(np.float32);depth=frame['DEPTH'].to_numpy(np.float32)
        normalized=np.clip((raw-fm)/fs,-8,8).astype(np.float16).astype(np.float32)
        n=len(frame);blocks=(n+127)//128
        for b,a in enumerate(range(0,n,128)):
            end=min(a+128,n);seen=np.isfinite(raw[a:end]);data=normalized[a:end]
            cnt=seen.sum(0);mean=np.where(seen,data,0).sum(0)/cnt.clip(1)
            var=np.where(seen,(data-mean)**2,0).sum(0)/cnt.clip(1)
            center=depth[a:end].mean()
            expected=np.r_[mean,np.sqrt(var),1-cnt/(end-a),(end-a)/128,
                           (center-depth[0])/max(float(depth[-1]-depth[0]),1e-6),
                           np.log1p(abs(float(depth[end-1]-depth[a]))/10)]
            error=float(np.max(np.abs(expected-bank['features'][w,b])))
            max_error=max(max_error,error);assert error<.004,(w,b,error)
            assert bank['lower'][w,b]==offset+a and bank['upper'][w,b]==offset+end
            assert abs(float(bank['centers'][w,b])-float(center))<1e-4
            assert bool(bank['valid'][w,b])==bool(seen.any())
            checked+=1
        assert not bank['valid'][w,blocks:].any()
        offset+=n
    result={'passed':True,'wells':len(rec),'patches':checked,'max_raw_reconstruction_error':max_error,
            'scope':'All 128-row patches independently reconstructed from nine raw curves, train-fold input scaling and depths. No target values read into patch statistics. Runtime same-well/overlap behavior separately tested.'}
    (out/'external_bank_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    verify(p.parse_args().out)
