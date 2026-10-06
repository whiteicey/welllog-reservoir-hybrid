"""Own-fold preprocessing, amplitude view and fixed source quality."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
from fullcv_context_v97 import prepare_fold_data, bind_fold_quality, add_well_bank, original_population, sha
from fullcv_models_v97 import fold_dir, root_path, root_dir
from fullcv_resume_v97 import atomic_save, atomic_npz


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3,4],required=True);args=ap.parse_args();f=args.fold
    out=fold_dir(f);out.mkdir(parents=True,exist_ok=True)
    if (out/'data.pt').exists():raise RuntimeError('Existing fold cache is immutable')
    torch.set_num_threads(3)
    master=torch.load(root_path(f),map_location='cpu',weights_only=False)
    pop=original_population();data=prepare_fold_data(master,f-1,pop)
    bind_fold_quality(data,master,root_dir(f)/f'fold_{f}_conflict_weights.npz')
    metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')}
    metadata.update(add_well_bank(data));metadata.update(sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]))
    curves=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT');train=data['population_rows']['train_wells'];raw=data['population_rows']['training_raw']
    med={n:float(np.nanmedian(np.concatenate([pop['records'][w][1][n].to_numpy(np.float64) for w in train]))) for n in curves}
    views=[]
    for _,frame in pop['records']:
        cols=[]
        for name in curves:
            s=frame[name].replace([np.inf,-np.inf],np.nan).interpolate(method='linear',limit_direction='both').fillna(med[name])
            values=s.to_numpy(np.float64)
            if name in ('GR','RXO','RT'):values=np.log1p(np.maximum(values,0))
            s=pd.Series(values)
            cols.append(values)
            cols.extend(s.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in (9,51,201))
            cols.extend(s.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9,51,201))
        views.append(np.stack(cols,-1).astype(np.float32))
    x=np.concatenate(views);q=np.quantile(x[raw],[.25,.5,.75],axis=0);center=q[1].astype(np.float32);scale=np.maximum(q[2]-q[0],1e-3).astype(np.float32)
    amplitude=np.clip((x-center)/scale,-8,8).astype(np.float16)
    atomic_npz(out/'amplitude.npz',amplitude=amplitude)
    manifest=dict(fold=f,root_sha256=sha(root_path(f)),train_wells=train,validation_wells=data['population_rows']['heldout_wells'],
        training_rows=len(raw),validation_rows=len(data['validation']),quality=data['qinfo'],amplitude_medians=med,amplitude_center=center.tolist(),amplitude_scale=scale.tolist(),labels_used_for_amplitude=False)
    (out/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    data={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
    metadata={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in metadata.items()}
    atomic_save(dict(data=data,metadata=metadata,root_sha256=sha(root_path(f))),out/'data.pt')
    print('FOLD_DATA_READY',json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
