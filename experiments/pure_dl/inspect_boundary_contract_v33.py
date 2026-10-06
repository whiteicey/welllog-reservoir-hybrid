"""Read-only data-contract inspection for the v33 boundary study."""
from pathlib import Path
import json
import torch

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/boundary_context_v33_20260929'

def main():
    torch.set_num_threads(2)
    d=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    fields={k:dict(type=type(v).__name__,shape=list(v.shape) if hasattr(v,'shape') else None) for k,v in d.items()}
    print('CACHE_FIELDS',json.dumps(fields))
    for k in ['train_raw','val_raw','validation_raw','fold','wells','truth','lower','upper','labeled','valid_raw']:
        if k in d: print('SAMPLE',k,str(d[k][:4]))
    meta=json.loads((ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json').read_text(encoding='utf-8'))
    print('FEATURE_META',json.dumps({k:meta[k] for k in ['fit_wells','monitor_wells','raw_points','fit_points','monitor_points','imputation_medians','profile']}))
    (STUDY/'data_contract.json').write_text(json.dumps(fields,indent=2),encoding='utf-8')

if __name__=='__main__':main()
