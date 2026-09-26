"""Standalone five-fold neural inference. No training data or labels are needed."""
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import contextlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data import INPUT_COLUMNS, load_split
from src.feature import FeatureBuilder
from src.utils import DEFAULTS, BOUNDS, kernel_matrix, make_rows
from src.model import Network, window_indices, retained_indices, calibration_indices
from src.probability import probability

def load_inputs(path):
    from src.data import read_curve
    path=Path(path)
    directory=path/'test' if (path/'test').is_dir() else path
    files=sorted(directory.glob('*.txt'))
    if not files: raise FileNotFoundError(f'No test TXT files found: {directory}')
    return [(p.stem,read_curve(p)) for p in files]


TARGETS = ['POR', 'PERM', 'SW']


class Predictor:
    def __init__(self, model_dir, folds, device):
        self.device = torch.device(device)
        self.checkpoints = [torch.load(Path(model_dir)/f'fold_{f}.pt', map_location='cpu', weights_only=False) for f in folds]
        self.models = []
        self.builders = []
        contract = json.loads((Path(model_dir) / 'feature_contract.json').read_text(encoding='utf-8'))
        for ck in self.checkpoints:
            view=retained_indices(contract['whitelist'])
            numeric,conditioning=calibration_indices(contract['whitelist'])
            assert ck['config']['well_view']['indices']==view
            assert ck['config']['input_calibration']['numeric_indices']==numeric
            assert ck['config']['input_calibration']['conditioning_indices']==conditioning
            model = Network(retain_indices=view,numeric_indices=numeric,conditioning_indices=conditioning).to(self.device).eval()
            model.load_state_dict(ck['state_dict'], strict=True)
            self.models.append(model)
            builder = FeatureBuilder()
            builder.medians_ = ck['medians']
            # Schema is rebuilt from the frozen full feature contract, never fitted.
            builder.enhanced_feature_names_ = contract['full_schema']
            self.builders.append(builder)
        self.actions = []
        self.utilities = []
        for j, name in enumerate(TARGETS):
            lo = min(float(c['centers'][j][0]) for c in self.checkpoints)
            hi = max(float(c['centers'][j][-1]) for c in self.checkpoints)
            grid = np.linspace(lo, hi, 800) if j == 2 else np.geomspace(max(lo*.9, .001), hi*1.1, 800)
            self.actions.append(np.unique(np.r_[DEFAULTS[name], np.clip(grid, *BOUNDS[name])]))
        for ck in self.checkpoints:
            self.utilities.append([torch.tensor(kernel_matrix(self.actions[j], np.r_[DEFAULTS[name], ck['centers'][j]], name).T,
                                                dtype=torch.float32, device=self.device)
                                   for j, name in enumerate(TARGETS)])

    @torch.inference_mode()
    def predict_well(self, frame, batch_windows=32):
        frame = frame.loc[:, INPUT_COLUMNS].copy()  # Labels never enter prediction.
        n = len(frame)
        if n==0: raise ValueError('Empty well')
        depth = torch.as_tensor(frame['DEPTH'].to_numpy(np.float32), device=self.device)
        if not torch.isfinite(depth).all(): raise ValueError('Nonfinite depth')
        xs = []
        absolutes = []
        for ck, builder in zip(self.checkpoints, self.builders):
            # Recover base schema deterministically; this does not fit statistics.
            base_frame, enhanced_frame = builder._transform_frames(frame)
            builder.base_feature_names_ = list(base_frame.columns)
            if list(enhanced_frame.columns) != builder.enhanced_feature_names_: raise ValueError('Feature schema mismatch')
            values = builder.transform(frame).enhanced[:, ck['selected_feature_indices']]
            if [builder.enhanced_feature_names_[i] for i in ck['selected_feature_indices']] != ck['selected_feature_names']:
                raise ValueError('Selected feature mismatch')
            values = np.clip((values-np.asarray(ck['feature_median'],np.float32))/np.asarray(ck['feature_scale'],np.float32), -8, 8).astype(np.float16)
            xs.append(torch.from_numpy(values).to(device=self.device,dtype=torch.float16 if self.device.type=='cuda' else torch.float32))
            absolutes.append(((depth-ck['absolute_depth_median'])/ck['absolute_depth_scale']).clamp(-8,8))
        lower = torch.zeros(n, dtype=torch.long, device=self.device)
        upper = torch.full((n,), n, dtype=torch.long, device=self.device)
        centers = torch.arange(0, n, 128, device=self.device)+64
        anchors = centers.clamp_max(n-1)
        predictions = np.empty((n, 3), np.float64)
        covered = np.zeros(n, np.uint8)
        for start in range(0, len(centers), batch_windows):
            cc, aa = centers[start:start+batch_windows], anchors[start:start+batch_windows]
            ix, exists = window_indices(cc, lower, upper, anchors=aa)
            valid = exists[:,64:192]
            rows = ix[:,64:192][valid].cpu().numpy()
            relative = torch.tanh((depth[ix]-depth[aa,None])/10)
            total = [torch.zeros((len(rows),len(a)),device=self.device) for a in self.actions]
            for k, (model, ck) in enumerate(zip(self.models,self.checkpoints)):
                amp = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else contextlib.nullcontext()
                with amp:
                    gate, raw = model(xs[k][ix], relative, exists, absolute_depth=absolutes[k][ix])
                gate = gate[:,64:192][valid].float().sigmoid()
                raw = raw[:,64:192][valid].float()
                weight, mu, sd = raw[:,:,0].softmax(-1), raw[:,:,1], F.softplus(raw[:,:,2])+.025
                for j in range(3):
                    edges = torch.as_tensor(ck['density_edges'][j],dtype=torch.float32,device=self.device)
                    cp = probability(j,weight[:,j],mu[:,j],sd[:,j],edges)
                    joint = torch.cat([gate[:,j:j+1],(1-gate[:,j:j+1])*cp],-1)
                    total[j] += (joint @ self.utilities[k][j])/len(self.models)
            assert not covered[rows].any() and len(np.unique(rows)) == len(rows)
            for j in range(3): predictions[rows,j] = self.actions[j][total[j].argmax(-1).cpu().numpy()]
            covered[rows] += 1
        assert (covered == 1).all() and np.isfinite(predictions).all()
        return predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir','--data-dir',dest='data_dir',type=Path,required=True,help='Data root containing test/, or direct test TXT directory')
    p.add_argument('--model_dir','--model-dir',dest='model_dir',type=Path,default=ROOT/'models')
    p.add_argument('--output',type=Path,default=ROOT/'result.json')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--batch_windows','--batch-windows',dest='batch_windows',type=int,default=32)
    args=p.parse_args()
    if args.output.suffix.lower() != '.json': raise ValueError('--output must be a JSON file path')
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.batch_windows<1: raise ValueError('batch-windows must be positive')
    started=time.perf_counter()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    predictor=Predictor(args.model_dir,[1,2,3,4,5],device)
    result_data=[]; total=0
    for well, frame in load_inputs(args.data_dir):
        pred=predictor.predict_well(frame,args.batch_windows)
        result_data.append({'logId':well,'predictions':make_rows(frame['DEPTH'].to_numpy(),pred,well)})
        total+=len(frame); print(f'PREDICTED {well} rows={len(frame)}',flush=True)
    result={'resultData':result_data}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    result_zip = args.output.parent/'result.zip'
    with zipfile.ZipFile(result_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.write(args.output,arcname='result.json')
    summary={'wells':len(result_data),'rows':total,'seconds':time.perf_counter()-started,'device':device,
             'folds':[1,2,3,4,5],'deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'aggregation':'equal two-member full posterior within each fold, equal five-fold expected utilities before action selection',
             'test_accuracy':'unknown: test labels unavailable',
             'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None}
    args.output.with_suffix('.inference.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
