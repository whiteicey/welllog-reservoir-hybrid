"""CPU reconstruction of a best0 initializer; no forward, optimizer or GPU work."""
from pathlib import Path
import hashlib, json, os, random, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
seed, stage = int(sys.argv[1]), sys.argv[2]
assert stage in ('PERM154', 'POR151')
os.environ['WELLLOG_RUN_SEED'] = str(seed)
sys.path.insert(0, str(ROOT/'experiments/pure_dl'))
import complete_seed_models_v362r1 as recipe

own = recipe.OUT/stage
report = json.loads((own/'report.json').read_text(encoding='utf-8'))
assert report['best_epoch'] == 0, 'Only observed best0 stages are candidates here'
recipe.bind_paths()
# Mirror the worker's top-level seed without applying the constructor offset twice.
recipe._manual_seed(seed); np.random.seed(seed); random.seed(seed)
master = torch.load(recipe.SOURCE, map_location='cpu', weights_only=False)
model = recipe.make_model(stage, master, {})
initialization = model.initialize_priors({})
dependencies = list(recipe.TRACE)
assert all(Path(x['path']).resolve() != (own/'model_best.pt').resolve() for x in dependencies)
fresh = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
selected = torch.load(own/'model_best.pt', map_location='cpu', weights_only=False)['state_dict']
missing = sorted(set(selected)-set(fresh)); extra = sorted(set(fresh)-set(selected))
different = [k for k in fresh.keys() & selected.keys()
             if fresh[k].dtype != selected[k].dtype or fresh[k].shape != selected[k].shape
             or not torch.equal(fresh[k], selected[k].cpu())]
def digest(state):
    h=hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()
result = dict(seed=seed, stage=stage, selected_best_epoch=0,
              initializer_state_exact=not (missing or extra or different),
              state_tensors=len(fresh), missing=missing, extra=extra,
              differing_tensors=different, fresh_state_sha256=digest(fresh),
              selected_state_sha256=digest(selected), initialization=initialization,
              construction_dependencies=dependencies, CPU_only=True,
              optimizer_steps=0, model_forward_calls=0, extra_training_stages=0,
              historical_executed_stage_count_unchanged=8,
              conclusion_scope='Initializer equivalence only. Supports future omission of this optimizer phase while retaining its untrained architecture; existing historical training cost remains eight phases. No descendant retraining or GPU predictive replay performed.')
(HERE/f'SEED{seed}_{stage}_INITIALIZER_EQUIVALENCE.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
