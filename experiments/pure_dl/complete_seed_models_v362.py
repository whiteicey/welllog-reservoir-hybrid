"""Isolated complete eight-stage seed recipe; historical modules stay unchanged."""
import hashlib,importlib,inspect,json,os,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/complete_seed_compare_v362_20261002'
SEEDS=(20261010,20261011,20261012,20261013,20261014)
SEED=int(os.environ['WELLLOG_RUN_SEED']);assert SEED in SEEDS
OUT=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'/('seed_'+str(SEED))
SOURCE=OUT/'SOURCE/fold_5.pt'
SPECS={
 'POR113':('compact_por_mixed_family_v113','t3_cauchy','SOURCE'),
 'POR151':('por_ranked_probability_v151','joint_rps','POR113'),
 'PERM154':('perm_ranked_probability_v154','perm_conditional','SOURCE'),
 'SW104':('compact_frequency_v104','local','SOURCE'),
 'POR341':('por_coupling_conditional_rps_v341','conditional','POR151'),
 'PERM336':('perm_low_rank_cross_v336','cross','PERM154'),
 'SW351':('sw_dynamic_cross_spectrum_v351','aligned','SW104'),
}
MODULES={k:importlib.import_module(v[0]) for k,v in SPECS.items()}
STAGES={k:MODULES[k].STAGES[v[1]] for k,v in SPECS.items()}
PARENT=SOURCE
_load=torch.load;_manual_seed=torch.manual_seed
TRACE=[]

def seed_internal_initializers(seed):
 caller=Path(sys._getframe(1).f_code.co_filename).name
 if caller not in ('train_complete_seed_stage_v362.py','train_complete_seed_source_v362.py'):
  seed=int(seed)+SEED-20261010
 return _manual_seed(seed)

def bind_paths():
 paths={
  ROOT/'outputs/pure_dl_paired_local_context_router80_sequence/fold_5.pt':SOURCE,
  ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt':OUT/'POR113/model_best.pt',
  ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt':OUT/'POR151/model_best.pt',
  ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt':OUT/'PERM154/model_best.pt',
  ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt':OUT/'SW104/model_best.pt',
 }
 paths={k.resolve():v.resolve() for k,v in paths.items()}
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if not f or not Path(f).resolve().is_relative_to(ROOT/'experiments/pure_dl'):continue
  for name in ('SOURCE','PARENT'):
   value=getattr(m,name,None)
   if isinstance(value,Path) and value.resolve() in paths:setattr(m,name,paths[value.resolve()])
  parent=getattr(m,'PARENT',None)
  if hasattr(m,'PARENT_SHA') and isinstance(parent,Path) and parent.is_relative_to(OUT):
   if parent.exists():m.PARENT_SHA=hashlib.sha256(parent.read_bytes()).hexdigest()
 def guarded_load(path,*a,**kw):
  if isinstance(path,(str,Path)):
   p=Path(path).resolve();p=paths.get(p,p)
   if p.suffix=='.pt' and p.is_relative_to(ROOT/'outputs'):
    if not p.is_relative_to(OUT):raise RuntimeError('Another seed or historical learned checkpoint forbidden: '+str(p))
    TRACE.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   return _load(p,*a,**kw)
  return _load(path,*a,**kw)
 torch.load=guarded_load;torch.manual_seed=seed_internal_initializers
 import champion_scaling_v21
 def own_data():
  data=torch.load(OUT/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
  for k,v in list(data.items()):
   if torch.is_tensor(v):data[k]=v.cuda()
  return data,torch.load(SOURCE,map_location='cpu',weights_only=False)
 champion_scaling_v21.load_data=own_data

def load_data(stage):return MODULES[stage].load_data(SPECS[stage][1])
def make_model(stage,master,metadata):return MODULES[stage].make_model(SPECS[stage][1],master,metadata)
def Evaluator(data,master,c,length,stage):
 if stage=='SW104':
  from train_stage_fullcv_v97 import Evaluator as E
  return E(data,master,c,length)
 return MODULES[stage].Evaluator(data,master,c,length,SPECS[stage][1])
def channel_objective(output,labels,default,mask,quality,c,stage,model,well_ids):
 fn=MODULES[stage].channel_objective;params=inspect.signature(fn).parameters
 args=[output,labels,default,mask,quality,c,SPECS[stage][1]]
 if 'model' in params:args.append(model)
 kwargs={'well_ids':well_ids} if 'well_ids' in params else {}
 return fn(*args,**kwargs)
def DecisionObjective(evaluator,metadata,stage):
 if stage=='SW104':
  from sw_utility_strength_v82 import DecisionObjective as D
  return D(evaluator,metadata,'a100')
 return MODULES[stage].DecisionObjective(evaluator,metadata,SPECS[stage][1])

