"""Frozen-feature, matched affine versus SiLU RQS readouts on own-fold data."""
import hashlib,json,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_rqs_staging_v58 import ROOT,Network as Parent,load_data,Evaluator as LiveEvaluator
from quantile_density_v48 import cdf_logpdf,channel_objective
from train_dual8222_heads80 import atomic_json,atomic_torch,cpu_copy
from train_scaling_v21 import Evaluator as Grid
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
import train_invariant as base

STUDY=ROOT/'results/deep_learning/sw_cached_readout_v59_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_rqs_staging_v58_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_rqs_staging_v58_SW_headstart_fold5/model_best.pt'
CACHE=STUDY/'frozen_features.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('affine','silu')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Readout(nn.Module):
    def __init__(self,core,mode):
        super().__init__();self.core=core;self.mode=mode
        self.input=nn.Linear(128,64);self.output=nn.Linear(64,34)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,h):
        z=self.input(h)
        if self.mode=='silu':z=F.silu(z)
        return self.core(h)+self.output(z)

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('headstart',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261159);self.output=Readout(self.output,mode)
        for p in self.parameters():p.requires_grad_(False)
        for p in self.output.parameters():p.requires_grad_(True)
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v58 headstart function; original density plus zero-output affine/SiLU factorized branch',
            frozen='body/norm/FiLM/default gate',trainable='complete density readout, including original core and new branch')
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if not n.startswith('output.'))).hexdigest()

def load_cache():return torch.load(CACHE,map_location='cpu',weights_only=False)

class Evaluator:
    def __init__(self,cache,reference):
        self.cache=cache;self.reference=reference
        self.val={k:v.cuda() if torch.is_tensor(v) else v for k,v in cache['validation'].items()}
        self.edges=cache['edges'].cuda();self.utilities=cache['utilities'].cuda();self.actions=cache['actions']
    @torch.no_grad()
    def __call__(self,model):
        model.eval();v=self.val;result=np.full(len(self.reference),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for lo in range(0,len(v['h']),16):
            h=v['h'][lo:lo+16];mask=v['mask'][lo:lo+16];dest=v['dest'][lo:lo+16][mask].cpu().numpy()
            # Match the live head's B x1024 row layout; outer rows cannot influence pointwise readout.
            padded=torch.zeros((len(h),1024,128),device='cuda',dtype=h.dtype);padded[:,448:576]=h
            with torch.autocast('cuda',dtype=torch.bfloat16):raw=model.output(padded).float()[:,448:576][mask]
            cdf,_=cdf_logpdf(raw,self.edges[None].expand(len(raw),-1),'rqs')
            cp=(cdf[:,1:]-cdf[:,:-1]).clamp_min(0);cp=cp/cp.sum(-1,keepdim=True).clamp_min(1e-12)
            gate=v['gate'][lo:lo+16][mask][:,0].sigmoid();joint=torch.cat((gate[:,None],(1-gate[:,None])*cp),-1)
            assert (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()];coverage[dest]+=1
        assert (coverage==1).all() and np.isfinite(result).all()
        combined=self.reference.copy();combined[:,2]=result;scores=base.metric(self.cache['truth_validation'],combined)
        return result,float(scores['SW']),scores

@torch.no_grad()
def prepare():
    if CACHE.exists():raise FileExistsError('Cache is immutable; reuse it explicitly')
    print('CACHE_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    torch.set_num_threads(3);torch.manual_seed(20261010);torch.backends.cudnn.benchmark=False
    tick=time.perf_counter();d,meta=load_data();ref=reference_prediction();model=Network('affine',meta).cuda().eval()
    grid=Grid(d,meta,2,ref);captured={}
    hook=model.norm.register_forward_hook(lambda module,args,value:captured.update(h=value.detach()))
    packs={};population={}
    for split in ('train','validation'):
        centers=[];anchors=[]
        for wi,rows in enumerate(d['row_ids']):
            isval=d['fold'][rows[0]]==4
            if isval!=(split=='validation'):continue
            cc=np.arange(d['starts'][wi],d['starts'][wi+1],128)+64
            centers.extend(cc);anchors.extend(np.minimum(cc,d['starts'][wi+1]-1))
        chunks={k:[] for k in ('h','gate','mask','dest','raw','y','default','quality')}
        for lo in range(0,len(centers),16):
            cc=torch.tensor(centers[lo:lo+16],device='cuda');aa=torch.tensor(anchors[lo:lo+16],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=1024,anchors=aa)
            relative=torch.tanh((d['depth'][ix]-d['depth'][aa,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):go,_=model(d['x'][ix],relative,valid,well_inputs=d['well_bank'][d['row_well'][aa]])
            raw=ix[:,448:576];mask=valid[:,448:576]&d['labeled'][raw]
            rows=d['raw_to_valid'][raw.cpu().numpy()]
            if split=='validation':mask=valid[:,448:576]&torch.as_tensor(rows>=0,device='cuda')
            dest=np.full(rows.shape,-1,np.int64)
            if split=='validation':dest[rows>=0]=grid.local[rows[rows>=0]]
            values=dict(h=captured['h'][:,448:576],gate=go[:,448:576],mask=mask,dest=torch.from_numpy(dest),raw=raw,
                y=d['labels'][raw,2],default=d['default'][raw,2],quality=d['quality'][raw])
            for k,v in values.items():chunks[k].append(v.detach().cpu().clone())
        pack={k:torch.cat(v) for k,v in chunks.items()}
        population[split]=pack['raw'][pack['mask']].numpy()
        if split=='train':
            keep=pack['mask'].any(1);pack={k:v[keep] for k,v in pack.items()}
        packs[split]=pack
        print('CACHE_SPLIT',split,len(pack['h']),int(pack['mask'].sum()),str(pack['h'].dtype),flush=True)
    hook.remove()
    train_raw=d['train_raw'].cpu().numpy()
    assert np.array_equal(np.sort(population['train']),np.sort(train_raw))
    dest=packs['validation']['dest'][packs['validation']['mask']].numpy()
    assert np.array_equal(np.sort(dest),np.arange(len(ref)))
    assert not np.intersect1d(population['train'],population['validation']).size
    manifest=dict(training_rows=len(population['train']),validation_rows=len(population['validation']),training_blocks=len(packs['train']['h']),
        validation_blocks=len(packs['validation']['h']),window_length=1024,center_points=128,train_context='fixed canonical nonoverlapping center128 blocks, unlike historical random-anchor windows',
        fit='no fitting; frozen own-fold source input forward only',labels_scope='training and validation stored in separate packs; only train pack sampled',
        source_checkpoint=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),core_sha256=model.core_hash(),
        training_raw_sha256=hashlib.sha256(np.sort(population['train']).tobytes()).hexdigest(),h_dtype=str(packs['train']['h'].dtype),
        parameter_count=sum(p.numel() for p in model.parameters()),active_density_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        bank_manifest=meta['bank_manifest'])
    cache=dict(**packs,metadata=cpu_copy(meta),edges=grid.edges.cpu(),utilities=grid.utilities.cpu(),actions=grid.actions,
        truth_validation=d['truth'][d['validation']],manifest=manifest)
    pred,score,_=Evaluator(cache,ref)(model)
    assert np.array_equal(pred,np.load(SOURCE.with_name('channel_pred.npy')))
    manifest.update(initial_full144665_parent_exact=True,initial_score=score,seconds=time.perf_counter()-tick,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
    atomic_torch(CACHE,cache);manifest['cache_bytes']=CACHE.stat().st_size;manifest['cache_sha256']=hashlib.sha256(CACHE.read_bytes()).hexdigest()
    atomic_json(STUDY/'cache_manifest.json',manifest)
    print('CACHE_COMPLETE',json.dumps({k:v for k,v in manifest.items() if k!='bank_manifest'}),flush=True)
