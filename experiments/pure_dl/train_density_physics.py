"""One POR-only neural density experiment with a matrix-density mean constraint.

The rho branch cannot see any density-derived input. Observed DEN enters the
specified physical mean formula only; the gate/mixture/scale branch may use DEN.
Other targets are SHA-locked pure-neural OOF placeholders, never truth.
"""
from pathlib import Path
import argparse,gc,hashlib,json,random,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split
from src.feature import FeatureBuilder
from src.utils import DEFAULTS,BOUNDS,kernel_matrix
from train import metric,write,TS
from train_invariant import FEATURES,SEGMENTS,PADDED_ENCODED,NumericEncoding,prepare_features,feature_contract

MIX=8;SIGMA_FLOOR=.025;DENSITY_BINS=256


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_contract():
    base=feature_contract();names=base['whitelist']
    forbidden=[i for i,n in enumerate(names) if 'DEN' in n.split('_') or 'DENSITY' in n.split('_') or n=='ARCHIE_PROXY']
    keep=[i for i in range(FEATURES) if i not in forbidden]
    # Independent input-level dependency check: changing raw DEN must not change
    # any rho input, including indirect density-porosity/Archie constructions.
    t=np.linspace(0,4,128)
    frame=pd.DataFrame({'DEPTH':1400+np.arange(128)*.5,'GR':60+20*np.sin(t),'PE':3+.2*np.cos(t),
        'SP':-30+np.sin(t),'CAL':22+.3*np.cos(t),'AC':230+20*np.sin(t),'DEN':2.45+.1*np.cos(t),
        'CNL':18+3*np.sin(t),'RXO':20+3*np.cos(t),'RT':60+20*np.sin(t),
        'DEVI':5+np.sin(t),'AZIM':120+10*np.sin(t),'BIT':np.full(128,21.59),'CASE':np.nan})
    builder=FeatureBuilder().fit([frame]);changed=frame.copy();changed['DEN']+=.12
    a=builder.transform(frame).enhanced[:,base['selected_indices']]
    b=builder.transform(changed).enhanced[:,base['selected_indices']]
    assert np.array_equal(a[:,keep],b[:,keep])
    assert not np.array_equal(a[:,names.index('DEN')],b[:,names.index('DEN')])
    return {'full_input_contract':base,'rho_feature_indices':keep,'rho_feature_whitelist':[names[i] for i in keep],
        'excluded_from_rho':[names[i] for i in forbidden],
        'rho_formula':'2.65 + 0.6*tanh(raw)', 'por_formula':'softplus(100*(rho-DEN)/(rho-1), beta=2)+1e-6',
        'fluid_density':1.,'rho_bounds':[2.05,3.25],'por_transform':'log1p',
        'synthetic_DEN_perturbation_keeps_rho_inputs_identical':True,
        'real_data_read_for_contract':False,
        'branch_rule':'rho only sees non-DEN-derived features. Gate, weights and sigma may see full206; observed imputed unnormalized DEN enters POR mean formula.'}


class Block(nn.Module):
    def __init__(self,c=128):
        super().__init__();self.norm=nn.LayerNorm(c)
        self.net=nn.Sequential(nn.Linear(c,c),nn.GELU(),nn.Dropout(.05),nn.Linear(c,c))
        self.scale=nn.Parameter(torch.full((c,),.1))
    def forward(self,x):return x+self.scale*self.net(self.norm(x))


def branch(cin):return nn.Sequential(nn.Linear(cin,128),nn.GELU(),Block(),Block(),Block(),nn.LayerNorm(128))


class Network(nn.Module):
    def __init__(self,contract,edges=None,median=2.5,scale=.5):
        super().__init__()
        if edges is None:edges=np.tile(np.linspace(-3,3,SEGMENTS+1),(FEATURES,1))
        self.encoding=NumericEncoding(edges)
        rho_ids=contract['rho_feature_indices']
        encoded_ids=rho_ids+[FEATURES+i*SEGMENTS+j for i in rho_ids for j in range(SEGMENTS)]
        self.register_buffer('rho_encoded_indices',torch.tensor(encoded_ids,dtype=torch.long))
        self.rho_pad=(-len(encoded_ids))%8
        self.full=branch(PADDED_ENCODED)
        self.rho_net=branch(len(encoded_ids)+self.rho_pad)
        self.gate=nn.Linear(128,1);self.mix=nn.Linear(128,MIX);self.std=nn.Linear(128,MIX)
        self.rho_head=nn.Linear(128,MIX)
        nn.init.normal_(self.rho_head.weight,0,.002)
        with torch.no_grad():
            self.rho_head.bias.copy_(torch.linspace(-.08,.08,MIX))
            self.std.weight.zero_();self.std.bias.fill_(float(np.log(np.expm1(.5-SIGMA_FLOOR))))
        self.register_buffer('por_median',torch.tensor(median,dtype=torch.float32))
        self.register_buffer('por_scale',torch.tensor(scale,dtype=torch.float32))
    def forward(self,x,den):
        encoded=self.encoding(x)
        full=self.full(encoded)
        rho_encoding=F.pad(encoded.index_select(1,self.rho_encoded_indices),(0,self.rho_pad))
        rho=2.65+.6*torch.tanh(self.rho_head(self.rho_net(rho_encoding)).float())
        phi_raw=100*(rho-den.float()[:,None])/(rho-1.)
        phi=F.softplus(phi_raw,beta=2.)+1e-6
        mu=(torch.log1p(phi)-self.por_median)/self.por_scale
        raw=torch.stack([self.mix(full).float(),mu,self.std(full).float()],1)
        return self.gate(full).float()[:,0],raw,rho


def mixture_params(raw):return raw[:,0].log_softmax(-1),raw[:,1],F.softplus(raw[:,2])+SIGMA_FLOOR


def step(model,opt,x,den,gates,targets,indices,batch):
    rows=indices[torch.randint(len(indices),(batch,),device='cuda')]
    tg=gates[rows];tz=targets[rows]
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):go,raw,rho=model(x[rows],den[rows])
    lw,mu,sd=mixture_params(raw)
    nll=-torch.logsumexp(lw-.5*((tz[:,None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi),-1)
    gl=F.binary_cross_entropy_with_logits(go,tg);nl=nll[tg<.5].mean();loss=gl+nl
    assert torch.isfinite(loss)
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    return np.array([float(loss.detach()),float(gl.detach()),float(nl.detach())])


def benchmark(args):
    contract=physical_contract();args.out.mkdir(parents=True,exist_ok=True);write(args.out/'feature_contract.json',contract)
    torch.manual_seed(args.seed);model=Network(contract).cuda()
    # Verify computational separation in addition to the raw-feature check.
    model.eval();xx=torch.randn(8,FEATURES,device='cuda',requires_grad=True);dd=torch.full((8,),2.45,device='cuda')
    _,_,rho=model(xx,dd);grad=torch.autograd.grad(rho.sum(),xx)[0]
    forbidden=[i for i in range(FEATURES) if i not in contract['rho_feature_indices']]
    assert torch.count_nonzero(grad[:,forbidden])==0 and torch.isfinite(grad).all()
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    x=torch.randn(723556,FEATURES,device='cuda',dtype=torch.float16).clamp(-8,8)
    den=(2.45+.15*torch.randn(723556,device='cuda')).clamp(1.5,3.5)
    gates=(torch.rand(723556,device='cuda')<.7).float();targets=torch.randn(723556,device='cuda')
    indices=torch.arange(578000,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(8):step(model,opt,x,den,gates,targets,indices,args.batch_points)
    torch.cuda.synchronize();started=time.perf_counter()
    for _ in range(40):step(model,opt,x,den,gates,targets,indices,args.batch_points)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    projected=elapsed/40*args.epochs*args.steps*5
    result={'benchmark_only':True,'real_data_read':False,'synthetic_updates':48,'timed_updates':40,
        'parameters':sum(p.numel() for p in model.parameters()),'full_features':FEATURES,'rho_features':len(contract['rho_feature_indices']),
        'rho_gradient_wrt_DEN_features_exactly_zero':True,'synthetic_DEN_input_invariance_passed':True,
        'mean_step_ms':elapsed/40*1000,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'projected_five_fold_updates_seconds':projected,'projected_total_seconds_range':[projected+130,projected+210],
        'timing_note':'Synthetic projection, not completed training; concurrent workload can change runtime.'}
    write(args.out/'benchmark.json',result);print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def probability(weight,mu,sd,edges):
    cdf=torch.special.ndtr((edges[None,:,None]-mu[:,None,:])/sd[:,None,:])
    p=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*weight[:,None,:]).sum(-1)
    return p/p.sum(1,keepdim=True).clamp_min(1e-12)


def train(args):
    start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=True)
    contract=physical_contract();cp=args.out/'feature_contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract
    else:write(cp,contract)
    reference_sha=digest(args.reference);reference=np.load(args.reference)
    rec=load_split(ROOT.parent/'data','train');ys=[fr[TS].to_numpy(np.float64) for _,fr in rec]
    valid=[np.isfinite(v).all(1) for v in ys]
    y=np.concatenate([v[m] for v,m in zip(ys,valid)])
    well=np.concatenate([np.full(m.sum(),i,np.int32) for i,m in enumerate(valid)])
    assert len(rec)==80 and y.shape==(723556,3)
    fold=np.full(len(y),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(y,groups=well)):fold[ix]=f
    assert np.array_equal(y,reference['truth']) and np.array_equal(well,reference['well']) and np.array_equal(fold,reference['fold'])
    fixed=reference['pred'].copy();pred=fixed.copy();pred[:,0]=np.nan
    row_ids=[np.flatnonzero(well==i) for i in range(80)]
    wf={i:int(fold[row_ids[i][0]]) for i in range(80)}
    config={**vars(args),'out':str(args.out),'reference':str(args.reference),'reference_sha256':reference_sha,
        'source_sha256':digest(Path(__file__)),'preprocessing_source_sha256':digest(ROOT/'experiments/pure_dl/train_invariant.py'),
        'feature_contract_sha256':digest(cp),'parameters':sum(p.numel() for p in Network(contract).parameters()),
        'architecture':'POR-only PLE16 + two independent width128 three-block residual MLPs. Full206 features for gate/weights/sigma; density-free whitelist for8 rho means.',
        'rho_formula':contract['rho_formula'],'por_formula':contract['por_formula'],'fluid_density':1.,
        'rho_features':len(contract['rho_feature_indices']),'target':'POR log1p','sigma_floor':SIGMA_FLOOR,
        'loss':'POR default BCE + nondefault POR Gaussian-mixture NLL; no extra physical/rho-based row exclusion',
        'decode':'fixed256 transformed support centers +800 actions + exact official kernel/default atom',
        'non_POR_outputs':'exact SHA-locked PERM/SW MDN analytic800 OOF predictions, never true labels',
        'encoding_fit':'input preprocessing and PLE knots fit only64 outer-training wells',
        'target_statistics_fit':'outer-training nondefault original POR labels only',
        'selection':'one predeclared40x64 final-epoch run; no outer-fold calibration/threshold/architecture tuning',
        'test_access':False,'valid_rows':len(y),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'outer_wells':{str(f):[rec[i][0] for i in range(80) if wf[i]==f] for f in range(5)}}
    config_path=args.out/'config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text())
        for key in ['source_sha256','reference_sha256','epochs','steps','batch_points','seed']:assert old[key]==config[key],key
    write(config_path,config);print('CONFIG',json.dumps({k:v for k,v in config.items() if k!='outer_wells'}),flush=True)
    reports=[];default=y[:,0]<=.1
    for f in range(5):
        fs_time=time.perf_counter();seed=args.seed+f
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cache=args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            pred[fold==f]=np.load(cache);reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()));continue
        tr=[i for i in range(80) if wf[i]!=f];assert len(tr)==64
        print(f'FOLD {f+1} fitting only64 training input wells and original nondefault POR labels',flush=True)
        x,builder,fm,fs,edges=prepare_features(rec,tr,valid,row_ids)
        density=np.concatenate([fr['DEN'].interpolate(limit_direction='both').fillna(builder.medians_['DEN']).to_numpy(dtype=np.float32)[m] for (_,fr),m in zip(rec,valid)])
        assert np.isfinite(density).all();den=torch.tensor(density,device='cuda')
        training=np.log1p(y[(fold!=f)&~default,0]);q25,median,q75=np.percentile(training,[25,50,75]);scale=max(float(q75-q25),.1)
        centers_z=np.linspace(training.min(),training.max(),DENSITY_BINS);centers=np.expm1(centers_z)
        density_edges=(np.r_[-np.inf,(centers_z[:-1]+centers_z[1:])/2,np.inf]-median)/scale
        targets=torch.tensor((np.log1p(y[:,0])-median)/scale,dtype=torch.float32,device='cuda')
        gates=torch.tensor(default,dtype=torch.float32,device='cuda')
        indices=torch.tensor(np.flatnonzero(fold!=f),dtype=torch.long,device='cuda')
        preprocessing={'fold':f+1,'training_wells':[rec[i][0] for i in tr],'validation_wells':config['outer_wells'][str(f)],
            'training_rows':int(np.sum(fold!=f)),'original_nondefault_POR_training_rows':len(training),
            'no_extra_physical_row_filter':True,'DEN_source':'observed DEN, same input range cleaning; within-well interpolation then train-only median; no normalized clipping',
            'rho_features':contract['rho_feature_whitelist'],'feature_names':contract['full_input_contract']['whitelist']}
        assert not set(preprocessing['training_wells'])&set(preprocessing['validation_wells'])
        write(args.out/f'fold_{f+1}_preprocessing.json',preprocessing)
        model=Network(contract,edges,median,scale).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs*args.steps,eta_min=.00001)
        history=[];torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train();losses=np.zeros(3)
            for _ in range(args.steps):
                losses+=step(model,opt,x,den,gates,targets,indices,args.batch_points);scheduler.step()
            history.append((losses/args.steps).tolist())
            if epoch==0 or (epoch+1)%5==0:
                print(f'fold={f+1} epoch={epoch+1}/{args.epochs} loss={history[-1][0]:.4f} gate={history[-1][1]:.4f} nll={history[-1][2]:.4f} seconds={time.perf_counter()-fs_time:.1f}',flush=True)
        model.eval();grid=np.unique(np.r_[DEFAULTS['POR'],np.clip(np.geomspace(max(centers[0]*.9,.001),centers[-1]*1.1,800),*BOUNDS['POR'])])
        kernel=torch.tensor(kernel_matrix(grid,np.r_[DEFAULTS['POR'],centers],'POR').T,dtype=torch.float32,device='cuda')
        de=torch.tensor(density_edges,dtype=torch.float32,device='cuda');vr=np.flatnonzero(fold==f)
        sg=[];sp=[];sw=[];sm=[];ss=[];sr=[]
        with torch.no_grad():
            for pos in range(0,len(vr),1024):
                rows=vr[pos:pos+1024];ri=torch.tensor(rows,device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):go,raw,rho=model(x[ri],den[ri])
                gate=go.sigmoid();lw,mu,sd=mixture_params(raw);weight=lw.exp()
                cond=probability(weight,mu,sd,de);p=torch.cat([gate[:,None],(1-gate[:,None])*cond],1)
                pred[rows,0]=grid[(p@kernel).argmax(1).cpu().numpy()]
                sg.append(gate.cpu().numpy());sp.append(cond.cpu().numpy().astype(np.float16))
                sw.append(weight.cpu().numpy());sm.append(mu.cpu().numpy());ss.append(sd.cpu().numpy());sr.append(rho.cpu().numpy())
        assert np.array_equal(pred[vr,1:],fixed[vr,1:])
        pack={'indices':vr,'POR_gate':np.concatenate(sg),'POR_prob':np.concatenate(sp),'POR_centers':centers,
            'POR_mixture_weight':np.concatenate(sw),'POR_mixture_mu':np.concatenate(sm),'POR_mixture_std':np.concatenate(ss),
            'POR_regression_median':median,'POR_regression_scale':scale,'POR_regression_transform':np.array('log1p'),
            'POR_rho_mean':np.concatenate(sr),'POR_density_input':density[vr],'pred':pred[vr]}
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz',**pack)
        fr={'fold':f+1,'scores':metric(y[vr],pred[vr]),'reference_scores':metric(y[vr],fixed[vr]),'seconds':time.perf_counter()-fs_time,
            'loss':history,'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_gpu_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'PERM_SW_exactly_unchanged':True}
        reports.append(fr);write(args.out/f'fold_{f+1}_report.json',fr);np.save(cache,pred[vr])
        torch.save({'state_dict':model.cpu().state_dict(),'config':config,'feature_median':fm.tolist(),'feature_scale':fs.tolist(),
            'ple_edges_normalized':edges.tolist(),'medians':builder.medians_,'feature_contract':contract,
            'regression_median':float(median),'regression_scale':float(scale),'centers':centers.tolist(),'density_edges':density_edges.tolist()},args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz',truth=y,pred=pred,fold=fold,well=well)
        print('FOLD_RESULT',json.dumps({k:v for k,v in fr.items() if k!='loss'}),flush=True)
        del model,opt,scheduler,x,den,targets,gates,indices,kernel,pack;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(pred).all() and np.array_equal(pred[:,1:],fixed[:,1:])
    assert digest(args.reference)==reference_sha and digest(Path(__file__))==config['source_sha256']
    assert digest(ROOT/'experiments/pure_dl/train_invariant.py')==config['preprocessing_source_sha256']
    score=metric(y,pred);refscore=metric(y,fixed)
    report={'config':config,'oof_scores':score,'reference_scores':refscore,'score_delta':{k:score[k]-refscore[k] for k in score},
        'folds':reports,'seconds':time.perf_counter()-start,'PERM_SW_exactly_unchanged':True,
        'per_well':{rec[i][0]:metric(y[well==i],pred[well==i]) for i in range(80)}}
    write(args.out/'report.json',report);print('FINAL',json.dumps({k:report[k] for k in ['oof_scores','score_delta','seconds']}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'outputs/pure_dl_density_physics')
    ap.add_argument('--reference',type=Path,default=ROOT/'outputs/pure_dl_residual/analytic_800/oof.npz')
    ap.add_argument('--epochs',type=int,default=40);ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--batch-points',type=int,default=4096);ap.add_argument('--seed',type=int,default=20260929)
    ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;assert torch.cuda.is_available()
    if args.benchmark:benchmark(args)
    else:train(args)


if __name__=='__main__':main()
