"""Read-only native distribution grids for four complete models; yields to training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['WELLLOG_RUN_SEED'] = '20261010'
from pathlib import Path
import argparse, hashlib, importlib, json, sys, time
import numpy as np
import psutil
import torch
from gpu_job_guard_v2 import check_slot
from multiscale_sequence import window_indices
import complete_seed_models_v362r1 as helper
import champion_scaling_v21

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/original_last_seed_fusion_v371_20261002'
BATCH = ROOT/'results/deep_learning/complete_seed_compare_v362_20261002/BATCH_STATE.json'
BASE_LOADER = champion_scaling_v21.load_data
SPECS = {
 'POR': ('POR341','por_coupling_conditional_rps_v341','conditional'),
 'PERM': ('PERM336','perm_low_rank_cross_v336','cross'),
 'SW': ('SW351','sw_dynamic_cross_spectrum_v351','aligned')}
HISTORICAL = {
 'SOURCE/fold_5.pt': ROOT/'outputs/pure_dl_paired_local_context_router80_sequence/fold_5.pt',
 'POR113/model_best.pt': ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt',
 'POR151/model_best.pt': ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt',
 'PERM154/model_best.pt': ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt',
 'SW104/model_best.pt': ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt'}
HASHES = {}

def reset_context():
    torch.load = helper._load
    torch.manual_seed = helper._manual_seed
    champion_scaling_v21.load_data = BASE_LOADER
    for module in list(sys.modules.values()):
        f = getattr(module, '__file__', None)
        if not f or not Path(f).resolve().is_relative_to(ROOT/'experiments/pure_dl'):
            continue
        if module is helper:
            continue
        for attr in ('SOURCE','PARENT'):
            p = getattr(module, attr, None)
            if isinstance(p, Path) and p.resolve().is_relative_to(ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'):
                leaf = '/'.join(p.parts[-2:])
                if leaf not in HISTORICAL:
                    raise RuntimeError('Unknown contextual checkpoint '+str(p))
                setattr(module, attr, HISTORICAL[leaf])
        p = getattr(module, 'PARENT', None)
        if hasattr(module, 'PARENT_SHA') and isinstance(p, Path) and p in HISTORICAL.values():
            if p not in HASHES:
                HASHES[p] = hashlib.sha256(p.read_bytes()).hexdigest()
            module.PARENT_SHA = HASHES[p]

def training_snapshot():
    state = json.loads(BATCH.read_text(encoding='utf-8-sig'))
    if state['status'] == 'complete':
        return state, None, False
    if state['status'] != 'running':
        return state, None, True
    try:
        process = psutil.Process(state['active_pid'])
        cmd = ' '.join(process.cmdline())
        if 'train_complete_seed_' not in cmd:
            return state, None, True
        state['trainer_create_time'] = process.create_time()
    except psutil.Error:
        return state, None, True
    phase = ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'/('seed_'+str(state['seed']))/('SW351_fixed' if state['stage']=='SW351' else state['stage'])
    runtime_file = phase/'runtime_status.json'
    runtime = json.loads(runtime_file.read_text(encoding='utf-8-sig')) if runtime_file.exists() else {}
    source_history=phase/'fold_5_validation_history.json'
    if state['stage']=='SOURCE' and source_history.exists():
        history=json.loads(source_history.read_text(encoding='utf-8-sig'))
        if history:
            runtime=dict(history[-1]);runtime['bad']=runtime.get('bad_epochs',0)
    epoch = runtime.get('epoch', 0)
    patience = runtime.get('effective_patience', 10)
    # A best-at-initialization head can stop exactly at minimum epoch 10.
    # Yield already at epoch 6 when its four-epoch safety margin is exhausted.
    near = (epoch >= 6 and runtime.get('bad',0) >= patience-4) or epoch >= 76
    near = near or (ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/STOP_BATCH.json').exists()
    return state, runtime, near

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channel', required=True, choices=list(SPECS))
    ap.add_argument('--extension', action='store_true', help='Newly completed seed13: four additional pairs, preserve initial cached results')
    args = ap.parse_args(); target = args.channel; c = list(SPECS).index(target)
    prefix = 'EXTRA_' if args.extension else ''
    final = STUDY/(prefix+target+'_GPU_FINAL.json')
    if final.exists():
        raise RuntimeError('Finished comparison immutable')
    started = time.perf_counter(); plan=json.loads((STUDY/('AUGMENT_PLAN.json' if args.extension else 'PLAN.json')).read_text())
    state, runtime, near = training_snapshot()
    if near:
        print('DEFER_BEFORE_GPU',json.dumps(dict(state=state,runtime=runtime)),flush=True)
        return
    signature = (state.get('active_pid'),state.get('trainer_create_time'))
    # Unlike the historical guard, include actual evaluation processes too.
    others=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or [])
            if p.pid != os.getpid() and (p.info['name'] or '').lower().startswith('python') and any(s in cmd for s in ('train_','benchmark','evaluate')):
                others.append(p.pid)
        except psutil.Error:
            pass
    if len(others)>1:
        raise RuntimeError('Maximum two GPU tasks: '+str(others))
    torch.set_num_threads(3); slot=check_slot(min_free_gib=4.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    stage,name,mode=SPECS[target]; module=importlib.import_module(name);length=module.STAGES[mode][1]
    models=[];traces={}
    for spec in plan['models']:
        reset_context()
        if spec['seed'] is None:
            data,master,metadata=module.load_data(mode)
            model=module.make_model(mode,master,metadata).cuda()
            model.initialize_priors(data)
            evaluator=module.Evaluator(data,master,c,length,mode)
        else:
            helper.SEED=spec['seed'];helper.OUT=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'/('seed_'+str(spec['seed']))
            helper.SOURCE=helper.OUT/'SOURCE/fold_5.pt';helper.TRACE.clear();helper.bind_paths()
            data,master,metadata=helper.load_data(stage)
            model=helper.make_model(stage,master,metadata).cuda();model.initialize_priors(data)
            evaluator=helper.Evaluator(data,master,c,length,stage)
        path=Path(spec['heads'][target]['checkpoint'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['heads'][target]['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);del saved;model.eval()
        traces[spec['id']]=list(helper.TRACE) if spec['seed'] is not None else []
        models.append((data,model,evaluator))
    reset_context()
    base=models[0][0];ev=models[0][2]
    for data,model,evaluator in models[1:]:
        assert np.array_equal(ev.actions,evaluator.actions)
        assert ev.centers==evaluator.centers and ev.anchors==evaluator.anchors
        assert np.array_equal(ev.local,evaluator.local)
        assert np.array_equal(base['validation'],data['validation']) and np.array_equal(base['truth'],data['truth'])
        for k in ('depth','lower','upper','row_well','absolute','raw_to_valid'):
            assert torch.equal(base[k],data[k]) if torch.is_tensor(base[k]) else np.array_equal(base[k],data[k]),k
    pairs=[p for p in plan['pairs'] if p['id']!=plan['reused_pair']]
    ids=[m['id'] for m in plan['models']];lookup={x:i for i,x in enumerate(ids)}
    weights=np.array(plan['weights']);n=len(base['validation']);batch=min(m[2].batch for m in models)
    cache=STUDY/(prefix+target+'_partial.npz');begin=0
    if cache.exists():
        with np.load(cache) as z:
            out=z['native'];coverage=z['coverage'];indices=z['indices'];begin=int(z['next_begin'])
    else:
        out=np.full((n,len(models)),np.nan,np.float64);coverage=np.zeros(n,np.int8)
        indices=np.full((len(pairs),n,101),65535,np.uint16)
    truth=base['truth'][base['validation']].copy()
    if not (STUDY/'truth.npy').exists():np.save(STUDY/'truth.npy',truth)
    else:assert np.array_equal(truth,np.load(STUDY/'truth.npy'))
    loaded=time.perf_counter()-started;torch.cuda.reset_peak_memory_stats()
    lo=length//2-64;hi=lo+128
    def should_yield():
        current,rt,near=training_snapshot()
        changed=current.get('status')!='complete' and (current.get('active_pid'),current.get('trainer_create_time'))!=signature
        return near or changed,current,rt
    with torch.inference_mode():
        for bi,offset in enumerate(range(begin,len(ev.centers),batch)):
            if bi%4==0:
                defer,current,rt=should_yield()
                if defer:
                    np.savez(cache,native=out,coverage=coverage,indices=indices,next_begin=offset)
                    note=dict(status='yielded_for_training',next_begin=offset,seconds=time.perf_counter()-started,peak_MiB=torch.cuda.max_memory_allocated()/2**20,trainer=current,runtime=rt)
                    (STUDY/(prefix+target+'_YIELD.json')).write_text(json.dumps(note,indent=2),encoding='utf-8')
                    print('YIELDED',json.dumps(note),flush=True);return
            cc=torch.tensor(ev.centers[offset:offset+batch],device='cuda');anchor=torch.tensor(ev.anchors[offset:offset+batch],device='cuda')
            ix,valid=window_indices(cc,base['lower'],base['upper'],length=length,anchors=anchor)
            rows=base['raw_to_valid'][ix[:,lo:hi].cpu().numpy()]
            use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=ev.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda');utilities=[]
            for j,(data,model,evaluator) in enumerate(models):
                rel=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
                context=data['well_bank'][data['row_well'][anchor]]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    gate,logmass=model(data['x'][ix],rel,valid,absolute_depth=data['absolute'][ix],well_inputs=context)
                g=gate[:,lo:hi][take][:,0].float().sigmoid();mass=logmass[:,lo:hi][take][:,0].float().exp()
                joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
                utilities.append(joint@evaluator.utilities)
                out[dest,j]=evaluator.actions[utilities[-1].argmax(-1).cpu().numpy()]
            for pi,pair in enumerate(pairs):
                ua=utilities[lookup[pair['A']]];ub=utilities[lookup[pair['B']]]
                for wi,w in enumerate(weights):
                    indices[pi,dest,wi]=(ua*float(w)+ub*float(1-w)).argmax(-1).cpu().numpy().astype(np.uint16)
    assert (coverage==1).all() and np.isfinite(out).all() and (indices<len(ev.actions)).all()
    for j,spec in enumerate(plan['models']):
        native=np.load(Path(spec['heads'][target]['checkpoint']).parent/'channel_pred.npy')
        assert np.array_equal(native,out[:,j]),'Native replay differs: '+spec['id']
    for pi,pair in enumerate(pairs):
        assert np.array_equal(ev.actions[indices[pi,:,0]],out[:,lookup[pair['B']]])
        assert np.array_equal(ev.actions[indices[pi,:,-1]],out[:,lookup[pair['A']]])
    # Uncompressed writes keep the GPU process out of the training handoff window.
    np.savez(STUDY/(prefix+target+'_grids.npz'),indices=indices,actions=ev.actions,weights=weights,native=out)
    result=dict(target=target,pairs=pairs,models=ids,rows=n,coverage_exactly_once=True,native_disk_replay_exact=True,endpoints_exact=True,
                actual_seed_loads=traces,seconds=time.perf_counter()-started,load_preparation_seconds=loaded,
                peak_MiB=torch.cuda.max_memory_allocated()/2**20,admission=slot,other_GPU_pids=others,
                concurrent_training_continues=True,includes=f'load/preparation/warmup/shared {len(models)}-model forward/all grids/output writing',
                extra_training_stages=0,screening_fold=5,complete_five_folds=False)
    final.write_text(json.dumps(result,indent=2),encoding='utf-8');print('GPU_FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
