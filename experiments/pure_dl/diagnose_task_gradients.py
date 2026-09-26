import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import train_invariant as base
from depth_norm_joint_sequence import Network
from population_likelihood_sequence import objective
from multiscale_sequence import window_indices
from gpu_job_guard import check_slot

def main():
    torch.set_num_threads(3);print('GPU_ADMISSION',check_slot(),flush=True)
    started=time.perf_counter();source=base.ROOT/'outputs/pure_dl_lr55_cosine_1e3';out=base.ROOT/'outputs/task_gradient_diagnostic';out.mkdir(exist_ok=True)
    assert json.loads((source/'screening_verification.json').read_text())['passed']
    ck=torch.load(source/'fold_1.pt',map_location='cpu',weights_only=False)
    rec=base.load_split(base.ROOT.parent/'data','train');oof=np.load(source/'oof.npz');fold=oof['fold'];well=oof['well']
    valid=[np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1) for _,frame in rec]
    truth=np.concatenate([frame[base.TS].to_numpy(np.float64)[m] for (_,frame),m in zip(rec,valid)])
    assert np.array_equal(truth,oof['truth'])
    starts=np.r_[0,np.cumsum([len(frame) for _,frame in rec])];nraw=int(starts[-1])
    mapping=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
    training=np.flatnonzero(fold!=0);train_raw=np.asarray(ck['training_raw_rows']);assert np.array_equal(train_raw,mapping[training])
    ids=sorted(np.unique(well[training]).tolist());assert len(ids)==64
    builder=base.FeatureBuilder().fit([rec[i][1] for i in ids]);assert builder.medians_==ck['medians']
    fm=np.asarray(ck['feature_median'],np.float32);fs=np.asarray(ck['feature_scale'],np.float32)
    x=torch.zeros(nraw,206,device='cuda',dtype=torch.float16)
    for i in ids:
        v=builder.transform(rec[i][1]).enhanced[:,ck['selected_feature_indices']]
        x[starts[i]:starts[i+1]]=torch.from_numpy(np.clip((v-fm)/fs,-8,8).astype(np.float16)).cuda()
    del v
    depth=torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _,frame in rec]),device='cuda')
    absolute=((depth-ck['absolute_depth_median'])/ck['absolute_depth_scale']).clamp(-8,8)
    lower=torch.tensor(np.concatenate([np.full(len(frame),starts[i]) for i,(_,frame) in enumerate(rec)]),device='cuda')
    upper=torch.tensor(np.concatenate([np.full(len(frame),starts[i+1]) for i,(_,frame) in enumerate(rec)]),device='cuda')
    tr=torch.tensor(train_raw,device='cuda');labeled=torch.zeros(nraw,dtype=torch.bool,device='cuda');labeled[tr]=True
    labels=torch.zeros(nraw,3,device='cuda');defaults=torch.zeros(nraw,3,device='cuda',dtype=torch.bool)
    z=np.column_stack([(base.TRANSFORM[j](truth[training,j])-ck['regression_median'][j])/ck['regression_scale'][j] for j in range(3)])
    labels[tr]=torch.tensor(z,device='cuda',dtype=torch.float32);defaults[tr]=torch.tensor(base.dmask(truth[training]),device='cuda')
    assert not labeled[torch.tensor(mapping[fold==0],device='cuda')].any()
    model=Network().cuda().eval();model.load_state_dict(ck['state_dict'])
    shared=[(n,p) for n,p in model.named_parameters() if not n.startswith(('main_head.','aux_mid.','aux_coarse.'))]
    params=[p for _,p in shared];core=torch.zeros(1,256,dtype=torch.bool,device='cuda');core[:,64:192]=True
    generator=torch.Generator(device='cuda').manual_seed(20261031);records=[]
    torch.cuda.reset_peak_memory_stats()
    for batch in range(16):
        anchors=tr[torch.randint(len(tr),(64,),device='cuda',generator=generator)]
        ix,exists=window_indices(anchors,lower,upper);supervised=exists&core&labeled[ix]
        assert ((ix>=lower[anchors,None])&(ix<upper[anchors,None])).all()
        relative=torch.tanh((depth[ix]-depth[anchors,None])/10)
        with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(x[ix],relative,exists,True,absolute_depth=absolute[ix])
        losses=[];gradients=[]
        for j in range(3):
            sliced=tuple((g[:,:,j:j+1],r[:,:,j:j+1]) for g,r in outputs)
            loss=objective(sliced,labels[ix][:,:,j:j+1],defaults[ix][:,:,j:j+1],supervised,True)
            grad=torch.autograd.grad(loss,params,retain_graph=j<2,allow_unused=False)
            vector=torch.cat([t.detach().float().reshape(-1) for t in grad]);assert torch.isfinite(vector).all()
            gradients.append(vector);losses.append(float(loss.detach()))
        g=torch.stack(gradients);norm=g.norm(dim=1);cos=(g@g.T)/(norm[:,None]*norm[None,:]).clamp_min(1e-12)
        records.append({'batch':batch+1,'losses':losses,'norms':norm.cpu().tolist(),'cosine':cos.cpu().tolist(),'supervised_rows':int(supervised.sum())})
        print('GRADIENT_BATCH',batch+1,json.dumps(records[-1]),flush=True)
        del outputs,g,gradients,vector,grad,loss,sliced
    assert all(torch.equal(value.cpu(),ck['state_dict'][name]) for name,value in model.state_dict().items())
    cosine=np.asarray([r['cosine'] for r in records]);norms=np.asarray([r['norms'] for r in records])
    pairs={}
    for i,j in [(0,1),(0,2),(1,2)]:
        values=cosine[:,i,j];pairs[base.TS[i]+'_'+base.TS[j]]={'mean_cosine':float(values.mean()),'median_cosine':float(np.median(values)),'negative_fraction':float((values<0).mean()),'min_max':values[[values.argmin(),values.argmax()]].tolist()}
    report={'source':str(source),'checkpoint_sha256':hashlib.sha256((source/'fold_1.pt').read_bytes()).hexdigest(),'training_wells':ids,'diagnostic_only':True,'optimizer_steps':0,'weights_unchanged':True,'validation_labels_used':False,'model_mode':'eval, dropout off; shared parameters excluding target heads','batches':16,'windows_per_batch':64,'gradient_norm_mean':dict(zip(base.TS,norms.mean(0).tolist())),'pairs':pairs,'records':records,'seconds':time.perf_counter()-started,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2}
    base.write(out/'report.json',report);print('SUMMARY',json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
