"""Train-only numeric piecewise-linear coordinates and joint raw encoder/readout."""
import numpy as np
import torch
from torch import nn
from perm_direct_share_v274 import ROOT,SOURCE,PARENT,PARENT_SHA,Evaluator,channel_objective,Network as Parent,load_data as original_data
STUDY=ROOT/'results/deep_learning/perm_numeric_ple_v276_20261001'
OUT=ROOT/'outputs/pure_dl_perm_numeric_ple_v276_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('scalar_repeat','ple')}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
def load_data(stage):
    data,master,metadata=original_data(stage);names=master['selected_feature_names'];indices=[names.index(n) for n in CURVES]
    values=data['x'].index_select(1,torch.tensor(indices,device=data['x'].device))[data['train_raw']].detach().cpu().numpy();knots=[];info=[]
    for c,v in zip(CURVES,values.T):
        finite=v[np.isfinite(v)];span=float(np.ptp(finite)) if len(finite) else 0.;tau=max(1e-6,span*1e-6)
        if len(finite) and span>tau:
            raw=np.quantile(finite,np.linspace(0,1,9));merged=[float(raw[0])]
            for q in raw[1:]:
                if float(q)-merged[-1]>tau:merged.append(float(q))
            k=merged if len(merged)>=2 else []
        else:k=[]
        knots.append(k);info.append(dict(curve=c,span=span,tolerance=tau,active_segments=max(len(k)-1,0)))
    assert sum(max(len(k)-1,0) for k in knots)>0,'No variable raw curve for numeric encoding'
    metadata.update(numeric_knots=knots,numeric_indices=indices,numeric_fit_info=info,numeric_fit_rows=len(values))
    return data,master,metadata
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'repeat');self.mode=mode;self.numeric_metadata=metadata;self.numeric_indices=metadata['numeric_indices'];self.numeric_counts=[max(len(k)-1,0) for k in metadata['numeric_knots']]
        for i,k in enumerate(metadata['numeric_knots']):self.register_buffer('numeric_knots_'+str(i),torch.tensor(k,dtype=torch.float32))
        size=207+sum(self.numeric_counts)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261276);self.stem=nn.Sequential(nn.Linear(size,48),nn.GELU())
        self.numeric_hook=self.stem.register_forward_pre_hook(self.encode)
    def encode(self,module,args):
        z=args[0];parts=[]
        for i,count in enumerate(self.numeric_counts):
            if not count:continue
            k=getattr(self,'numeric_knots_'+str(i));x=z[...,self.numeric_indices[i]:self.numeric_indices[i]+1].float()
            if self.mode=='ple':part=((x-k[:-1])/(k[1:]-k[:-1])).clamp(0,1)
            else:part=((x-k[0])/(k[-1]-k[0])).clamp(0,1).expand(*x.shape[:-1],count)
            parts.append(part.to(z.dtype))
        return (torch.cat((z,*parts),dim=-1),)
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(numeric_encoding=self.mode,numeric_curves=list(CURVES),numeric_indices=self.numeric_indices,numeric_knots=self.numeric_metadata['numeric_knots'],numeric_fit_info=self.numeric_metadata['numeric_fit_info'],numeric_fit_rows=self.numeric_metadata['numeric_fit_rows'],numeric_fit_only_training_inputs=True,labels_or_Q_not_used_for_knots=True,fit_uses_original_already_imputed_training_curve_values=True,merged_duplicate_or_close_knots=True,no_inactive_padding_coordinates=True,active_numeric_dimensions=sum(self.numeric_counts),new_stem_input=207+sum(self.numeric_counts),same_registered_capacity_but_function_capacity_differs=True,base206_missing_flags_and_scalars_retained=True,prior154_frozen=True,new_encoder_and_readout_jointly_trainable=True,final_alpha=.25,reference_final_alpha=.25,no274_weights_loaded=True)
        return info
def make_model(stage,master,metadata):return Network(master,metadata,stage)
