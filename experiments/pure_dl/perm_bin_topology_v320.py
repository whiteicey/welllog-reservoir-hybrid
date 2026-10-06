"""Joint raw encoder/readout with ordered versus shuffled target-axis residual."""
import torch
from torch import nn
from perm_sparse_reacquisition_v291 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_bin_topology_v320_20261002'
OUT=ROOT/'outputs/pure_dl_perm_bin_topology_v320_fold5'
STAGES={s:('PERM',256,False,0.,3e-4) for s in ('ordered','shuffled')}
class BinResidual(nn.Module):
    def __init__(self,mode):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261320)
            self.conv=nn.Sequential(nn.Conv1d(1,8,5,padding=2,padding_mode='reflect'),nn.GELU(),nn.Conv1d(8,1,5,padding=2,padding_mode='reflect'))
            nn.init.zeros_(self.conv[-1].weight);nn.init.zeros_(self.conv[-1].bias)
        order=torch.arange(256)
        if mode=='shuffled':order[1:-1]=1+torch.randperm(254,generator=torch.Generator().manual_seed(20261320))
        self.register_buffer('order',order);self.register_buffer('inverse',order.argsort())
    def forward(self,logits):
        shape=logits.shape;z=logits.float()-logits.float().mean(-1,keepdim=True)
        z=z.index_select(-1,self.order).reshape(-1,1,256)
        delta=self.conv(z).reshape(shape).index_select(-1,self.inverse).float()
        return logits.float()+.5*torch.tanh(delta/.5)
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'point');self.topology_mode=mode
        self.bin_residual=BinResidual(mode)
        self.readout.register_forward_hook(self.correct_logits)
    def correct_logits(self,module,args,logits):return self.bin_residual(logits)
    def groups(self):return super().groups()+[dict(params=list(self.bin_residual.parameters()),lr=3e-4,name='PERM_target_bin_axis_residual')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        n=sum(p.numel() for p in self.parameters() if p.requires_grad);assert n==36555
        info.update(topology_mode=self.topology_mode,active_parameters=n,new_bin_axis_parameters=89,initial_direct_logits_unchanged=True,original154_only=True,no291_or312_trained_weights=True,new_raw_encoder_and_readout_jointly_trainable=True,endpoint_bins_fixed=True,permutation_seed=20261320,receptive_field_bins=9,residual_logit_bound=.5,main_and_aux_use_same_corrected_logits=True,ordered_transformed_target_not_uniform_physical_distance=True)
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
