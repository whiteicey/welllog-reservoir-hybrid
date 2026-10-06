"""Eight-step bounded binary-state messages; self-state matched control.

Finite-radius learned message passing, not full-chain CRF or Viterbi decoding.
Original conditional density and official point action decoder remain fixed.
"""
import torch
from torch import nn
from torch.nn import functional as F
from quality_readout_v37 import Network as ParentBuilder,ROOT,PREVIOUS,load_data,reference_prediction,Evaluator,TARGETS

STUDY=ROOT/'results/deep_learning/state_messages_v44_20260929'
MODES=('self','neighbor')

def left(x):return F.pad(x[:,:-1],(1,0))
def right(x):return F.pad(x[:,1:],(0,1))

class StateHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261144)
            self.input=nn.Conv1d(319,32,1);self.output=nn.Conv1d(32,1,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,parent,context,side,valid):
        x=torch.cat((hidden[0].float(),hidden[1].float(),side.transpose(1,2).float()),1).masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=self.output(F.gelu(self.input(x))).squeeze(1).float()
        # J[t] is the symmetric edge linking t-1,t, bounded to [-.5,.5].
        edge_valid=valid&left(valid)
        coupling=.5*torch.tanh((z+left(z))/2)
        coupling=coupling.masked_fill(~edge_valid,0)
        jl=coupling.tanh();jr=right(coupling).tanh()
        g=parent[0].squeeze(-1).float()
        ml=torch.zeros_like(g);mr=torch.zeros_like(g)
        for _ in range(8):
            gl=left(g+ml) if self.mode=='neighbor' else g+ml
            gr=right(g+mr) if self.mode=='neighbor' else g+mr
            ml=2*torch.atanh((jl*torch.tanh(gl/2)).clamp(-.999,.999))
            mr=2*torch.atanh((jr*torch.tanh(gr/2)).clamp(-.999,.999))
        updated=(g+ml+mr).masked_fill(~valid,0)
        # Use original g on padding too; valid outputs start exactly unchanged.
        updated=torch.where(valid,updated,g)
        return updated.unsqueeze(-1),parent[1]

class Network(ParentBuilder):
    def __init__(self,target,master,modes=MODES):
        if target!='SW':raise ValueError('SW only')
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:StateHead(m) for m in modes})
        self.factor='bounded_state_messages8';self.train(False)

def state_objective(output,truth,default,mask,quality,target_index):
    g=output[0][mask][:,0].float();d=default[mask][:,target_index].float()
    loss=(F.binary_cross_entropy_with_logits(g,d,reduction='none')*quality[mask].detach().float()).mean()/3
    return loss,loss,loss.new_zeros(())

@torch.no_grad()
def factor_scope(model,data,mode):
    rows=data['train_raw'][[0,100]]
    from multiscale_sequence import window_indices
    ix,valid=window_indices(rows,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][rows,None])/10)
    h,old,ctx,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    new=model.heads[mode](h,old,ctx,data['x'][ix,...,206:],valid)
    assert torch.equal(new[1],old[1])
    return dict(conditional_density_exact=True,gate_change_max=float((new[0]-old[0]).abs().max()),
        message_steps=8,max_message_radius=8 if mode=='neighbor' else 0,
        pair_coupling_bound=.5,gate_delta_bound=2.,scope='two fixed training windows; labels not used')
