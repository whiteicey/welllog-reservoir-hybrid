"""Train-only length-powered well sampling, unchanged compact SW model/score."""
import torch
from compact_frequency_v104 import Network as Parent,ROOT,SOURCE,load_data
STUDY=ROOT/'results/deep_learning/compact_sw_well_sampling_v120_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_well_sampling_v120_fold5'
STAGES={n:('SW',1024,True,1.,3e-4) for n in ('sqrt_well','equal_well')}
class Network(Parent):
    def __init__(self,metadata,mode):super().__init__(metadata,'local');self.sampling_mode=mode
def make_model(stage,master,metadata):return Network(metadata,stage)
class WellSampler:
    def __init__(self,data,mode):
        raw=data['train_raw'];ids=data['row_well'][raw];order=ids.argsort(stable=True)
        wells,counts=torch.unique(ids[order],sorted=True,return_counts=True)
        self.anchors=raw[order];self.counts=counts;self.offsets=counts.cumsum(0)-counts
        self.exponent=.5 if mode=='sqrt_well' else 0.;self.weights=counts.float().pow(self.exponent);self.weights/=self.weights.sum()
        assert len(wells)==64 and int(counts.sum())==len(raw)
        self.info=dict(mode=mode,well_length_exponent=self.exponent,well_ids=wells.cpu().tolist(),training_anchor_counts=counts.cpu().tolist(),well_probabilities=self.weights.cpu().tolist(),fit_scope='original training wells/input anchor availability only, no heldout labels',anchor_population_preserved=True,windows_not_point_reward_reweighted=True)
    def sample(self,batch,generator):
        w=torch.multinomial(self.weights,batch,replacement=True,generator=generator)
        local=(torch.rand(batch,device=self.anchors.device,generator=generator)*self.counts[w]).long()
        return self.anchors[self.offsets[w]+local]
