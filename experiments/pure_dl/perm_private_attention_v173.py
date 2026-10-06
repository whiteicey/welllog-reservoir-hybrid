"""Random private PERM local/token-attention backbone, no old predictor loading."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,load_data,Evaluator,channel_objective
from divergent_models_v45 import LongSection
from src.utils import DEFAULTS
PARENT=SOURCE
STUDY=ROOT/'results/deep_learning/perm_private_attention_v173_20261001'
OUT=ROOT/'outputs/pure_dl_perm_private_attention_v173_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,6e-4)}
class Network(nn.Module):
    def __init__(self,master,mode):
        super().__init__();self.body=LongSection();self.gate=nn.Linear(128,1);self.density=nn.Linear(128,48)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.density.weight)
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][1][1:-1],dtype=torch.float32))
        self.register_buffer('default_z',torch.tensor((math.log10(DEFAULTS['PERM'])-float(master['regression_median'][1]))/float(master['regression_scale'][1])))
    def groups(self):return [dict(params=list(self.parameters()),lr=6e-4,name='private_PERM_complete_model')]
    def initialize_priors(self,data):
        rows=data['train_raw'];d=data['default'][rows,1];y=data['labels'][rows,1][~d].float()
        assert bool(torch.isfinite(y).all()) and len(y)>0
        prob=d.float().mean().clamp(.001,.999);means=torch.quantile(y,torch.linspace(.05,.95,16,device=y.device))
        sd=(y.std(unbiased=False)*.5).clamp_min(.05);rawsd=torch.log(torch.expm1(sd-.025))
        with torch.no_grad():
            self.gate.bias.copy_(torch.logit(prob).reshape(1));self.density.bias.copy_(torch.cat((torch.zeros_like(means),means,rawsd.expand(16))))
        return dict(random_private_backbone=True,no_source_predictor_weights_loaded=True,initial_distribution_train64_only=True,
                    training_anchors=len(rows),continuous_training_anchors=len(y),default_probability=float(prob),
                    source_counted_for_other_targets=True,executed_training_stages_including_source=5,
                    Gaussian_components=16,TCN_width=96,token_attention_blocks=4,token_pool=16,
                    joint_ranked_probability_loss=True,rps_coefficient=1.,original_finite_edges=255,
                    original_transform='log10',default_physical=DEFAULTS['PERM'])
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        hidden,_,_=self.body(x[...,:206],relative,valid)
        gate=self.gate(hidden).float();raw=self.density(hidden).reshape(*hidden.shape[:2],1,3,16).float()
        return gate,raw
def make_model(stage,master,metadata):return Network(master,stage)
