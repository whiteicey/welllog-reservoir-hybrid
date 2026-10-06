"""Train-only numerical token embedding in fresh joint POR curve encoder."""
import torch
from torch import nn
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_curve_ple_v305_20261002'
OUT=ROOT/'outputs/pure_dl_por_curve_ple_v305_fold5'
STAGES={'ple':('POR',256,False,0.,3e-4)}

class PiecewiseNumeric(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('knots',torch.zeros(9,9))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261305);self.projection=nn.Linear(8,32)
        self.fit_info=None
    @torch.no_grad()
    def fit(self,curves):
        assert curves.ndim==2 and curves.shape[1]==9 and torch.isfinite(curves).all()
        knots=torch.quantile(curves.float(),torch.linspace(0,1,9,device=curves.device),dim=0).T
        adjusted=0
        # Repeated quantiles receive deterministic tiny positive widths; no labels.
        for c in range(9):
            floor=max(float(knots[c,-1]-knots[c,0])*1e-6,1e-5)
            for j in range(1,9):
                if knots[c,j]<knots[c,j-1]+floor:
                    knots[c,j]=knots[c,j-1]+floor;adjusted+=1
        self.knots.copy_(knots)
        self.fit_info=dict(train_rows=len(curves),quantiles=9,intervals=8,adjusted_quantile_edges=adjusted,edge_width_floor='max(training range *1e-6,1e-5)',statistics_train_only=True,first_last_basis_linear_extrapolation=True)
    def forward(self,curves):
        # CurveInteraction flattens B*L into point rows and retains curve axis9.
        x=curves.squeeze(-1).float()
        lo=self.knots[:,:-1];width=self.knots[:,1:]-lo
        z=(x.unsqueeze(-1)-lo)/width
        basis=z.clamp(0,1)
        basis=torch.cat((z[...,:1].clamp_max(1),basis[...,1:-1],z[...,-1:].clamp_min(0)),dim=-1)
        return self.projection(basis.to(self.projection.weight.dtype))

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'full')
        self.curve_interaction.numeric=PiecewiseNumeric()
    def initialize_priors(self,data):
        self.curve_interaction.numeric.fit(data['x'][data['train_raw']].index_select(-1,self.curve_indices))
        info=super().initialize_priors(data)
        info.update(numeric_token_embedding='shared8segmentPLEprojection32, curve-specific train-input quantiles',numeric_embedding_fit=self.curve_interaction.numeric.fit_info,new_active_parameters=76816,new_interaction_parameters=19248,extra_parameters_vs302=224,same_capacity_ablation=False,reference='historical302full, source and budget unchanged except numerical embedding',no302_trained_weights=True,not_first_PLE=True,only_original151_prior=True)
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
