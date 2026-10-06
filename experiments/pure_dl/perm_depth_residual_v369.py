"""Retain effective PERM336 design; small depth-context residual in its new encoder."""
import torch
from torch import nn
from perm_low_rank_cross_v336 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent

STUDY=ROOT/'results/deep_learning/perm_depth_residual_v369_20261002'
OUT=ROOT/'outputs/pure_dl_perm_depth_residual_v369_fold5'
STAGES={mode:('PERM',256,False,0.,3e-4) for mode in ('point','local')}

class DepthPatch(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.norm=nn.LayerNorm(48,elementwise_affine=False)
        self.down=nn.Linear(48,12,bias=False)
        self.kernel=nn.Parameter(torch.zeros(5,12))
        self.up=nn.Linear(12,48)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self,h,valid):
        with torch.autocast(device_type=h.device.type,enabled=False):
            z=self.down(self.norm(h.float()))
            weights=self.kernel.softmax(0)
            row=torch.arange(h.shape[1],device=h.device)
            numerator=torch.zeros_like(z);denominator=torch.zeros_like(z)
            offsets=(-8,-4,0,4,8) if self.mode=='local' else (0,0,0,0,0)
            for i,offset in enumerate(offsets):
                ids=row+offset;inside=(ids>=0)&(ids<h.shape[1]);ids=ids.clamp(0,h.shape[1]-1)
                use=(valid[:,ids]&inside.unsqueeze(0)).unsqueeze(-1)
                mass=use.float()*weights[i]
                numerator=numerator+z[:,ids]*mass
                denominator=denominator+mass
            mixed=numerator/denominator.clamp_min(1e-12)
            return ((h.float()+.25*torch.tanh(self.up(mixed)))*valid.unsqueeze(-1)).to(h.dtype)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'cross')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261369);self.depth_patch=DepthPatch(mode)
        # Parent first applies its effective low-rank cross adapter. This hook
        # adds depth information to that feature flow before both selectors.
        self.stem.register_forward_hook(self.patch_stem)

    def patch_stem(self,module,args,h):
        assert self.current_valid is not None
        return self.depth_patch(h,self.current_valid)

    def groups(self):
        return super().groups()+[dict(params=list(self.depth_patch.parameters()),lr=3e-4,name='PERM_joint_depth_residual_encoder')]

    def initialize_priors(self,data):
        self.depth_patch.requires_grad_(False)
        try:info=super().initialize_priors(data)
        finally:self.depth_patch.requires_grad_(True)
        count=sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert count==38646
        info.update(new_active_parameters=count,active_parameters=count,new_depth_patch_parameters=1260,
                    temporal_mode=self.depth_patch.mode,latent_width=12,local_offsets=[-8,-4,0,4,8],
                    valid_weight_renormalization=True,new_patch_bound=.25,zero_up_initial_function_identity=True,
                    original154_only=True,parent336_learned_checkpoint_loaded=False,
                    retained_design='sparse reacquisition/rational body and readout/low-rank cross adapter',
                    one_PERM_additional_phase=True,joint_new_body_and_readout=True,
                    primary_control='same architecture/parameter count/init point versus local connectivity',
                    effective_function_capacity_matched=False,
                    control_limit='point repeats center at five taps, so kernel weights are functionally redundant; this is a connectivity ablation, not matched effective capacity')
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
