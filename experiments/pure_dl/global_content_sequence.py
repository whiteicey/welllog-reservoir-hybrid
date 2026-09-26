"""Content-only global cosine attention; local relative-position bias retained."""
import torch
from global_cosine_sequence import Network as Baseline
from cosine_patch_sequence import SmallAttention, cosine_scores

class ContentAttention(SmallAttention):
    def __init__(self):
        super().__init__()
        del self.relative_bias
    def forward(self,x,valid):
        batch,length,width=x.shape
        assert length<=self.maximum_tokens
        z=self.norm(x).masked_fill(~valid[...,None],0)
        q,k,v=self.qkv(z).reshape(batch,length,3,self.heads,width//self.heads).permute(2,0,3,1,4)
        scores=cosine_scores(q,k,self.log_scale).masked_fill(~valid[:,None,None,:],-1e4)
        result=(scores.softmax(-1).to(v.dtype)@v).transpose(1,2).reshape(batch,length,width)
        return self.project(result).masked_fill(~valid[...,None],0)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        old=self.stem.hierarchy.global_patches
        with torch.random.fork_rng(devices=[]): new=ContentAttention()
        state={k:v for k,v in old.state_dict().items() if k!='relative_bias'}
        new.load_state_dict(state)
        self.stem.hierarchy.global_patches=new
