"""Use adjacent positions at each scale; global patch attention keeps long context."""
from patch_hierarchy_sequence import Network as Baseline
from blur_ungated_sequence import UngatedBlock

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        count=0;changed=0
        for module in self.modules():
            if isinstance(module,UngatedBlock):
                convolution=module.depthwise
                assert convolution.kernel_size==(7,) and convolution.padding_mode=='zeros'
                changed+=int(convolution.dilation!=(1,))
                convolution.dilation=(1,);convolution.padding=(3,)
                count+=1
        assert count==9 and changed==3
