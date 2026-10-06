"""Expose existing route weights for router-only utility supervision."""
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge


class Network(Baseline):
    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        if not auxiliary:
            return super().forward(features,relative_depth,valid,False,absolute_depth)
        members=[m(features,relative_depth,valid,True,absolute_depth) for m in self.members]
        main=[output[0] for output in members]
        logits=self.route_logits(features,valid,main)
        detached=[tuple(value.detach() for value in output) for output in main]
        return members,routed_merge(detached,logits),logits.float().softmax(-1)
