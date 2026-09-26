"""Fixed training-only utility fine-tuning of each same-fold invariant network.

Reuse the audited fold-local data/training loop unchanged. Warm starts are local
models trained only on the corresponding64 training wells, never release weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import train_invariant as base
from warmstart_contract import build_contract, validate_resume

SOURCE = base.ROOT/'outputs/pure_dl_invariant'
OriginalNetwork = base.Network
OriginalAdamW = torch.optim.AdamW
original_write = base.write
SOURCE_CHECKPOINTS = []
RESUME_CONTRACT = {}


class FineTuneAdamW(OriginalAdamW):
    def __init__(self, params, *args, **kwargs):
        kwargs['lr'] = .00015
        super().__init__(params, *args, **kwargs)


def component_utilities(mu, targets, median, scale):
    transformed_mean = mu*scale[None, :, None]+median[None, :, None]
    transformed_truth = targets*scale[None, :]+median[None, :]
    por_p = torch.expm1(transformed_mean[:, 0].clamp(-5, 5)).clamp(0, 50)
    por_y = torch.expm1(transformed_truth[:, 0])
    sw_p = transformed_mean[:, 2].clamp(-5, 5).exp().clamp(8.305, 99.9)
    sw_y = transformed_truth[:, 2].exp()
    por = (1-(por_p-por_y[:, None]).abs()/(.08*(por_y.abs()[:, None]+1e-5))).clamp(0, 1)
    perm = (1-(transformed_mean[:, 1].clamp(-2, 4)-transformed_truth[:, 1, None]).abs()).clamp(0, 1)
    sw = (1-(sw_p-sw_y[:, None]).abs()/(.05*(sw_y.abs()[:, None]+1e-5))).clamp(0, 1)
    return torch.stack([por, perm, sw], 1)


class UtilityNetwork(OriginalNetwork):
    def __init__(self, edges=None, initial_means=None):
        super().__init__(edges, initial_means)
        med, scale = np.zeros(3), np.ones(3)
        if edges is not None:
            matches = [ck for ck in SOURCE_CHECKPOINTS
                       if np.array_equal(edges, np.asarray(ck['ple_edges_normalized'], np.float32))]
            if len(matches) != 1:
                raise ValueError('Warm start must match exactly one audited fold-specific input encoding')
            ck = matches[0]
            self.load_state_dict(ck['state_dict'], strict=True)
            med, scale = ck['regression_median'], ck['regression_scale']
        self.register_buffer('utility_median', torch.tensor(med, dtype=torch.float32))
        self.register_buffer('utility_scale', torch.tensor(scale, dtype=torch.float32))


def training_step(model, opt, x, gates, targets, training_indices, batch_points):
    rows = training_indices[torch.randint(len(training_indices), (batch_points,), device='cuda')]
    target_gate, target_z = gates[rows], targets[rows]
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        gate_logits, raw = model(x[rows])
    logw, mu, sd = base.mixture_params(raw)
    logp = logw-.5*((target_z[:, :, None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi)
    nll = -torch.logsumexp(logp, -1)
    utility = (logw.exp()*component_utilities(mu, target_z, model.utility_median, model.utility_scale)).sum(-1)
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits.float(), target_gate)
    density_loss = sum((.25*nll[:, j]-utility[:, j])[target_gate[:, j] < .5].mean() for j in range(3))/3
    loss = gate_loss+density_loss
    assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
    opt.step()
    return float(loss.detach()), float(gate_loss.detach()), float(density_loss.detach())


def write(path, value):
    if Path(path).name == 'config.json':
        value['method'] = 'Same-fold invariant NN warm start; 20 fixed epochs; .25 conditional MDN NLL minus weighted component-mean official utility, plus gate BCE'
        value['loss'] = 'gate BCE + mean_target(.25*NLL - sum_mixture_weight*official_score(component_mean, truth))'
        value['warmstart_source'] = str(SOURCE)
        value['warmstart_sha256'] = {str(f+1): hashlib.sha256((SOURCE/f'fold_{f+1}.pt').read_bytes()).hexdigest() for f in range(5)}
        value['finetune_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        value['warmstart_selection'] = 'Match all input PLE knots to one source fold; no outer labels used for warm start or optimization'
        value['finetune_learning_rate'] = .00015
        value.update(RESUME_CONTRACT)
    original_write(path, value)


def self_check():
    truth = torch.tensor([[np.log1p(10.), np.log10(5.), np.log(50.)]], dtype=torch.float32)
    mu = truth[:, :, None].expand(-1, -1, 8).clone().requires_grad_(True)
    u = component_utilities(mu, truth, torch.zeros(3), torch.ones(3))
    assert torch.allclose(u, torch.ones_like(u), atol=1e-5)
    distant = torch.tensor([[np.log1p(1.), np.log10(5000.), np.log(90.)]], dtype=torch.float32)[:, :, None].expand(-1, -1, 8)
    assert torch.equal(component_utilities(distant, truth, torch.zeros(3), torch.ones(3)), torch.zeros_like(u))
    component_utilities(mu+.005, truth, torch.zeros(3), torch.ones(3)).sum().backward()
    assert torch.isfinite(mu.grad).all()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = FineTuneAdamW([parameter], lr=.001)
    assert optimizer.param_groups[0]['lr'] == .00015
    parameter.square().backward(); optimizer.step()
    assert torch.isfinite(parameter)
    print('PASS: transformed component utility matches perfect/far cases and has finite local gradients')


def main():
    global RESUME_CONTRACT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=base.ROOT/'outputs/pure_dl_utility_finetune')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    args.epochs = 20; args.steps = 64; args.batch_points = 4096
    args.seed = 20260929; args.benchmark = False; args.audit_preprocessing = False
    RESUME_CONTRACT = build_contract(SOURCE, __file__, base.__file__, args,
                                    'finetune_source_sha256', 'finetune_learning_rate', .00015)
    validate_resume(args.out, RESUME_CONTRACT)
    if not (SOURCE/'verification.json').exists() or not (SOURCE/'preprocessing_verification.json').exists():
        raise RuntimeError('Audited invariant source required')
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        SOURCE_CHECKPOINTS.extend(torch.load(SOURCE/f'fold_{f+1}.pt', map_location='cpu', weights_only=True) for f in range(5))
    for ck in SOURCE_CHECKPOINTS:
        probe = UtilityNetwork(np.asarray(ck['ple_edges_normalized'], np.float32))
        for name, tensor in ck['state_dict'].items():
            assert torch.equal(probe.state_dict()[name], tensor), name
        assert np.allclose(probe.utility_median.numpy(), ck['regression_median'])
    del probe
    base.Network = UtilityNetwork
    base.training_step = training_step
    base.write = write
    # Process-local override: the reused loop's optimizer is otherwise fixed at
    # its from-scratch learning rate. No source or other process is modified.
    base.torch.optim.AdamW = FineTuneAdamW
    torch.set_num_threads(3); torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    base.train(args)


if __name__ == '__main__':
    main()
