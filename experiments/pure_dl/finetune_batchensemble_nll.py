"""Continue the audited same-fold206-feature BatchEnsemble for20 fixed NLL epochs.

This tests incomplete optimization rather than changing the target objective.
All data handling and decoding reuse the existing audited training loop.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

import train_batchensemble_invariant as base
from warmstart_contract import build_contract, validate_resume

SOURCE = base.ROOT/'outputs/pure_dl_batchensemble_invariant'
OriginalNetwork = base.Network
OriginalAdamW = torch.optim.AdamW
original_write = base.write
CHECKPOINTS = []
RESUME_CONTRACT = {}


class WarmNetwork(OriginalNetwork):
    def __init__(self, edges=None, initial_means=None):
        super().__init__(edges, initial_means)
        if edges is not None:
            matches = [ck for ck in CHECKPOINTS if np.array_equal(edges, np.asarray(ck['ple_edges_normalized'], np.float32))]
            if len(matches) != 1:
                raise ValueError('Expected a unique same-fold warm start matching every input-encoding knot')
            self.load_state_dict(matches[0]['state_dict'], strict=True)


class WarmAdamW(OriginalAdamW):
    def __init__(self, params, *args, **kwargs):
        kwargs['lr'] = .0003
        super().__init__(params, *args, **kwargs)


def write(path, value):
    if Path(path).name == 'config.json':
        value['method'] = 'Continue same-fold audited206-feature4-member BatchEnsemble for20 fixed NLL epochs; original loss/decoding retained'
        value['warmstart_source'] = str(SOURCE)
        value['warmstart_sha256'] = {str(f+1): hashlib.sha256((SOURCE/f'fold_{f+1}.pt').read_bytes()).hexdigest() for f in range(5)}
        value['continuation_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        value['continuation_learning_rate'] = .0003
        value['source_training_epochs'] = 40
        value['total_training_epochs'] = 60
        value['selection'] = 'Fixed 40 source epochs plus 20 continuation epochs, 64 steps per epoch; no outer-fold selection'
        value.update(RESUME_CONTRACT)
    original_write(path, value)


def main():
    global RESUME_CONTRACT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=base.ROOT/'outputs/pure_dl_batchensemble_nll60')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.epochs = 20; args.steps = 64; args.batch_points = 4096
    args.seed = 20261001; args.benchmark = False
    if not (SOURCE/'verification.json').exists():
        raise RuntimeError('Verified same-fold source required')
    torch.set_num_threads(3)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        CHECKPOINTS.extend(torch.load(SOURCE/f'fold_{f+1}.pt', map_location='cpu', weights_only=True) for f in range(5))
    for ck in CHECKPOINTS:
        model = WarmNetwork(np.asarray(ck['ple_edges_normalized'], np.float32))
        for name, value in ck['state_dict'].items():
            assert torch.equal(model.state_dict()[name], value), name
    del model
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = WarmAdamW([parameter], lr=.001)
    assert optimizer.param_groups[0]['lr'] == .0003
    parameter.square().backward(); optimizer.step()
    assert torch.isfinite(parameter)
    print('PASS: five unique exact warm starts and continuation optimizer', flush=True)
    if args.check_only:
        return
    RESUME_CONTRACT = build_contract(SOURCE, __file__, base.__file__, args,
                                    'continuation_source_sha256', 'continuation_learning_rate', .0003)
    validate_resume(args.out, RESUME_CONTRACT)
    base.Network = WarmNetwork
    base.write = write
    base.torch.optim.AdamW = WarmAdamW
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    base.train(args)


if __name__ == '__main__':
    main()
