"""Assemble only fold-local public factors and private extractions for v4 stages."""
from pathlib import Path
import argparse
import hashlib
import json
import torch

from fullcv_context_v97 import STUDY, sha
from fullcv_models_v97 import fold_dir, root_path, best_path
from paired_local_context_router_sequence import Network as Original, retained_indices, calibration_indices
from dual8222_factorized_readout import Network as Factors
from independent_channels_v8 import Network as Private


def original_model(master):
    numeric, conditioning = calibration_indices(master['selected_feature_names'])
    model = Original(retain_indices=retained_indices(master['selected_feature_names']),
                     numeric_indices=numeric, conditioning_indices=conditioning)
    model.load_state_dict(master['state_dict'], strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, choices=[1, 2, 3], required=True)
    ap.add_argument('--action', choices=['factors', 'private'], required=True)
    args = ap.parse_args()
    f = args.fold
    out = fold_dir(f)
    master = torch.load(root_path(f), map_location='cpu', weights_only=False)
    if args.action == 'factors':
        model = Factors(original_model(master), 'gate_only')
        por = torch.load(best_path(f, 'por_density'), map_location='cpu', weights_only=False)
        perm = torch.load(best_path(f, 'perm_gate'), map_location='cpu', weights_only=False)
        states = [por['head_state_dict'], perm['head_state_dict'],
                  {k: v.detach().cpu().clone() for k, v in model.heads[2].state_dict().items()}]
        for cp in (por, perm):
            assert cp['original_fold'] == f and cp['source_sha256'] == sha(root_path(f))
        torch.save({'head_state_dicts': states, 'source_kinds': ['target_group'] * 3,
                    'original_fold': f, 'source_checkpoint': str(root_path(f)),
                    'source_sha256': sha(root_path(f))}, out / 'selected_factors.pt')
    else:
        previous = torch.load(best_path(f, 'por_point'), map_location='cpu', weights_only=False)
        proof = []
        for target in ('POR', 'PERM'):
            model = Private(master, previous, target, 'dual')
            path = out / f'private_{target}.pt'
            state = model.state_dict()
            assert all(torch.isfinite(v).all() for v in state.values())
            torch.save({'state_dict': state, 'best_epoch': 0, 'original_fold': f,
                        'target': target, 'identity_stage': True,
                        'source_checkpoint': str(root_path(f)), 'source_sha256': sha(root_path(f))}, path)
            proof.append({'target': target, 'best_epoch': 0, 'all_state_tensors_exact': True,
                          'checkpoint': str(path), 'sha256': sha(path)})
        (STUDY / 'identity_stage_proof.json').write_text(json.dumps({'proof': proof,
            'protocol': 'Public fold-local identity extraction; no internal weights or labels'}, indent=2), encoding='utf-8')
    print(json.dumps({'fold': f, 'action': args.action, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
