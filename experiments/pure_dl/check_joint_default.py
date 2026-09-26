"""Independent categorical enumeration and model interface checks; CPU by default."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline
from joint_default_sequence import Network, marginal_logits, state_bits
from joint_default_loss import head_loss, objective
from por_t3_patch_loss import head_loss as original_loss

p = argparse.ArgumentParser(); p.add_argument('--cuda', action='store_true'); args = p.parse_args()
torch.set_num_threads(2)
torch.manual_seed(117)
independent = torch.randn(32, 3, requires_grad=True)
bits = state_bits()
joint = independent @ bits.float().T
explicit = torch.prod(torch.where(bits[None], independent.sigmoid()[:, None],
                                  1 - independent.sigmoid()[:, None]), -1)
torch.testing.assert_close(joint.softmax(-1), explicit)
torch.testing.assert_close(marginal_logits(joint), independent, atol=1e-6, rtol=1e-6)
labels = bits[torch.arange(32) % 8]
state = (labels.long() * torch.tensor([1, 2, 4])).sum(-1)
ce = F.cross_entropy(joint, state, reduction='none') / 3
bce = F.binary_cross_entropy_with_logits(independent, labels.float(), reduction='none').mean(-1)
torch.testing.assert_close(ce, bce)
arbitrary = torch.randn(32, 8) * 20
expected = arbitrary.softmax(-1) @ bits.float()
torch.testing.assert_close(marginal_logits(arbitrary).sigmoid(), expected, atol=1e-6, rtol=1e-6)
assert torch.isfinite(marginal_logits(arbitrary * 1000)).all()

torch.manual_seed(61); baseline = Baseline(features=12).eval()
torch.manual_seed(61); model = Network(features=12).eval()
x = torch.randn(2, 256, 12); d = torch.randn(2, 256)
valid = torch.ones(2, 256, dtype=torch.bool); valid[:, :17] = False
with torch.no_grad():
    original = baseline(x, d, valid, absolute_depth=d)
    actual = model(x, d, valid, absolute_depth=d)
    torch.testing.assert_close(actual[0], original[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[1], original[1], rtol=0, atol=0)
    changed = x.clone(); changed[~valid] += 100
    altered = model(changed, d, valid, absolute_depth=d)
    for a, b in zip(actual, altered): torch.testing.assert_close(a, b, rtol=0, atol=0)
    changed = x.clone(); changed[1] += 100
    altered = model(changed, d, valid, absolute_depth=d)
    for a, b in zip(actual, altered): torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
target = torch.randn(2, 256, 3)
default = torch.rand(2, 256, 3) > .5
q = torch.rand(2, 256, requires_grad=True)
new = model(x, d, valid, joint_gate=True, absolute_depth=d)
torch.testing.assert_close(head_loss(new, target, default, valid, q),
                           original_loss(original, target, default, valid, q), atol=2e-6, rtol=2e-6)
loss = objective(model(x, d, valid, True, absolute_depth=d), target, default, valid, True, q, epoch=0)
loss.backward()
assert q.grad is None
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
raw = torch.randn(1, 8, 3, 3, 8, requires_grad=True)
gate = torch.randn(1, 8, 8, requires_grad=True)
all_default = torch.ones(1, 8, 3, dtype=torch.bool)
head_loss((gate, raw), torch.randn(1, 8, 3), all_default,
          torch.ones(1, 8, dtype=torch.bool), torch.ones(1, 8)).backward()
assert raw.grad.abs().sum() == 0 and gate.grad.abs().sum() > 0
result = {'passed': True, 'scope': 'CPU enumeration/initial equivalence/masking/gradients/isolation',
          'parameters': sum(v.numel() for v in Network().parameters()), 'gpu_tested': False}
if args.cuda:
    from gpu_job_guard_v2 import check_slot
    print('GPU_ADMISSION', check_slot(), flush=True)
    model = Network().cuda(); torch.cuda.reset_peak_memory_stats()
    xx = torch.randn(64, 256, 206, device='cuda'); dd = torch.randn(64, 256, device='cuda')
    vv = torch.ones(64, 256, device='cuda', dtype=torch.bool)
    with torch.autocast('cuda', dtype=torch.bfloat16): outputs = model(xx, dd, vv, True, absolute_depth=dd)
    loss = objective(outputs, torch.randn(64, 256, 3, device='cuda'),
                     torch.rand(64, 256, 3, device='cuda') > .5, vv, True,
                     torch.ones_like(dd), epoch=0)
    loss.backward(); torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    result.update(gpu_tested=True, synthetic_gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2)
out = Path('outputs/pure_dl_joint_default_sequence'); out.mkdir(exist_ok=True)
(out / ('preflight.json' if args.cuda else 'preflight_cpu.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
