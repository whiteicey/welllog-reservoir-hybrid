"""Two-pass L2 SAM with exact weight restoration and matched dropout masks.

Reference: https://arxiv.org/abs/2010.01412
Uses AdamW as the base optimizer; rho is fixed before heldout evaluation.
"""
import json
import torch


def sam_step(model, optimizer, closure, rho=.05, max_grad_norm=2.):
    parameters = [p for p in model.parameters() if p.requires_grad]
    devices = sorted({p.device.index for p in parameters if p.is_cuda})
    cpu_rng = torch.get_rng_state()
    gpu_rng = {d: torch.cuda.get_rng_state(d) for d in devices}
    optimizer.zero_grad(set_to_none=True)
    first_loss = closure()
    assert torch.isfinite(first_loss)
    first_loss.backward()
    active = [p for p in parameters if p.grad is not None]
    gradients = [p.grad for p in active]
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
    assert torch.isfinite(norm)
    originals = [p.detach().clone() for p in active]
    try:
        with torch.no_grad():
            perturbations = torch._foreach_mul(gradients, rho/(norm+1e-12))
            torch._foreach_add_(active, perturbations)
            del perturbations
        optimizer.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng)
            for d in devices:
                torch.cuda.set_rng_state(gpu_rng[d], d)
            second_loss = closure()
            assert torch.isfinite(second_loss)
            second_loss.backward()
    finally:
        with torch.no_grad():
            torch._foreach_copy_(active, originals)
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return second_loss.detach()


def self_test():
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad(): model.weight.copy_(torch.tensor([[3., 4.]]))
    initial = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    expected = initial-.1*(initial+.05*initial/initial.norm())
    sam_step(model, optimizer, lambda: .5*model.weight.square().sum(), max_grad_norm=100.)
    assert torch.allclose(model.weight, expected, atol=1e-8)
    # An exception on the perturbed pass must never leave modified parameters.
    before = model.weight.detach().clone(); calls = []
    def failing():
        calls.append(1)
        if len(calls) == 2: raise ValueError('test')
        return model.weight.square().sum()
    try: sam_step(model, optimizer, failing)
    except ValueError: pass
    assert torch.equal(before, model.weight)
    draws = []
    def stochastic():
        mask = torch.rand_like(model.weight); draws.append(mask.detach().clone())
        return (model.weight*mask).square().sum()
    sam_step(model, optimizer, stochastic)
    assert torch.equal(draws[0], draws[1])
    return {'passed': True, 'quadratic_update_matches_analytic_SAM': True,
            'exception_restores_exact_weights': True, 'two_passes_use_same_random_masks': True,
            'rho': .05}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
