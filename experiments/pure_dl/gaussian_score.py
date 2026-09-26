"""Differentiable triangular official-score expectation under Gaussian components."""
import numpy as np
import torch


def interval(mu, sigma, lower, upper, exponential=False, offset=0.):
    a, b = (lower-mu)/sigma, (upper-mu)/sigma
    probability = (torch.special.ndtr(b)-torch.special.ndtr(a)).clamp_min(0)
    if exponential:
        # +/-60 is an overflow guard, inactive for normal training-scale values.
        moment = (mu+.5*sigma.square()).clamp(-60, 60).exp()*(
            torch.special.ndtr(b-sigma)-torch.special.ndtr(a-sigma))
        moment = moment-offset*probability
    else:
        phi = lambda z: torch.exp(-.5*z.square())/np.sqrt(2*np.pi)
        moment = mu*probability+sigma*(phi(a)-phi(b))
    return probability, moment


def component_score(mu, sigma, truth, target):
    """mu/sigma in transformed physical units, truth in original physical units."""
    if target == 1:
        center = truth.log10()
        low, high = center-1, center+1
        p1, m1 = interval(mu, sigma, low, center)
        p2, m2 = interval(mu, sigma, center, high)
        return ((1-center)*p1+m1+(1+center)*p2-m2).clamp(0, 1)
    tolerance = (.08 if target == 0 else .05)*(truth.abs()+1e-5)
    offset = 1. if target == 0 else 0.
    low = (truth-tolerance+offset).clamp_min(1e-20).log()
    center = (truth+offset).log()
    high = (truth+tolerance+offset).log()
    p1, m1 = interval(mu, sigma, low, center, True, offset)
    p2, m2 = interval(mu, sigma, center, high, True, offset)
    return ((1-truth/tolerance)*p1+m1/tolerance+
            (1+truth/tolerance)*p2-m2/tolerance).clamp(0, 1)


def mixture_score(logw, mu, sigma, targets, median, scale):
    m = mu*scale[None, None, :, None]+median[None, None, :, None]
    s = sigma*scale[None, None, :, None]
    z = targets*scale[None, None, :]+median[None, None, :]
    truth = [z[:, :, 0].expm1(), 10**z[:, :, 1], z[:, :, 2].exp()]
    return torch.stack([(logw[:, :, j].exp()*component_score(
        m[:, :, j], s[:, :, j], truth[j][:, :, None], j)).sum(-1) for j in range(3)], -1)


def self_test():
    from scipy.integrate import quad
    from scipy.stats import norm
    errors = []
    for target, truth in enumerate([12., 20., 60.]):
        center = [np.log1p, np.log10, np.log][target](truth)
        for delta in [-.2, 0., .2]:
            for sd in [.03, .2, .8]:
                mu = torch.tensor(center+delta, dtype=torch.float64, requires_grad=True)
                sigma = torch.tensor(sd, dtype=torch.float64, requires_grad=True)
                actual = component_score(mu, sigma, torch.tensor(truth, dtype=torch.float64), target)
                if target == 1:
                    lower, middle, upper = center-1, center, center+1
                    score = lambda z: max(0, 1-abs(z-center))
                else:
                    tol = (.08 if target == 0 else .05)*(truth+1e-5)
                    offset = 1 if target == 0 else 0
                    lower, middle, upper = np.log([truth-tol+offset, truth+offset, truth+tol+offset])
                    score = lambda z: max(0, 1-abs(np.exp(z)-offset-truth)/tol)
                f = lambda z: score(z)*norm.pdf(z, loc=center+delta, scale=sd)
                expected = quad(f, lower, middle, epsabs=1e-11)[0]+quad(f, middle, upper, epsabs=1e-11)[0]
                errors.append(abs(actual.item()-expected))
                assert errors[-1] < 1e-9
                actual.backward()
                assert torch.isfinite(mu.grad) and torch.isfinite(sigma.grad)
    return {'passed': True, 'scipy_quadrature_cases': len(errors), 'max_abs_error': max(errors),
            'mean_and_scale_gradients_finite': True}


if __name__ == '__main__':
    print(self_test())
