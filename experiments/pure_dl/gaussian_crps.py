"""Closed-form conditional CRPS for a univariate Gaussian mixture."""
import numpy as np
import torch


def absolute_normal_mean(delta, sigma):
    ratio = delta/sigma
    return sigma*np.sqrt(2/np.pi)*torch.exp(-.5*ratio.square())+delta*torch.erf(ratio/np.sqrt(2))


def mixture_crps(logw, mu, sigma, truth):
    weight = logw.exp()
    observed = (weight*absolute_normal_mean(truth[..., None]-mu, sigma)).sum(-1)
    delta = mu[..., :, None]-mu[..., None, :]
    pair_scale = torch.sqrt(sigma[..., :, None].square()+sigma[..., None, :].square())
    pair_weight = weight[..., :, None]*weight[..., None, :]
    dispersion = (pair_weight*absolute_normal_mean(delta, pair_scale)).sum((-2, -1))
    return observed-.5*dispersion


def self_test():
    from scipy.integrate import quad
    from scipy.stats import norm
    errors = []
    for truth in [-2., 0., 1., 3.]:
        weights = np.array([.2, .5, .3]); means = np.array([-1., 0., 2.]); scales = np.array([.2, .5, .8])
        cdf = lambda x: np.dot(weights, norm.cdf((x-means)/scales))
        expected = quad(lambda x: cdf(x)**2, -np.inf, truth, epsabs=1e-9)[0]
        expected += quad(lambda x: (1-cdf(x))**2, truth, np.inf, epsabs=1e-9)[0]
        logits = torch.tensor(np.log(weights), requires_grad=True)
        mu = torch.tensor(means, requires_grad=True); sd = torch.tensor(scales, requires_grad=True)
        actual = mixture_crps(logits.log_softmax(-1), mu, sd, torch.tensor(truth))
        errors.append(abs(actual.item()-expected)); assert errors[-1] < 1e-8
        actual.backward()
        assert all(torch.isfinite(x.grad).all() for x in [logits, mu, sd])
    center = mixture_crps(torch.zeros(1), torch.zeros(1), torch.ones(1), torch.tensor(0.))
    assert torch.allclose(center, center.new_tensor((np.sqrt(2)-1)/np.sqrt(np.pi)), atol=1e-7)
    return {'passed': True, 'independent_CDF_integral_cases': len(errors),
            'max_abs_error': max(errors), 'single_gaussian_identity_and_gradients': True}


if __name__ == '__main__':
    print(self_test())
