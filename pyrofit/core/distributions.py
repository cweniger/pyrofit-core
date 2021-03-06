#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi, log, gamma
import typing

import numpy as np
import torch
from torch.distributions.transformed_distribution import (
    TransformedDistribution)
from torch.distributions import Normal, TransformedDistribution, Transform
from torch.distributions.transforms import PowerTransform, ExpTransform
import warnings

import pyro
import pyro.distributions as dist
from torch.distributions import constraints

from .utils import kNN_d2, kNN, broadcast_index, moveaxis

try:
    from torchinterp1d import Interp1d
except ModuleNotFoundError:
    print("WARNING: InverseTransformSampling not available!")
    print("         please install https://github.com/aliutkus/torchinterp1d")


class _TruncatedPower(TransformedDistribution):
    arg_constraints = {'low': constraints.positive, 'high': constraints.positive}
    has_rsample = True

    def __init__(self, low, high, alpha):
        if alpha == -1.:
            raise ValueError("Not implemented for alpha = -1")
        self.support = constraints.interval(low**(alpha+1), high**(alpha+1))
        base_dist = torch.distributions.Uniform(low**(alpha+1), high**(alpha+1))
        super(_TruncatedPower, self).__init__(base_dist, [PowerTransform(1/(alpha+1))])

    def __call__(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape=sample_shape)

class TruncatedPower(_TruncatedPower, dist.torch_distribution.TorchDistributionMixin):
    pass

class InverseTransformSampling(dist.TorchDistribution):
    """Flexible pyro 1-dim distribution, based on inverse transform sampling."""
    has_rsample = True

    def __init__(self, log_prob, grid, expand_shape = torch.Size([])):
        self._log_prob = log_prob
        self.device = 'cpu' if not grid.is_cuda else grid.get_device()
        self._grid = grid
        self._event_shape = torch.Size([])  # all variables are independent (but may have different pdfs)
        self._prob_shape = log_prob(grid[0]).shape  # shape of log_prob (pdfs might differ)
        self._expand_shape = expand_shape
        self._batch_shape = self._expand_shape + self._prob_shape
        N = grid.shape[0]  # Num grid points for tabulating log_prob

        # Expand grid to match (N,) + prob_shape
        if len(grid.shape) == 1:
            grid = grid.reshape((N,)+(1,)*len(self._prob_shape))
        grid = grid.expand((grid.shape[0],)+self._prob_shape)

        # TODO: tensor shapes correct?
        self._support = constraints.interval(grid[0], grid[-1])  # define finite support

        cdf, grid, norm = self._get_cdf(log_prob, grid)
        self.D = self._prob_shape.numel()  # Number of pdfs
        self.R = self.batch_shape.numel()  # Number of batch evaluations
        self.R_D = int(self.R/self.D)      # Batch evaluations with identical set of pdfs
        self.x = cdf.reshape(N, self.D).permute(1, 0)   # Prepare for interp1d
        self.y = grid.reshape(N, self.D).permute(1, 0)  # Prepare for interp1d
        self.log_scale = torch.log(norm)

        self.interp1d = Interp1d()

    @property
    def arg_constraints(self):
        return {}

    @property
    def support(self):
        return self._support

    @property
    def mean(self):
        x = (self._grid[:-1] + self._grid[1:]) / 2
        dx = (self._grid[1:] - self._grid[:-1])
        return (x * self.log_prob(x).exp() * dx).sum() / (self.log_prob(x).exp() * dx).sum()


    @staticmethod
    def _get_cdf(log_prob, grid, th = 0.):
        log_prob_grid = log_prob(grid)
        prob_grid = torch.exp(log_prob_grid)
        dgrid = grid[1:] - grid[:-1]
        dp = dgrid*(prob_grid[1:]+prob_grid[:-1])/2
        dp = torch.cat((torch.zeros_like(dp[:1]), dp), 0)
        dp_cumsum = torch.cumsum(dp, 0)
        norm = dp_cumsum[-1]
        cdf = dp_cumsum/norm
        if (cdf[1:] - cdf[:-1]).min() == 0.:
            warnings.warn("WARNING: PDF inversion beyond machine precision.")
        return cdf, grid, norm
        #mask = torch.cat((torch.ByteTensor([0]), (cdf[1:]-cdf[:-1]) > th), 0)
        #return cdf[mask], grid[mask], norm

    def cdf(self, y):
        return self.interp1d(self.y, self.x, y).reshape(y.shape)

    def icdf(self, x):
        return self.ppf(x).reshape(x.shape)

    def ppf(self, x, right_sample_dims_count=0):
        """

        Parameters
        ----------
        x: torch.tensor
            the shape should either be (self._prob_shape..., ...) or
            (..., self._prob_shape..., sample dims...), where there are
            right_sample_dims_count sample dimensions to the right of
            self._prob_shape.
        right_sample_dims_count

        Returns
        -------
            The ppf at the given input, using the correctly broadcasted pdfs.
        """

        permute = self._prob_shape != x.shape[:len(self._prob_shape)]
        _x = (x.permute(*range(-len(self._prob_shape), -right_sample_dims_count),
                        *range(len(x.shape) - right_sample_dims_count - len(self._prob_shape)),
                        *range(-right_sample_dims_count, 0))
              if permute else x)

        res = self.interp1d(self.x, self.y,
                            _x.reshape((self.D, -1))
                            ).reshape(_x.shape)
        return (res.permute(*range(len(self._prob_shape), len(x.shape) - right_sample_dims_count),
                            *range(len(self._prob_shape)),
                            *range(-right_sample_dims_count, 0))
                if permute else res).reshape(x.shape)

    def rsample(self, sample_shape=torch.Size()):
        P = torch.Size(sample_shape).numel()
        xnew = torch.rand(self.D, P*self.R_D, device=self.device)
        out = self.ppf(xnew)  # (D, P*R/D)
        return out.permute(1, 0).reshape(self.shape(sample_shape))

    def __call__(self, *args, **kwargs):
        return self.rsample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self._log_prob(*args, **kwargs) - self.log_scale

    def expand(self, batch_shape, **kwargs):
        if len(self._prob_shape) > 0:
            assert batch_shape[-len(self._prob_shape):] == self._prob_shape
            expand_shape = batch_shape[:-len(self._prob_shape)]
        else:
            expand_shape = batch_shape
        return type(self)(self._log_prob, grid=self._grid, expand_shape=expand_shape)


class GaussianSampler(InverseTransformSampling):
    def __init__(self, log_prob,
                 rng: tuple, grid_size=201, logspaced=True, grid=None,
                 expand_shape=torch.Size([]), device='cpu'):
        grid = (torch.logspace if logspaced else torch.linspace)(*rng, grid_size, device=device) if grid is None else grid
        super().__init__(log_prob, grid, expand_shape=expand_shape)
        self.standard_normal = dist.Normal(
            torch.tensor(0., device=self.device),
            torch.tensor(1., device=self.device)
        ).expand(self.batch_shape)

    def draw(self, name: str, sample_shape: typing.Union[torch.Size, typing.Sequence[int]]):
        return pyro.sample(name, self.standard_normal.expand_by(sample_shape).to_event(len(sample_shape)))

    def transform_sample(self, sample):
        return self.ppf(self.standard_normal.cdf(sample)).reshape(sample.shape)

    def inverse_transform(self, values):
        return self.standard_normal.icdf(moveaxis(self.cdf(values.reshape(-1, values.shape[-1])).reshape(*self.standard_normal.batch_shape, values.shape[-1]), -1, 0))

    def sample(self, name: str, sample_shape: typing.Union[torch.Size, typing.Tuple[int]]):
        return self.transform_sample(self.draw(name, sample_shape))


class CDFTransform(Transform):
    """
    Makes a cdf Transform from a distribution.
    """

    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __init__(self, base_dist):
        self.base_dist = base_dist
        self.domain = base_dist.support
        super().__init__()

    def __eq__(self, other):
        return (
            isinstance(other, CDFTransform) and
            other.base_dist == self.base_dist
        )

    def _call(self, x):
        return self.base_dist.cdf(x)

    def _inverse(self, y):
        return self.base_dist.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.base_dist.log_prob(x)


class Entropy:
    def __init__(self, k=50, kmin=4, scale=20, power=0.33, device='cpu'):
        self.kmin, self.k = kmin, k
        self.weights = self.get_weights(k, kmin, scale, power, device=device)
        self.device = device

    @staticmethod
    def get_weights(k=50, kmin=4, scale=20, power=1., device='cpu'):
        w = torch.arange(float(kmin), float(k)+1)
        w = w**power * torch.exp(-w / scale)
        weights = torch.zeros(k, device=device)
        weights[kmin-1:] = w / w.sum()
        return weights

    @staticmethod
    def ballfactor(ndim):
        return pi**(ndim/2) / gamma(1 + ndim/2)

    def normalising_factor(self, N, ndim):
        return ((torch.digamma(torch.arange(1., self.k+1, device=self.device)) * self.weights).sum()
                - torch.digamma(torch.tensor(float(N)))
                - log(self.ballfactor(ndim)))

    def d2(self, x):
        return kNN_d2(x, x, len(self.weights) + 1)[..., :, 1:]  # [0] was the point itself

    def log_p(self, x, d_min=1e-3):
        ndim = x.shape[-1]
        return ndim / 2 * (self.weights * torch.log(self.d2(x) + d_min**2)).sum(-1)

    def entropy_loss(self, x, d_min=1e-3):
        return - self.log_p(x, d_min).sum(-1)

    def full_entropy(self, x, d_min=1e-3):
        N, ndim = x.shape[-2:]
        return self.log_p(x, d_min).mean(-1) - self.normalising_factor(N, ndim)


class ConstantNeighboursEntropy(Entropy):
    def __init__(self, *args, x0, **kwargs):
        super().__init__(*args, **kwargs)
        self.i = kNN(x0, x0, len(self.weights)+1)

    def d2(self, x):
        return (x.unsqueeze(-2) - broadcast_index(x, self.i)).pow(2.).sum(-1)[..., :, 1:]


def test1():
    import pyro
    import pylab as plt
    x = pyro.sample("x", TruncatedPower(torch.ones(100000), 100., -0.2)).numpy()
    plt.hist(x, bins = 100)
    plt.show()

def test2():
    import pyro
    import pylab as plt
    alpha = torch.tensor(0.0, requires_grad = True)
    log_prob = lambda x: x**alpha
    grid = torch.logspace(-1, 1, 100)
    x = pyro.sample("x", InverseTransformSampling(log_prob, grid).expand_by((10000,)))
    x.sum().backward()
    print(alpha.grad)
    plt.hist(x.detach().numpy())
    plt.show()

if __name__ == "__main__":
    test2()
