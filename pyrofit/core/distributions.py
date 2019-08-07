#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.distributions.transformed_distribution import (
    TransformedDistribution)
from torch.distributions import Normal
from torch.distributions.transforms import PowerTransform, ExpTransform
import pyro.distributions as dist

class _TruncatedPower(TransformedDistribution):
    def __init__(self, low, high, alpha):
        if alpha == -1.:
            raise ValueError("Not implemented for alpha = -1")
        base_dist = torch.distributions.Uniform(low**(alpha+1), high**(alpha+1))
        super(_TruncatedPower, self).__init__(base_dist, [PowerTransform(1/(alpha+1))])

    def __call__(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape=sample_shape)

class TruncatedPower(_TruncatedPower, dist.torch_distribution.TorchDistributionMixin):
    pass

class InverseTransformSampling(dist.TorchDistribution):
    """Flexible pyor.ai 1-dim distribution, based on inverse transform sampling."""
    has_rsample = True

    def __init__(self, log_prob, grid, expand_shape = torch.Size([])):
        self._log_prob = log_prob
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
        self._support = constr.interval(grid[0], grid[-1])  # define finite support

        cdf, grid, norm = self._get_cdf(log_prob, grid)
        self.D = np.prod(tuple(self._prob_shape), dtype = np.int32)  # Number of pdfs
        self.R = np.prod(tuple(self.batch_shape), dtype = np.int32)  # Number of batch evaluations
        self.R_D = int(self.R/self.D)  # Batch evaluations with identical set of pdfs
        self.x = cdf.reshape(N, self.D).permute(1, 0)  # Prepare for interp1d
        self.y = grid.reshape(N, self.D).permute(1, 0)  # Prepare for interp1d
        self.log_scale = torch.log(norm)

        self.interp1d = Interp1d()

    @property
    def support(self):
        return self._support

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
            print("WARNING: PDF inversion beyond machine precision.")
        return cdf, grid, norm
        #mask = torch.cat((torch.ByteTensor([0]), (cdf[1:]-cdf[:-1]) > th), 0)
        #return cdf[mask], grid[mask], norm

    def rsample(self, sample_shape = torch.Size()):
        P = np.prod(tuple(sample_shape), dtype = np.int16)
        xnew = torch.rand(self.D, P*self.R_D)
        out = self.interp1d(self.x, self.y, xnew)  # (D, P*R/D)
        return out.permute(1,0).reshape(self.shape(sample_shape))

    def __call__(self, *args, **kwargs):
        return self.rsample(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        return self._log_prob(*args, **kwargs) - self.log_scale
    
    def expand(self, batch_shape):
        if len(self._prob_shape) > 0:
            assert batch_shape[-len(self._prob_shape):] == self._prob_shape
            expand_shape = batch_shape[:-len(self._prob_shape)]
        else:
            expand_shape = batch_shape
        return type(self)(self._log_prob, self._grid, expand_shape = expand_shape)

#def test():
#    import pyro
#    import pylab as plt
#    x = pyro.sample("x", TruncatedPower(torch.ones(100000), 100., -0.2)).numpy()
#    plt.hist(x, bins = 100)
#    plt.show()
#test()
