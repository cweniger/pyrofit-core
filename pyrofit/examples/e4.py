import pyro
import torch
import pyro.distributions as dist
from pyrofit.core import *
from pykeops.torch import LazyTensor

@register
class Entropy:
    def __init__(self, device = 'cpu'):
        pass

    @staticmethod
    def _kNN(x, y, K):
        """Get K nearest neighbours using keops"""
        x_i = LazyTensor(x[:, None, :])  # (M, 1, k)
        y_j = LazyTensor(y[None, :, :])  # (1, N, k)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml):
        N = 200  # Number of points
        k = 20  # NNs

        # Sample flux from normal distribution
        x = pyro.sample("x", dist.Normal(0., 1).expand_by((N,))).unsqueeze(1)

        ind_kNN = self._kNN(x, x, k)  # get sorted indices of 50 NNs

        # Calculate linear distance to 1+NN
        d = ((x[ind_kNN][...,:1,:] - x[ind_kNN][...,2:,:])**2).sum((2,))**0.5

        # Construct weights
        w = torch.linspace(0, 20, 20)[2:].unsqueeze(0)**2
        w = w*torch.exp(-w/20)
        w /= w.sum()

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 1.0*(-torch.log(d)*w).sum()
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        noise = torch.ones_like(x)*2.0
        mu = x*a
        obs = pyro.sample("obs", dist.Normal(mu, noise))
        observe("mu", mu)
