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
        N = 100  # Number of sources
        Npix = 200  # Number of image pixels
        SIGMA = 0.04  # Point source size
        kmin, kmax = 10, 50  # max NNs

        # Sample flux from normal distribution
        sigma = torch.tensor([1., 1.,])
        x = pyro.sample("x", dist.Normal(0., sigma).expand_by((N,)))

        ind_kNN = self._kNN(x, x, kmax)  # get sorted indices of NNs

        # Calculate linear distance to 1+NN
        d = ((x[ind_kNN][...,:1,:] - x[ind_kNN][...,kmin:,:])**2).sum((2,))**0.5

        # Construct weights
        #w = torch.ones(kmax - kmin)
        w = torch.linspace(kmin, kmax-1, kmax - kmin)
        w /= w.sum()

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 2.0*(-torch.log(d)*w).sum()
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        # Derive flux and position
        flux = x[:,0]*a
        pos = torch.erf(x[:,1]/2**0.5)

        # Calculate observables
        grid = torch.linspace(-1, 1, Npix)

        d = torch.abs(pos.unsqueeze(1) - grid.unsqueeze(0))
        mu = (flux.unsqueeze(1)*torch.exp(-(d/SIGMA)**1)).sum(0) + b

        noise = torch.ones_like(mu)*0.3
        pyro.sample("obs", dist.Normal(mu, noise))
        observe("mu", mu)
        observe("pos", pos)
        observe("flux", flux)
