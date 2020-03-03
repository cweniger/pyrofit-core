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
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml):
        N = 200  # Number of pixels 
        kmin, kmax = 10, 50

        # Fixed source positions
        x = torch.linspace(-1, 1, N)

        # Sample flux from standard normal distribution
        y = pyro.sample("y", dist.Normal(0., 1.).expand_by((N,)))

        # Transform y
        y = y.unsqueeze(1)

        # Construct vectors for distance calculation
        ind = self._kNN(y, y, kmax)

        # Calculate linear distance to kmin+ NNs
        d2 = (y[ind][...,:1,:] - y[ind][...,kmin:,:])**2
        d = d2[:,:,0]**0.5

        # Construct weights
        w = torch.linspace(kmin, kmax-1., kmax - kmin)
        w /= w.sum()

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 1.0*(-torch.log(d)*w).sum()

        # Workaround sample call to add entropy loss to log_density
        #pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        # Construct spectrum
        mu = (y*(a/b)*torch.exp(-(x.unsqueeze(0) - x.unsqueeze(1))**2/b**2/2)).sum(0)
        pyro.sample("obs", dist.Normal(mu, 0.1))
        observe("mu", mu)
