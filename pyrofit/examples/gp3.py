import pyro
import torch
import pyro.distributions as dist
from pyrofit.core import *
from pykeops.torch import LazyTensor
import random

@register
class Entropy:
    def __init__(self, device = 'cpu'):
        self.device = device
        self.D = 0.01

    @staticmethod
    def _kNN(x, y, K):
        """Get K nearest neighbours using keops"""
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml, c:Yaml, use_entropy: Yaml=True):
        N = 200  # Number of pixels 
        L = 2  # Number of layers
        kmin, kmax = 1, 20

        # Fixed source positions
        x = torch.linspace(-1+1.0/N, 1-1.0/N, N, device = self.device)

        # Sample flux from standard normal distribution
        sigma = torch.tensor(1., device = self.device)
        y = pyro.sample("y", dist.Normal(0., sigma).expand_by((N*L,)))

        mode = 1

        if use_entropy and mode == 1:
            xr = x.repeat(2)
            xr[200:] += 100
            xy = torch.stack([y*0, xr], dim = 1)
            xy2 = torch.stack([y*0, y], dim = 1)

            # Construct vectors for distance calculation
            ind = self._kNN(xy, xy, kmax)

            # Calculate linear distance to kmin+ NNs
            d2 = (xy2[ind][...,:1,:] - xy2[ind][...,kmin:,:])**2
            d = (d2.sum(2)+self.D**2)**0.5

            # Construct weights
            w = torch.ones(kmax - kmin, device = self.device)
            #w = torch.linspace(kmin, kmax, kmax - kmin, device = self.device)
            #w = torch.rand(kmax - kmin, device = self.device)
            w /= w.sum()

            # Loss function obtained from weighted sum of log of distances
            entropy_loss = 1.0*(-torch.log(d)*w).sum() #+ 2*torch.log(a)

            # Workaround sample call to add entropy loss to log_density
            pyro.sample("fake", dist.Delta(torch.zeros(1, device = self.device), log_density = -entropy_loss), obs = torch.zeros(1, device = self.device))

        # Construct spectrum

        # First layer
        yu = y[:N].unsqueeze(1)
        mu = (yu*(a/b)*torch.exp(-(x.unsqueeze(0) - x.unsqueeze(1))**2/b**2/2)).sum(0)

        # Second layer
        mu  += c*y[N:2*N]

        obs = pyro.sample("obs", dist.Normal(mu, 0.01))
        #observe("mu", mu)
        print(a, b, c)
