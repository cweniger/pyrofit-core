import pyro
import torch
import pyro.distributions as dist
from pyrofit.core import *
from pykeops.torch import LazyTensor

@register
class Entropy:
    def __init__(self, device = 'cpu'):
        self.device = device
        self.k_counter = 0

    @staticmethod
    def _kNN(x, y, K):
        """Get K nearest neighbours using keops"""
        x_i = LazyTensor(x[:, None, :])  # (M, 1, k)
        y_j = LazyTensor(y[None, :, :])  # (1, N, k)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml):
        print(a, b)
        N = 1000  # Number of sources
        Npix = 2000  # Number of image pixels
        SIGMA = 0.004  # Point source size
        kmin, kmax = 10, 100  # max NNs

        # Sample flux from normal distribution
        sigma = torch.tensor([0., 1.], device = self.device)
        sigma[0] += a
        x0 = torch.tensor([0., 0.], device = self.device)
        x = pyro.sample("x", dist.Normal(x0, sigma).expand_by((N,)))
        #b_real = pyro.sample("b_real", dist.Normal(b, 1.0))
#        pyro.sample("xmean", dist.Normal(x[:,0].mean(), 0.001), obs = torch.tensor(0., device = self.device))

        if self.k_counter == 0:
          self.ind_kNN = self._kNN(x, x, kmax)  # get sorted indices of NNs
          self.k_counter = 10
        self.k_counter -= 1

        # Calculate linear distance to 1+NN
        d = ((x[self.ind_kNN][...,:1,:] - x[self.ind_kNN][...,kmin:,:])**2).sum((2,))**0.5

        # Construct weights
        #w = torch.ones(kmax - kmin)
        w = torch.linspace(kmin, kmax-1, kmax - kmin, device = self.device)
        w /= w.sum()

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 2.0*(-torch.log(d)*w).sum()
        pyro.sample("fake", dist.Delta(torch.zeros(1, device = self.device), log_density = -entropy_loss), obs = torch.zeros(1, device = self.device))

        # Derive flux and position
        flux = x[:,0] #- x[:,0].mean()
        pos = torch.erf(x[:,1]/2**0.5)

        # Calculate observables
        grid = torch.linspace(-1, 1, Npix, device = self.device)

        d = torch.abs(pos.unsqueeze(1) - grid.unsqueeze(0))
        mu = (flux.unsqueeze(1)*torch.exp(-(d/SIGMA)**1)).sum(0) + b #_real

        noise = torch.ones_like(mu)*0.3
        obs = pyro.sample("obs", dist.Normal(mu, noise))
        #print("obs:", obs.mean())
        observe("mu", mu)
        observe("pos", pos)
        observe("flux", flux)
