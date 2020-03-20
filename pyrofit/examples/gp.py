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
        self.pref = 0.00001

    @staticmethod
    def _kNN(x, y, K):
        """Get K nearest neighbours using keops"""
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml, c:Yaml, use_entropy: Yaml=True):
        N = 200  # Number of pixels 
        kmin, kmax = 1, 5

        # Fixed source positions
        x = torch.linspace(-1+1.0/N, 1-1.0/N, N, device = self.device)

        # Sample flux from standard normal distribution
        sigma = torch.tensor(1., device = self.device)
        y = pyro.sample("y", dist.Normal(0., sigma).expand_by((N,)))

        use_entropy = True

        if self.pref < 1.:
            self.pref *= 1.003

        if use_entropy:
            # Transform y
            # phi-1(p) = sqrt(2)*erfinv(2p-1)
            xnorm = 2**0.5*torch.erfinv(x)
            #print(y.std().detach().cpu().numpy(), (y**2).std().detach().cpu().numpy())
            xy = torch.stack([xnorm, y], dim = 1)

            # Construct vectors for distance calculation
            ind = self._kNN(xy, xy, kmax)

            # Calculate linear distance to kmin+ NNs
            d2 = (xy[ind][...,:1,:] - xy[ind][...,kmin:,:])**2
            d = (d2.sum(2))**0.5

            # Construct weights
            w = torch.linspace(kmin, kmax-1., kmax - kmin, device = self.device)
            w /= w.sum()

            # Loss function obtained from weighted sum of log of distances
            entropy_loss = self.pref*2.0*(-torch.log(d)*w).sum() #+ 2*torch.log(a)
            print(self.pref, entropy_loss)

            # Workaround sample call to add entropy loss to log_density
            pyro.sample("fake", dist.Delta(torch.zeros(1, device = self.device), log_density = -entropy_loss), obs = torch.zeros(1, device = self.device))

        # Construct spectrum
        y = y.unsqueeze(1)
        mu = (y*(a/b)*torch.exp(-(x.unsqueeze(0) - x.unsqueeze(1))**2/b**2/2)).sum(0)
        mu = mu + y[:,0]*c
        pyro.sample("obs", dist.Normal(mu, 0.1))
        observe("mu", mu)
        #print(a, b, c)
