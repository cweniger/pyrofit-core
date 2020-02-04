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
        k = 50  # NNs
        kmin = 10

        # Sample flux from normal distribution
        x = pyro.sample("x", dist.Normal(0., 1).expand_by((N,))).unsqueeze(1)

        ind_kNN = self._kNN(x, x, k)  # get sorted indices of 50 NNs

        # Calculate linear distance to 1+NN
        d = ((x[ind_kNN][...,:1,:] - x[ind_kNN][...,kmin:,:])**2).sum((2,))**0.5

        # Construct weights
        w = torch.linspace(0, k, k)[kmin:].unsqueeze(0)**2
        w = w*torch.exp(-w/1000)
        w[:] = 1.
        w /= w.sum()

#        # Calculate effective k
#        d_i1 = d.unsqueeze(2)
#        d_1j = d.unsqueeze(1)
#        D = 2*(d_i1 - d_1j)/(d_i1 + d_1j)
#        keff = torch.sigmoid(D*10).sum(2) - 0.5 + kmin
#        # Construct weights according to effective k
#        w = (keff-kmin+1)**2 * torch.exp(-(keff/k)**2*5)
#        w = w/w.sum(1).unsqueeze(1)

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 1.0*(-torch.log(d)*w).sum()
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        noise = torch.ones_like(x)*0.2
        mu = x*a
        obs = pyro.sample("obs", dist.Normal(mu, noise))
        res = (obs - mu)
        observe("mu", mu)
        print(a)

        #ind_kNN = self._kNN(res, res, k)  # get sorted indices of 50 NNs

        #d = ((res[ind_kNN][...,:1,:] - res[ind_kNN][...,4:,:])**2).sum((2,))**0.5

        ## Construct weights
        #w = torch.linspace(0, 20, 20)[4:].unsqueeze(0)**2
        #w = w*torch.exp(-w/200)
        #w /= w.sum()

        ### Loss function obtained from weighted sum of log of distances
        #entropy_loss = 1.0*(-torch.log(d)*w).sum()
        #pyro.sample("fake2", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))
