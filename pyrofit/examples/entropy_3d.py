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
        N = 100  # Number of points
        Npix = 100  # Number of image pixels
        SIGMA = 0.2  # Point source size
        ndim = 3  # number of dimensions
        k = 20  # NNs

        # Sample flux from normal distribution
        sigma = torch.tensor([1., 1., 1.])
        x = pyro.sample("x", dist.Normal(0., sigma).expand_by((N,)))

        ind_kNN = self._kNN(x, x, k)  # get sorted indices of 50 NNs

        # Calculate linear distance to 1+NN
        d = ((x[ind_kNN][...,:1,:] - x[ind_kNN][...,2:,:])**2).sum((2,))**0.5
#        d_i1 = d.unsqueeze(2)
#        d_1j = d.unsqueeze(1)
#        D = 2*(d_i1 - d_1j)/(d_i1 + d_1j)
#        keff = torch.sigmoid(D*30).sum(2)
#        print(keff.shape)
#        print(keff[0])
#        print(d[0])
#        quit()

        # Construct weights
        w = torch.linspace(0, 20, 20)[2:].unsqueeze(0)**2
        w = w*torch.exp(-w/20)
        w /= w.sum()
        #w *= 0
        #w[0,2] = 1.

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 0*3.0*(-torch.log(d)*w).sum() #+ a*1e1
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        # Workaround sample call to add entropy loss to log_density
        #print(x.std(dim=0))
        #u = torch.erf(x/2**0.5)
        #pyro.sample("uobs", dist.Normal(0, .1), obs = u)
        #observe("u", u)

        # Calculate observables
        grid = torch.linspace(-1, 1, Npix)
        X, Y = torch.meshgrid(grid, grid)
        flux = x[:,2] * a

        d2 = (x[:,0].unsqueeze(1).unsqueeze(1) - X.unsqueeze(0))**2 + (x[:,1].unsqueeze(1).unsqueeze(1) - Y.unsqueeze(0))**2
        mu = (flux.unsqueeze(1).unsqueeze(1)*torch.exp(-d2/SIGMA**2)).sum(0)

        noise = torch.ones_like(X)*0.3
        #noise[:,25:] *= 100
        pyro.sample("obs", dist.Normal(mu, noise))
        #print(a)
        observe("mu", mu)
