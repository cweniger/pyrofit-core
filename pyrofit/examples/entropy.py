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
        x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
        y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
        D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
        return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

    def __call__(self, a:Yaml, b:Yaml):
        #print(a)
        #a = a*0. + 0.9
        N = 4000
        x_pos2 = pyro.sample("x_pos", dist.Normal(0., 1.0).expand_by((N,)))
        #x_pos = torch.linspace(-4, 4, N)
        x_pos = x_pos2*b
        x_f = pyro.sample("x_f", dist.Normal(0., 1.).expand_by((N,)))
        x_f2 = x_f*a - 1.5
        x_flux = 10**x_f2
        #x_flux = pyro.sample("x_f", dist.LogNormal(-1.5, a).expand_by((N,)))
        #print(x_flux)
        print(a, b)

        x1 = torch.stack([x_pos*0, x_f], dim = 1)
        x2 = torch.stack([x_pos, x_f], dim = 1)

        D1 = self._kNN(x1, x1, 50)
        D2 = self._kNN(x2, x2, 50)
        d_min = 1e-10
        #d = ((x[D][...,0,0] - x[D][...,1,0])**2 + d_min**2)**0.5
        d2_1 = ((x1[D1][...,:1,:] - x1[D1][...,4:,:])**2).sum((2,))
        d2_2 = ((x2[D2][...,:1,:] - x2[D2][...,4:,:])**2).sum((2,))
        d_1 = (d2_1 + d_min**2)**0.5
        d_2 = (d2_2 + d_min**2)**0.5
        #loss = -1*torch.log(d).sum()/35  * 1e-0
        w = torch.linspace(0, 50, 50)[4:].unsqueeze(0)
        w = w*torch.exp(-w/5)
        w /= w.sum()
        #loss = 1.0*(-torch.log(d_1)*w).sum()
        loss = 1.0*(-2*torch.log(d_2)*w).sum()
        #loss += 600*torch.log(a)
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -loss), obs = torch.zeros(1))

        ## Observation 1
        #pyro.sample("obs", dist.Normal(1., 1.), obs = x)

        x_grid = torch.linspace(-5, 5, 1000)
        flux = (x_flux.unsqueeze(1)*torch.exp(-(x_grid.unsqueeze(0) -
            x_pos.unsqueeze(1))**2/0.1**2)).sum(0)
        noise = torch.ones(1000)
        noise[500:] *= 6
        #noise *= 1e-10
        pyro.sample("flux", dist.Normal(flux, noise))
