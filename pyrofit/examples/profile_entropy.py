import pyro
import torch
import pyro.distributions as dist
from pykeops.torch import LazyTensor
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import pylab as plt



def _kNN(x, y, K):
    """Get K nearest neighbours using keops"""
    x_i = LazyTensor(x[:, None, :])  # (M, 1, 2)
    y_j = LazyTensor(y[None, :, :])  # (1, N, 2)
    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (M, N) symbolic matrix of squared distances
    return D_ij.argKmin(K, dim=1)  # (M, K) Minimum indices

def test():
    N = 300
    x = pyro.sample("x", dist.Normal(0., 1.).expand_by((N,)))
    x = x.unsqueeze(1)

    kmin, kmax = 2, 10
    k = torch.linspace(kmin, kmax-1, kmax-kmin)
    w = k
    #w = k*torch.exp(-(k-20)**2/2./5.**2)
    #w[:] = 1.
    w = w/w.sum()

    out = []

    for x0 in np.linspace(-1.1, 1.1, 1000):
        x[0,0] = x0
        ind = _kNN(x, x, kmax)

        d = ((x[ind][...,:1,:] - x[ind][...,kmin:,:])**2).sum((2,))

        loss = 1.0*(-torch.log(d)*w).sum()
        out.append([x0, loss.numpy()])

    out = np.array(out)
    plt.plot(out[:,0], out[:,1])
    plt.savefig("e.pdf")

test()
