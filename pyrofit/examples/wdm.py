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
        N = 100  # Number of sources in the model
        Nbins = 1000  # Number of bins
        D_MIN = 1e-10  # Entropy cutoff

        # Fixed source positions
        x_pos = torch.linspace(-4, 4, N)
        #x_pos = pyro.sample("x_pos", dist.Normal(0., 1.0).expand_by((N,)))

        # Sample flux from normal distribution
        flux_normal = pyro.sample("x_f", dist.Normal(0., 1.).expand_by((N,)))

        # Calculate physical flux (arbitrary rescaling function)
        flux_phys = 10**(flux_normal*0.5 - 1.5)

        # Construct vectors for distance calculation
        x1 = torch.stack([x_pos*0, flux_normal], dim = 1)
        #x2 = torch.stack([x_pos, x_f], dim = 1)

        D1 = self._kNN(x1, x1, 50)  # get sorted indices of 50 NNs

        # Calculate linear distance to 4+ NN
        d2_1 = ((x1[D1][...,:1,:] - x1[D1][...,4:,:])**2).sum((2,))
        d_1 = (d2_1 + D_MIN**2)**0.5

        #D2 = self._kNN(x2, x2, 50)
        #d = ((x[D][...,0,0] - x[D][...,1,0])**2 + D_MIN**2)**0.5
        #d2_2 = ((x2[D2][...,:1,:] - x2[D2][...,4:,:])**2).sum((2,))
        #d_2 = (d2_2 + D_MIN**2)**0.5
        #loss = -1*torch.log(d).sum()/35  * 1e-0

        # Construct weights
        w = torch.linspace(0, 50, 50)[4:].unsqueeze(0)**2
        w = w*torch.exp(-w/200)
        w /= w.sum()

        # Loss function obtained from weighted sum of log of distances
        entropy_loss = 1.0*(-torch.log(d_1)*w).sum()

        #entropy_loss = 1.0*(-2*torch.log(d_2)*w).sum()
        #loss += 600*torch.log(a)
    
        # Workaround sample call to add entropy loss to log_density
        pyro.sample("fake", dist.Delta(torch.zeros(1), log_density = -entropy_loss), obs = torch.zeros(1))

        # Construct physical spectrum
        x_grid = torch.linspace(-4, 4, Nbins)

        # !!! sigma becomes large if the physical flux drops below the value of a
        sigma = 0.1 + 0.4*torch.sigmoid(-torch.log(flux_phys/a)*4)

        flux = (0.1*flux_phys.unsqueeze(1)/sigma.unsqueeze(1)*torch.exp(-(x_grid.unsqueeze(0) -
            x_pos.unsqueeze(1))**2/sigma.unsqueeze(1)**2)).sum(0)+b
        noise = torch.ones(Nbins)*0.1
        #noise[500:] *= 6
        pyro.sample("flux", dist.Normal(flux, noise))
        observe("obs_flux", flux)
