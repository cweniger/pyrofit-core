import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyrofit.core import *
from pyrofit.core.guides import PyrofitGuide

@register
class Model:
    """A linear function."""
    def __init__(self, device = 'cpu'):
        self.N = 10000  # Length of data vector
        self.M = 1000  # Number of nuisance parameters y

        self.std = torch.linspace(0.001, 0.01, self.M)*0.001
        self.T = torch.ones(self.N, self.M)

        self.w = torch.zeros(self.N)
        self.w[0] += 1.

        self.sigma = 0.01  # Measurement error of Gaussian likelihood

        self.device = device

    def __call__(self, z:Yaml, observations = {}, truth = {}):

        # Background parameters
        y = pyro.sample("y", dist.Normal(0., self.std))

        # Signal + background
        mu = (self.T * y.unsqueeze(0)).sum(-1) + self.w * z

        # Observation
        x = pyro.sample("x", dist.Normal(mu, self.sigma))

class ConLearnGuide(PyrofitGuide, nn.Module):
    def __init__(self, model, guide_conf):
        nn.Module.__init__(self)
        PyrofitGuide.__init__(self, model)
        self.N = 10000
        self.fc1 = nn.Linear(10000, 1)
        pyro.module("guide", self, update_module_params = True)

    def guide(self, observations={}, truth = {}):
        x = observations['model/x']

        z_loc = self.fc1(x)
        z_scale = torch.exp(pyro.param("guide_scale", torch.zeros(1)))
        print(z_scale)

        pyro.sample("model/z", dist.Normal(z_loc, z_scale))
