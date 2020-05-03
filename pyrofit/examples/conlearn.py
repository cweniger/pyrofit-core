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
        self.grid= torch.linspace(-5, 5, 20)
        self.device = device

    def __call__(self, offset:Yaml, slope:Yaml, observations = {}, truth = {}):
        sigma = 1.0
        pyro.sample("x", dist.Normal(offset+ slope*self.grid, sigma))


class ConLearnGuide(PyrofitGuide, nn.Module):
    def __init__(self, model, guide_conf):
        nn.Module.__init__(self)
        PyrofitGuide.__init__(self, model)
        self.fc1 = nn.Linear(20, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def guide(self, observations={}, truth = {}):
        pyro.module("guide", self)
        x = observations['model/x']
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        z_scale = pyro.param("guide_z_scale", torch.ones(2), constraint = constraints.positive)

        offset = pyro.sample("model/offset", dist.Normal(x[0], z_scale[0]))
        slope = pyro.sample("model/slope", dist.Normal(x[1], z_scale[1]))

        samples = {"model/offset": offset, "model/slope": slope}

        return None, samples
