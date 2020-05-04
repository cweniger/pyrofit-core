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
        self.grid= torch.linspace(-5, 5, 50)
        self.device = device

    def __call__(self, offset:Yaml, slope:Yaml, observations = {}, truth = {}):
        sigma = 0.1

        # FIXME: This is an annoying hack to avoid propagating gradients from
        # p(x|z) back to the sleep phase parameters while wake.  Needs to be
        # solved differently.
        slope = slope.detach_()

        x = pyro.sample("x", dist.Normal(offset + slope*self.grid, sigma))


class ConLearnGuide(PyrofitGuide, nn.Module):
    def __init__(self, model, guide_conf):
        nn.Module.__init__(self)
        PyrofitGuide.__init__(self, model)
        self.fc1 = nn.Linear(50, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        pyro.module("guide", self, update_module_params = True)

    def guide(self, observations={}, truth = {}):
        x = observations['model/x']
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        z_scale = torch.exp(pyro.param("guide_z_scale", torch.zeros(2)))
        z_loc = pyro.param("guide_z_loc", torch.zeros(1))
        print(z_scale, z_loc[0], x[0])

        offset = pyro.sample("model/offset", dist.Normal(z_loc[0], z_scale[0]))
        slope = pyro.sample("model/slope", dist.Normal(x[0], z_scale[1]))

        samples = {"model/offset": offset, "model/slope": slope}

        return None, samples
