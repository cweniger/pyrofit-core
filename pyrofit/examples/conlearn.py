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
        with self.sleep_grad():
            x = observations['model/x']
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            slope_loc = self.fc3(x)[0]
            ppp = pyro.param("guide_slope_scale", torch.zeros(1))
            slope_scale = torch.exp(ppp[0])

        with self.wake_grad():
            offset_scale = 1.*torch.exp(pyro.param("guide_z_scale", torch.zeros(1)))[0]
            offset_loc = 1.*pyro.param("guide_z_loc", torch.zeros(1))[0]

        print(offset_loc, offset_scale, slope_loc, slope_scale)

        offset = pyro.sample("model/offset", dist.Normal(offset_loc, offset_scale))
        slope = pyro.sample("model/slope", dist.Normal(slope_loc, slope_scale))
