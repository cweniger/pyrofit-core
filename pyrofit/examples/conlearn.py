import pyro
import torch
import pyro.distributions as dist
from pyrofit.core import *
from pyrofit.core.guides import PyrofitGuide

from torch.distributions import constraints

@register
class Model:
    """A linear function."""
    def __init__(self, device = 'cpu'):
        self.grid= torch.linspace(-5, 5, 20)
        self.device = device

    def __call__(self, offset:Yaml, slope:Yaml):
        sigma = 1.0
        pyro.sample("x", dist.Normal(offset+ slope*self.grid, sigma))


class ConLearnGuide(PyrofitGuide):
    def __init__(self, model, guide_conf):
        super(ConLearnGuide, self).__init__(model)
        self.mygroup = None

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group(match = ".*")
        z_loc = pyro.param("guide_z_loc", self.z_init_loc)
        z_scale = pyro.param("guide_z_scale", torch.ones_like(self.z_init_loc), constraint = constraints.positive)

        guide_z, model_zs = self.mygroup.sample('guide_z', dist.Normal(z_loc, z_scale).to_event(1))
        return guide_z, model_zs
