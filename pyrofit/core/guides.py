#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.distributions import biject_to
import pyro
from pyro.contrib.easyguide import EasyGuide
import pyro.distributions as dist

class DeltaGuide(EasyGuide):
    def __init__(self, model):
        super(DeltaGuide, self).__init__(model)
        self.mygroup = None

    def init(self, site):
        """Return constrained mean or explicit init value."""
        if 'init' in site['infer']:
            return site['infer']['init']
        else:
            N = 1000
            return sum([site['fn']() for i in range(N)])/N

    def _get_group(self, match = '.*'):
        """Return group and unconstrained initial values."""
        group = self.group(match = match)
        z = []
        for site in group.prototype_sites:
            constrained_z = self.init(site)
            transform = biject_to(site['fn'].support)
            z.append(transform.inv(constrained_z).reshape(-1))
        z_init = torch.cat(z, 0)
        return group, z_init

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init = self._get_group()
        auto_z = pyro.param("guide_z_map", self.z_init)
        guide_z, model_zs = self.mygroup.sample('guide_z',
                dist.Delta(auto_z).expand((1,)).to_event(1))
        return model_zs, guide_z


#######
# Tests
#######

def model():
    x = pyro.sample("x", dist.Uniform(0., 1.), infer = {'init': torch.tensor(0.3)})
    #x = pyro.sample("x", dist.Uniform(1000., 1001.))
    return x

if __name__ == '__main__':
    guide = DeltaGuide(model)
    print(guide())
