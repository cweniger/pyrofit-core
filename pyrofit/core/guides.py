#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.distributions import biject_to
import pyro
from pyro.contrib.easyguide import EasyGuide
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.distributions.util import eye_like

class PyrofitGuide(EasyGuide):
    def __init__(self, model):
        super(PyrofitGuide, self).__init__(model)

    def init(self, site):
        """Return constrained mean or explicit init value."""
        init = site['infer'].get('init', 'sample')
        if not isinstance(init, str):
            return init
        if init  == 'sample':
            return site['fn']()
        elif init == 'mean':
            N = 1000
            return sum([site['fn']() for i in range(N)])/N
        else:
            raise KeyError

    def _get_group(self, match = '.*'):
        """Return group and unconstrained initial values."""
        group = self.group(match = match)
        z = []
        for site in group.prototype_sites:
            constrained_z = self.init(site)
            transform = biject_to(site['fn'].support)
            print(site['infer'])
            z.append(transform.inv(constrained_z).reshape(-1))
        z_init = torch.cat(z, 0)
        return group, z_init

class DeltaGuide(PyrofitGuide):
    def __init__(self, model):
        super(DeltaGuide, self).__init__(model)
        self.mygroup = None

    def guide(self):
        # Initialize guide if necessary
        if self.mygroup is None:
            # Dummy group formation to collect all site names
            self.mygroup = self.group()
            # This initializes the values of the MAP estimator
            for site in self.mygroup.prototype_sites:
                site['value'] = self.init(site)

        model_zs = {}
        for site in self.mygroup.prototype_sites:
            model_zs[site['name']] = self.map_estimate(site['name'])
        return model_zs

class DiagonalNormalGuide(PyrofitGuide):
    def __init__(self, model):
        super(DiagonalNormalGuide, self).__init__(model)
        self.mygroup = None

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group()
            self.z_init_scale = (self.z_init_loc**2)**0.5*0.01 + 0.01
        z_loc = pyro.param("guide_z_loc", self.z_init_loc)
        z_scale = pyro.param("guide_z_scale", self.z_init_scale, constraint = constraints.positive)
        guide_z, model_zs = self.mygroup.sample('guide_z',
                dist.Normal(z_loc, z_scale).to_event(1))
        return model_zs

class MultivariateNormalGuide(PyrofitGuide):
    def __init__(self, model):
        super(MultivariateNormalGuide, self).__init__(model)
        self.mygroup = None

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group()
        z_loc = pyro.param("guide_z_loc", self.z_init_loc)
        z_scale_tril = pyro.param("guide_z_scale_tril", 0.01*eye_like(z_loc, len(z_loc)),
                                constraint=constraints.lower_cholesky)
        # TODO: More flexible initial error
        guide_z, model_zs = self.mygroup.sample('guide_z',
                dist.MultivariateNormal(z_loc, scale_tril = z_scale_tril))
        return model_zs

class SuperGuide(PyrofitGuide):
    def __init__(self, model):
        super(SuperGuide, self).__init__(model)
        self.mygroup = None

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group()
        z_loc = pyro.param("guide_z_loc", self.z_init_loc)
        z_scale_tril = pyro.param("guide_z_scale_tril", 0.01*eye_like(z_loc, len(z_loc)),
                                constraint=constraints.lower_cholesky)
        # TODO: More flexible initial error
        guide_z, model_zs = self.mygroup.sample('guide_z',
                dist.MultivariateNormal(z_loc, scale_tril = z_scale_tril))
        return model_zs


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
