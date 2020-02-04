#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import torch
from torch.distributions import biject_to
import pyro
from pyro.contrib.easyguide import EasyGuide
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.distributions.util import eye_like
from .utils import load_param_store

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
        #print(match)
        group = self.group(match = match)
        z = []
        for site in group.prototype_sites:
            constrained_z = self.init(site)
            transform = biject_to(site['fn'].support)
            #print(site['infer'])
            z.append(transform.inv(constrained_z).reshape(-1))
        z_init = torch.cat(z, 0)
        return group, z_init

    def _get_orig_sampler(self, match = '.*'):
        """Return constrained initial values and prior sampler."""
        group = self.group(match = match)

        # Generate constrained
        z_con = []
        for site in group.prototype_sites:
            z_con_init = self.init(site)
            z_con.append(z_con_init.reshape(-1))
        z_con_init = torch.cat(z_con, 0)

        def sampler():
            z_con = []
            model_zs = {}
            for site in group.prototype_sites:
                val = pyro.sample(site['name'], site['fn'])
                model_zs[site['name']] = val
                z_con.append(val.reshape(-1))
            z_con = torch.cat(z_con, 0)
            return z_con, model_zs

        return sampler, z_con_init

class MAPGuide(PyrofitGuide):
    """Delta guide without variable concatenation."""
    def __init__(self, model):
        super(MAPGuide, self).__init__(model)
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
        return guide_z, model_zs

class DeltaGuide(PyrofitGuide):
    """Delta gudie with variable concatentation."""
    def __init__(self, model, guide_conf, prefix = None):
        super(DeltaGuide, self).__init__(model)
        self.mygroup = None
        self.guide_conf = guide_conf
        self.prefix = "" if prefix is None else prefix+"/"

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group(self.guide_conf['match'])
        z_loc = pyro.param(self.prefix+"guide_z_loc", self.z_init_loc)
        guide_z, model_zs = self.mygroup.sample(self.prefix+'guide_z',
                dist.Delta(z_loc).to_event(1))
        return guide_z, model_zs

class ProfileLikelihood(PyrofitGuide):
    """Profile likelihood guide."""
    def __init__(self, model, guide_conf, prefix = None):
        super(ProfileLikelihood, self).__init__(model)
        self.mygroup0 = None
        self.mygroup1 = None
        self.guide_conf = guide_conf
        self.prefix = "" if prefix is None else prefix+"/"

        if self.guide_conf['mode'] == 'linear': self.func = self._linear
        if self.guide_conf['mode'] == 'grid': self.func = self._grid
        if self.guide_conf['mode'] == 'NN': self.func = self._NN

    def _NN(self, z0):
        layers = self.guide_conf['layers']
        b0 = pyro.param(self.prefix+"guide_b0", torch.randn(layers[0]))
        A0 = pyro.param(self.prefix+"guide_A0", 0.1*torch.randn((layers[0], self.shape0[0])))
        b1 = pyro.param(self.prefix+"guide_b1", torch.randn(layers[1]))
        A1 = pyro.param(self.prefix+"guide_A1", 0.1*torch.randn((layers[1], layers[0])))
        b2 = pyro.param(self.prefix+"guide_b2", torch.randn(self.shape1[0]))
        A2 = pyro.param(self.prefix+"guide_A2", 0.1*torch.randn((self.shape1[0], layers[1])))

        A0d = pyro.param(self.prefix+"guide_A0d", 0.1*torch.randn((layers[0], self.shape0[0])))
        A1d = pyro.param(self.prefix+"guide_A1d", 0.1*torch.randn((layers[1], self.shape0[0])))
        A2d = pyro.param(self.prefix+"guide_A2d", 0.1*torch.randn((self.shape1[0], self.shape0[0])))

        z0 = z0-1.1
        z01 = torch.relu(b0 + (A0*z0).sum(1))  + (A0d*z0).sum(1)
        z02 = torch.relu(b1 + (A1*z01).sum(1)) + (A1d*z0).sum(1)
        z1  = b2 + (A2*z02).sum(1) + (A2d*z0).sum(1)
        return z1

    def _grid(self, z0):
        N = 10
        sigma = 0.5
        pos0 = torch.tensor([1.0, -2.0])
        pos = pyro.param(self.prefix+"guide_pos", torch.randn((N, self.shape0[0]))+pos0)
        pos = pos.detach()
        val = pyro.param(self.prefix+"guide_val", 1*torch.randn((N, self.shape1[0])))
        b0 = pyro.param(self.prefix+"guide_b0", torch.randn((self.shape1[0],)))
        w = torch.exp(-((z0-pos)**2).sum(1)/2/sigma**2)
        b = (w.unsqueeze(1)*val).sum(0) + 0*b0
        return b

    def _linear(self, z0):
        b = pyro.param(self.prefix+"guide_b", 0.1*torch.randn(self.shape1))
        #b0 = pyro.param(self.prefix+"guide_b0", 0.1*torch.randn(self.shape0))
        A = pyro.param(self.prefix+"guide_A", 0.1*torch.randn(self.shape1 + self.shape0))
        z0[1] = z0[1] + 2
        z0[0] = z0[0] - 1.0
        z1 = b + (A*(z0)).sum(1)
        return z1

    def guide(self):
        if self.mygroup0 is None:
            self.mysampler0, self.z0_init_loc = self._get_orig_sampler(self.guide_conf['match_master'])
            self.mygroup1,   self.z1_init_loc = self._get_group(self.guide_conf['match_slave'])
            self.shape0 = self.z0_init_loc.shape
            self.shape1 = self.z1_init_loc.shape

        guide_z0, model_zs0 = self.mysampler0()

        z1_loc = self.func(guide_z0)
        guide_z1, model_zs1 = self.mygroup1.sample(self.prefix+'guide_z_slave', dist.Delta(z1_loc).to_event(1))

        # Combine model predictions
        model_zs = model_zs0
        model_zs.update(model_zs1)

        return None, model_zs

class DiagonalNormalGuide(PyrofitGuide):
    def __init__(self, model, guide_conf, prefix = None):
        super(DiagonalNormalGuide, self).__init__(model)
        self.mygroup = None
        self.guide_conf = guide_conf
        self.prefix = "" if prefix is None else prefix+"/"

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group(self.guide_conf['match'])
            self.z_init_scale = (self.z_init_loc**2)**0.5*0.01 + 0.01
        z_loc = pyro.param(self.prefix+"guide_z_loc", self.z_init_loc)
        z_scale = pyro.param(self.prefix+"guide_z_scale", self.z_init_scale, constraint = constraints.positive)
        guide_z, model_zs = self.mygroup.sample(self.prefix+'guide_z',
                dist.Normal(z_loc, z_scale).to_event(1))
        return guide_z, model_zs

class MultivariateNormalGuide(PyrofitGuide):
    def __init__(self, model, guide_conf, prefix = None):
        super(MultivariateNormalGuide, self).__init__(model)
        self.mygroup = None
        self.guide_conf = guide_conf
        self.prefix = "" if prefix is None else prefix+"/"

    def guide(self):
        if self.mygroup is None:
            self.mygroup, self.z_init_loc = self._get_group(match = self.guide_conf['match'])
        z_loc = pyro.param(self.prefix+"guide_z_loc", self.z_init_loc)
        z_scale_tril = pyro.param(self.prefix+"guide_z_scale_tril", 0.01*eye_like(z_loc, len(z_loc)),
                                constraint=constraints.lower_cholesky)
        # TODO: Flexible initial error
        guide_z, model_zs = self.mygroup.sample(self.prefix+'guide_z',
                dist.MultivariateNormal(z_loc, scale_tril = z_scale_tril))
        return guide_z, model_zs

#class LinearBias:
#    def __init__(self, N, M):
#        self.A_init = torch.zeros(N, M)
#
#    def __call__(self, x):
#        A = pyro.param(self.prefix+"bias", self.A_init)
#        return A.dot(x)

class SuperGuide(PyrofitGuide):
    def __init__(self, model, guide_conf):
        super(SuperGuide, self).__init__(model)
        self.guides = {}
#        self.biases = {}
        for key, entry in guide_conf['groups'].items():
            guide = GUIDE_MAP[entry['type']](model, entry, prefix = key)
            self.guides[key] = guide
#            if 'biased_by' in entry.keys():
#                for bias in entry['biased_by']:
#                    bias['group']
#                    if bias['bias'] == 'linear':
#                        bias_fn = LinearBias(
#                    self.biases[key]

    def guide(self):
        model_zs_con = {}
        guide_z_dict = {}
        for key, guide in self.guides.items():
#            if key in self.biases:
#                bias = self.biases[key]['bias_fn'](guide_z_dict)
#            else:
#                bias = None
            guide_z, model_zs = guide()
            model_zs_con.update(model_zs)
            guide_z_dict[key] = guide_z
        return None, model_zs_con

def get_custom_guide(cond_model, guide_conf):
    module_name = guide_conf['module']
    my_module = importlib.import_module("pyrofit."+module_name)
    name = guide_conf['name']
    guide = getattr(my_module, name)

    return guide


GUIDE_MAP = {
        "Delta": DeltaGuide,
        "MAP": MAPGuide,
        "DiagonalNormal": DiagonalNormalGuide,
        "MultivariateNormal": MultivariateNormalGuide,
        #"Custom": get_custom_guide,
        "ProfileLikelihood": ProfileLikelihood,
        "SuperGuide": SuperGuide
        }

def init_guide(cond_model, guide_conf, guidefile = None, device = 'cpu'):
    guidetype = guide_conf['type']
    if guidefile is not None:
        load_param_store(guidefile, device = device)
    try:
        proto_guide = GUIDE_MAP[guidetype]
    except KeyError:
        raise KeyError("Guide type unknown")
    guide = proto_guide(cond_model, guide_conf)

    # We hide the first argument (unconstrained parameters) from the main code
    return lambda *args, **kwargs: guide(*args, **kwargs)[1]


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
