import pyro
import torch
import pyro.distributions as dist
from pyrofit.core import *


##################################################
# Simple linear regression via registered function
##################################################

@register
def linear(a:Yaml, b:Yaml, x:Yaml):
    pyro.sample("y", dist.Normal(a + b*x, 1.0))


#######################################
# Simple quadratic regression via class
#######################################

@register
class Quadratic:
    def __init__(self, x:Yaml):
        self.x = x

    def __call__(self, a:Yaml, b:Yaml, c:Yaml):
        pyro.sample("y", dist.Normal(a + b*self.x + c*self.x**2, 1.0))


########################################################################
# More complex model with several "Source" instances + linear regression
########################################################################

@register
class Source:
    def __init__(self, xgrid):
        self.xgrid = xgrid

    def __call__(self, h0:Yaml, x0:Yaml, w0:Yaml):
        """Gaussian source with free height, width and position."""
        return h0*torch.exp(-0.5*(self.xgrid-x0)**2/w0**2)

@register
class SpecModel:
    def __init__(self, xgrid:Yaml):
        self.xgrid = xgrid
        self.sources = instantiate("source", xgrid = xgrid)  # Assumed to be compatible with 'Source' class

    def __call__(self, a:Yaml, b:Yaml):
        spec = sum([source() for _, source in self.sources.items()])
        spec += self.xgrid*b+a
        pyro.sample("spec", dist.Normal(spec, 1.0))
