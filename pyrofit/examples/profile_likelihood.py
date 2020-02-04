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
