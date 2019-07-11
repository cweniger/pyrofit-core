# Import YAML file parameter sections as autoname named objects (nobs)
#
# Author: Christoph Weniger <c.weniger@uva.nl>
# Date: July 2019

import numpy as np
import torch
from pyro.contrib.autoname import named
# These may be needed by the eval statements below
import pyro.distributions as dist
#from .utils import LogUniform
#from .utils import TruncatedNormal

INIT_VALUES = {}
FIX_ALL = False

def set_fix_all(flag):
    """All parameters with specified `init` value are fixed."""
    global FIX_ALL 
    FIX_ALL = flag 

def get_init_values():
    return INIT_VALUES.copy()


######################
# Auxilliary functions
######################

def _parse_val(val, device='cpu', dtype = torch.float32):
    """Parse input value.  Note special treatment for augmented strings.
    """
    if isinstance(val, str):
        if val[:6] == "$EVAL ":
            tmp = eval(val)
        elif val[:5] == "$NPY ":
            tmp = np.load(val)
        elif val[:5] == "$CSV ":
            tmp = np.genfromtxt(val)
        else:
            return val
    else:
        tmp = val
    return torch.tensor(val, dtype=dtype, device=device)

def _parse_entry(key, val, nob, name, device):
    global INIT_VALUES

    # Parse non-dict values as fixed init parameter
    if not isinstance(val, dict):
        setattr(nob, key, _parse_val(val, device=device))
        # Store initial value in global INIT_VALUES
        return

    # Store all initial values in global INIT_VALUES
    if "init" in val.keys():
        INIT_VALUES[name + '.' + key] = _parse_val(
            val['init'], device=device)

    # Parse dict entries
    if (FIX_ALL and 'init' in val.keys()):
        setattr(nob, key, _parse_val(val['init'], device=device))
        return
    if "prior" in val.keys():
        fn = eval(val['prior'][0])
        args = [_parse_val(x, device=device) for x in val['prior'][1:]]
        getattr(nob, key).sample_(fn(*args))
        return
    if "init" in val.keys():
        getattr(nob, key).param_(
            _parse_val(val['init'], device=device))
        return
    else:
        raise KeyError("Either prior or init or both must be specified.")

def yaml2sampler(nob_name, yaml_parameters, device='cpu'):
    """Import YAML dictionary and pass it as `pyro.contrib.autoname.named`
    object as first argument to the decorated function.

    Example YAML entry to be parsed as `config`
    -------------------------------------------
    u: 1.                               # fixed tensor (requires_grad = False)
    w:                                  
      init: 1.                          # fixed tensor (requires_grad = False)
    x:
      prior: [dist.Normal, 0., 1.]      # sample from N(0, 1)
    y:
      init: 0.5                         # use 0.5 as initial value for (eg) HMC
      prior: [dist.Normal, 2., 1.5]     # sample from N(2, 1.5)

    Parameters
    ----------
    nob_name : `named` object name [str]
    yaml_parameters: associated YAML config section
    device: Generate all tensors on 'device' [str] (default 'cpu')

    Note
    ----
    Strings are supported and parsed as follows (depending on prefix token)
    - "$EVAL np.sin(3)" --> torch.tensor(np.sin(3))
    - "$NPY FILENAME" reads *.npy array
    - "$CSV FILENAME" reads CSV file
    - Everything else remains a string

    Returns
    -------

    Sampler [func]: Returns a sampler function that (re-)samples parameters,
    reads YAML file values, and returns a `named object' object with the
    results.
    """

    # FIXME: Right now, every sampler call will re-parse the YAML file,
    # including reading *.npy files from disk.  Needs speed-up.
    def sampler():
        nob = named.Object(nob_name)
        for key, val in yaml_parameters.items():
            _parse_entry(key, val, nob, nob_name, device)
        return nob

    return sampler
