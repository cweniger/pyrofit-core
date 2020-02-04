# Import YAML file parameter sections as autoname named objects (nobs)
#
# Author: Christoph Weniger <c.weniger@uva.nl>
# Date: July 2019

import numpy as np
import torch
import pyro
from pyro.contrib.autoname import named
# These may be needed by the eval statements below
import pyro.distributions as dist
from . import distributions as newdist
#from .utils import LogUniform
#from .utils import TruncatedNormal
from torch.distributions import constraints

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

def _parse_val(key, val, device='cpu', dtype = torch.float32):
    """Parse input value.  Note special treatment for augmented strings.
    """
    if val is None:
        return None
    try:
        if isinstance(val, str):
            if val[:6] == "$EVAL ":
                tmp = eval(val[6:])
            elif val[:5] == "$NPY ":
                tmp = np.load(val[5:])
            elif val[:5] == "$NPZ ":
                i, j = val.find("["), val.find("]")
                if i > -1:  # If "[...]" given, use that as tag
                    name = val[i+1:j]
                    tmp = np.load(val[5:i])[name]
                else:
                    tmp = np.load(val[5:])[key]
            elif val[:5] == "$CSV ":
                tmp = np.genfromtxt(val)
            else:
                return val
        else:
            tmp = val
        if isinstance(tmp, torch.Tensor):
            return tmp.to(device)
        else:
            return torch.tensor(tmp, dtype=dtype, device=device)
    except ValueError:
        raise ValueError("Could not parse %s"%str(val))

def _entry2action(key, val, module, device):
    global INIT_VALUES
    global FIX_ALL

    # Parse non-dict values as fixed values
    if not isinstance(val, dict):
        val = _parse_val(key, val, device=device)
        return lambda param: val
    keys = list(val.keys())
    keys.sort()
    if keys == ['sample']:
        try:
            fn = eval(val['sample'][0])
        except:
            try:
                fn = eval(f"module.{val['sample'][0]}")
            except:
                raise ValueError(f"Could not parse distribution {val['sample'][0]}")

        args = [_parse_val(key, x, device=device) for x in val['sample'][1:]]


        def sampler(param):
            val = pyro.sample(param, fn(*args))
            # print("Sampled", param, "with val", val)
            return val

        return sampler
        # return lambda param: pyro.sample(param, fn(*args))
    if keys == ['init', 'sample']:
        if "init" in val.keys():
            # TODO: is this ever used anywhere??
            infer = {'init': _parse_val(key, val['init'], device=device)}
            batch_shape = infer["init"].shape
        else:
            infer = None
            batch_shape = torch.Size([])

        try:
            fn = eval(val['sample'][0])
        except:
            try:
                fn = eval(f"module.{val['sample'][0]}")
            except:
                raise ValueError(f"Could not parse distribution {val['sample'][0]}")

        args = [_parse_val(key, x, device=device) for x in val['sample'][1:]]

        def sampler(param):
            obs = None if not FIX_ALL else infer["init"]
            val = pyro.sample(param, fn(*args).expand(batch_shape), infer=infer, obs=obs)
            # print("Sampled (with 'init')", param, "with val", val)
            return val

        return sampler
        # return lambda param: pyro.sample(param, fn(*args).expand(batch_shape), infer=infer)
    if keys == ['param']:
        arg = _parse_val(key, val['param'], device=device)
        return lambda param: pyro.param(param, arg)
    if keys == ['constraint', 'param']:
        arg = _parse_val(key, val['param'], device=device)
        return lambda param: pyro.param(param, arg, constraint = eval(val['constraint']))
    raise KeyError("Incompatible parameter section entries with keys %s"%str(keys))

def yaml2settings(yaml_params, module, device='cpu'):
    """Import YAML dictionary and pass it as `pyro.contrib.autoname.named`
    object as first argument to the decorated function.

    Example YAML entry to be parsed as `config`
    -------------------------------------------
    u: 1.                               # fixed tensor (requires_grad = False)
    w:
      init: 1.                          # fixed tensor (requires_grad = False)
    w:
      param: 1.                          # free tensor, without prior (requires_grad = True)
    x:
      sample: [dist.Normal, 0., 1.]      # sample from N(0, 1)
    y:
      init: 0.5                         # use 0.5 as initial value for (eg) HMC
      sample: [dist.Normal, 2., 1.5]     # sample from N(2, 1.5)

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
    settings = {}
    for key, val in yaml_params.items():
        settings[key] = _parse_val(key, val, device)
    return settings

def yaml2actions(name, yaml_params, module, device='cpu'):
    """Import YAML dictionary and pass it as `pyro.contrib.autoname.named`
    object as first argument to the decorated function.

    Example YAML entry to be parsed as `config`
    -------------------------------------------
    u: 1.                               # fixed tensor (requires_grad = False)
    w:
      init: 1.                          # fixed tensor (requires_grad = False)
    w:
      param: 1.                          # free tensor, without prior (requires_grad = True)
    x:
      sample: [dist.Normal, 0., 1.]      # sample from N(0, 1)
    y:
      init: 0.5                         # use 0.5 as initial value for (eg) HMC
      sample: [dist.Normal, 2., 1.5]     # sample from N(2, 1.5)

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
    actions = {}
    for key, val in yaml_params.items():
        action = _entry2action(key, val, module, device)
        actions[key] = lambda action = action, par_name = name+"/"+key: action(par_name)

    return actions
