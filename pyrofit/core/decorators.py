#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import yaml_parser
from inspect import signature
import inspect
#from typing import Int

#from ruamel.yaml import YAML
#yaml = YAML()
import yaml
import pyro
import pyro.distributions as dist
from yaml_parser import yaml2actions, yaml2settings

#with open(yamlfile, "r") as stream:
#    yaml_config = yaml.load(stream)

yaml_config = yaml.load("""
test:
  z:
    sample: [dist.Normal, 0., 3.]
alpha.cool:
  class: Model
  settings:
    alpha: 3
  params:
    z:
        sample: [dist.Normal, 0., 3.]
test2:
  settings:
    alpha: 3
""")
CLS_LIST = {}


#MODEL_LIST = defaultdict(lambda: {})
#COMPS = {}
#DEVICE = 'cuda:0'
#
#def model(fn):
#    MODEL_LIST[fn.__qualname__]["fn"] = fn
#    return fn
#
#def params(*params):
#    def wrapper(fn):
#        MODEL_LIST[fn.__qualname__]["params"] = params
#        return fn
#    return wrapper
#
#def settings(*settings):
#    def wrapper(fn):
#        MODEL_LIST[fn.__qualname__]["settings"] = settings
#        return fn
#    return wrapper
#
#def context(name, regex = None):
#    if regex == None:
#        regex = "name"
#    comps = None  # get components
#    def wrapper(f):
#        return lambda *args, **kwargs: f(comps)
#    return wrapper
#
#def initialize_fn(name, cls, kwargs):
#    # Get model function and expected parameters
#    fn = MODEL_LIST[cls]['fn']
#    params = MODEL_LIST[cls]['params']
#
#    # Get sampler function from yaml file
#    actions = yaml_parser.yaml2actions(name, yaml_config[name]['params'])
#
#    # Get settings from yaml file
#    settings = yaml_parser.yaml2settings(yaml_config[name]['settings'])
#
#    intersec = set(actions.keys()).intersection(settings.keys())
#    if len(intersec) > 0:
#        raise KeyError("Multiple parameter definitions for component '%s': %s.  Check model/YAML file."%(name, str(tuple(intersec))))
#
#    # Construct actionable parameter list
#    argactionlist = ()
#    for param in params:
#        if param in kwargs.keys() and param in actions.keys():
#            raise KeyError("Parameter '%s' of component '%s' doubly defined. Check model/YAML file."%(param, name))
#        elif param in kwargs.keys():
#            argactionlist = argactionlist + (lambda kwargs, param = param: kwargs[param],)
#        elif param in actions.keys():
#            val = lambda param = param: actions[param]()
#            argactionlist = argactionlist + (lambda kwargs, val = val: val(),)
#        else:
#            raise KeyError("Parameter '%s' of component '%s' is unspecified."%(param, name))
#    for key in kwargs.keys():
#        if key not in params:
#            raise KeyError("Parameter '%s' not required for component '%s'. Check model."%(key, name))
#    for key in actions.keys():
#        if key not in params:
#            raise KeyError("Parameter '%s' not required for component '%s'. Check YAML file."%(key, name))
#
#    # Define initialized function with reference to parameter list
#    def initialized_fn(**kwargs):
#        args = (x(kwargs) for x in argactionlist)
#        return fn(*args)
#
#    # Initialize by overwriting component overwriting component function
#    COMPS[name] = initialized_fn
#
#    # Call function first time
#    return initialized_fn(**kwargs)
#
#"""
#alphas.StrongLensing[lensing.NFWhalo]:
#    params:
#    settings:
#f:
#    x: 3.
#    y: 2
#"""
#  
#@model
#class Model:
#    def __init__(self, this, that, whatnot):
#        cadsc
#
#    def __call__(self, x, y, z):
#        calculation
#
#@model
#def simple_func(this, that, what_not):
#    none
#
#@model
#def sun_distance(r):
#    return r
#
#@model
#def main():
#    f(y = torch.tensor(3))
#
#def configure(yaml):
#    for name in yaml:
#        cls = yaml[name]['class']
#        COMPS[name] = lambda **kwargs: initialize_fn(name, cls, kwargs)
#
#@model
#@params("x", "y", "z")
#def f(x, y, z):
#    print(x, y, z)
#    return x
#
#@model
#@params("x", "y")
#@settings("X")
#def main(x, y):
#    for _, fn in alphas.items():
#        alpha = fn(X = X, Y = Y)
#        res = f(x = 4.)
#
#var = 3
#setting = 3
#
#def context(a):
#    return 1
#

#####################


def reg(obj):
    if inspect.isclass(obj):
        return _reg_cls(obj)
    else:
        return _reg_fn(obj)

def _reg_fn(fn):
    name = fn.__qualname__
    try:
        params = yaml_config[name]['params']
    except KeyError:
        # Just return function if not mentioned in YAML file
        return fn
    actions = yaml2actions(name, params)
    def wrapped_fn(**kwargs):
        updates = {key: val() for key, val in actions.items()}
        kwargs.update(updates)
        return fn(**kwargs)
    return wrapped_fn

def _reg_cls(cls):
    name = cls.__qualname__
    class Wrapped(cls):
        def __init__(self, name, **kwargs):
            try:
                params = yaml_config[name]['params']
            except KeyError:
                self.pyrofit_actions = {}
            else:
                self.pyrofit_actions = yaml2actions(name, yaml_config[name]['params'])
            try:
                yaml_settings = yaml_config[name]['settings']
            except KeyError:
                settings = {}
            else:
                settings = yaml2settings(yaml_config[name]['settings'])
            # TODO -- check overwriting
            kwargs.update(settings)
            cls.__init__(self, **kwargs)

        def __call__(self, **kwargs):
            updates = {key: val() for key, val in self.pyrofit_actions.items()}
            kwargs.update(updates)
            cls.__call__(self, **kwargs)
    CLS_LIST[name] = Wrapped
    return Wrapped

def instantiate(start, **kwargs):
    names = yaml_config.keys()
    result = {}
    for name in names:
        if name.startswith(start):
            cls_name = yaml_config[name]['class']
            result[name] = CLS_LIST[cls_name](name, **kwargs)
    return result

#####################

@reg
def addition(x, y, z):
    pyro.sample("image", dist.Normal(x+y+z, 0.1))
    return {"image": image}

@reg
def test(x, y, z):
    print("fn (test):", x, y, z)
    return y

@reg
class Model:
    def __init__(self, alpha):
        print("init", alpha)

    def __call__(self, z):
        print("call", z)

def main():
    m = Model("test2")
    res = instantiate("alpha")
    res['alpha.cool']()

if __name__ == "__main__":
    main()

