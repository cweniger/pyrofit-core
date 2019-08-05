from typing import TypeVar
from collections import defaultdict
import inspect
import yaml
from pyro.contrib.autoname import scope
from .yaml_params2 import yaml2actions, yaml2settings

#############################################################
# Global wrapped class register (used for auto-instantiation)
#############################################################

CLS_LIST = {}
def get_wrapped_class(name):
    return CLS_LIST[name]

####################################################
# Dummy type definitions for type annotation parsing
####################################################

YamlSet = TypeVar("YamlSet")
YamlVar = TypeVar("YamlVar")



####################################
# Read in and parse config yaml file
####################################

YAML_CONFIG = None
SETTINGS = defaultdict(lambda: {})
VARIABLES = defaultdict(lambda: {})
def load_yaml(yamlfile):
    global YAML_CONFIG
    with open(yamlfile, "r") as stream:
        YAML_CONFIG = yaml.load(stream)
    for name, entry in YAML_CONFIG.items():
        if name in ['pyrofit', 'conditioning'] or entry is None:
            continue
        if 'variables' in entry.keys() and entry['variables'] is not None:
            VARIABLES[name] = yaml2actions(name, entry['variables'])
        if 'settings' in entry.keys() and entry['settings'] is not None:
            SETTINGS[name] = yaml2settings(entry['settings'])
    return YAML_CONFIG


#########################
# Registration decorators
#########################

def register(obj):
    """Register function or class for pyrofit use.

    This decorator has two effects:
    - Replace arguments annotated with `YamlVar` or `YamlSet` by values
      specified in the yaml initialization file.
    - Prepend `name/` to all sampling site names, where `name` denotes either
      the function or the class instance.
    """
    if inspect.isclass(obj):
        return _reg_cls(obj)
    else:
        return _reg_fn(obj)

def _parse_signature(fn):
    sig = inspect.signature(fn)
    params = [key for key in sig.parameters
            if (sig.parameters[key].annotation != YamlVar)
            and (sig.parameters[key].annotation != YamlSet)]
    yaml_var = [key for key in sig.parameters
            if sig.parameters[key].annotation == YamlVar]
    yaml_set = [key for key in sig.parameters
            if sig.parameters[key].annotation == YamlSet]
    params.sort()
    yaml_var.sort()
    yaml_set.sort()
    return {'params':params, 'yaml_var': yaml_var, 'yaml_set': yaml_set}

def _reg_fn(fn):
    # Prefix sample sites
    name = fn.__qualname__
    fn = scope(fn, prefix = name)

    # Inspect function signature
    sig = _parse_signature(fn)

    def wrapped_fn(**kwargs):
        # Checking consistency of kwargs
        assert sorted(list(kwargs.keys())) == sig['params'], """
        '%s': keyword arguments %s expected, but %s given"""%(
                name, str(sig['params']), str(list(kwargs)))
        assert sorted(list(VARIABLES[name])) == sig['yaml_var'], """
        '%s': yaml variables %s expected, but %s given"""%(
                name, str(sig['yaml_var']), str(VARIABLES[name]))
        assert sorted(list(SETTINGS[name])) == sig['yaml_set'], """
        '%s': yaml settings %s expected, but %s given"""%(
                name, str(sig['yaml_set']), str(SETTINGS[name]))

        updates_set = {key: val for key, val in SETTINGS[name].items()}
        updates_var = {key: val() for key, val in VARIABLES[name].items()}

        # Update kwargs and run wrapped function
        kwargs.update(updates_var)
        kwargs.update(updates_set)
        return fn(**kwargs)

    return wrapped_fn

def _reg_cls(cls):
    # Get class name
    name = cls.__qualname__

    # Inspect __init__ function signature
    sig_init = _parse_signature(cls.__init__)
    assert len(sig_init['yaml_var']) == 0, """
    '%s': No YamlVar allowed in __init__ method"""%name
    sig_call = _parse_signature(cls.__call__)
    assert len(sig_call['yaml_set']) == 0, """
    '%s': No YamlSet allowed in __call__ method"""%name

    class Wrapped(cls):
        def __init__(self, name, **kwargs):
            self._pyrofit_instance_name = name
            if self._pyrofit_instance_name is None:
                cls.__init__(self, **kwargs)
            updates_set = {key: val for key, val in SETTINGS[name].items()}
            kwargs.update(updates_set)
            return scope(cls.__init__, prefix = name)(self, **kwargs)

        def __call__(self, **kwargs):
            if self._pyrofit_instance_name is None:
                cls.__call__(self, **kwargs)
            updates_var = {key: val() for key, val in 
                    VARIABLES[self._pyrofit_instance_name].items()}
            kwargs.update(updates_var)
            return scope(cls.__call__, prefix = name)(self, **kwargs)

    # Register wrapped class name
    CLS_LIST[name] = Wrapped

    return Wrapped

def instantiate(start, **kwargs):
    names = YAML_CONFIG.keys()
    result = {}
    for name in names:
        if name in ['pyrofit', 'conditioning']:
            continue
        if name.startswith(start):
            cls_name = YAML_CONFIG[name]['class']
            result[name] = get_wrapped_class(cls_name)(name, **kwargs)
    return result
