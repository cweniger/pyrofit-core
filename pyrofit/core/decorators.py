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
    try:
        return CLS_LIST[name]
    except KeyError:
        raise KeyError("class %s unknown"%name)

####################################################
# Dummy type definitions for type annotation parsing
####################################################

Yaml = TypeVar("Yaml")


####################################
# Read in and parse config yaml file
####################################

def split_name(name):
    if "[" in name and "]" in name:
        i, j = name.find("["), name.find("]")
        cls = name[i+1:j]
        name = name[:i]
        return name, cls
    else:
        return name, None

YAML_CONFIG = None
SETTINGS = defaultdict(lambda: {})
VARIABLES = defaultdict(lambda: {})
CLASSES = {}#defaultdict(lambda: None)
def load_yaml(yamlfile):
    global YAML_CONFIG
    with open(yamlfile, "r") as stream:
        YAML_CONFIG = yaml.load(stream)
    for key, entry in YAML_CONFIG.items():
        # Ignore reserved keyword entries
        if key in ['pyrofit', 'conditioning'] or entry is None:
            continue
        # If instance...
        name, cls = split_name(key)
        if cls is None:
            VARIABLES[name] = yaml2actions(name, entry)
        else:
            if 'variables' in entry.keys() and entry['variables'] is not None:
                VARIABLES[name] = yaml2actions(name, entry['variables'])
            if 'settings' in entry.keys() and entry['settings'] is not None:
                SETTINGS[name] = yaml2settings(entry['settings'])
            CLASSES[name] = cls
    return YAML_CONFIG


#########################
# Registration decorators
#########################

def register(obj):
    """Register function or class for pyrofit use.

    This decorator has two effects:
    - Replace arguments annotated with `Yaml` by vales specified in YAML file.
    - Prepend `name/` to all sampling site names, where `name` denotes either
      the function or the class instance.
    """
    if inspect.isclass(obj):
        return _reg_cls(obj)
    else:
        return _reg_fn(obj)

def _parse_signature(fn):
    sig = inspect.signature(fn)
    params = [key for key in sig.parameters if sig.parameters[key].annotation != Yaml]
    yaml = [key for key in sig.parameters if sig.parameters[key].annotation == Yaml]
    params.sort()
    yaml.sort()
    return {'params':params, 'yaml': yaml}

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
        assert sorted(list(VARIABLES[name])) == sig['yaml'], """
        '%s': yaml variables %s expected, but %s given"""%(
                name, str(sig['yaml_var']), str(VARIABLES[name]))

        updates = {key: val() for key, val in VARIABLES[name].items()}

        # Update kwargs and run wrapped function
        kwargs.update(updates)
        return fn(**kwargs)

    return wrapped_fn

def _reg_cls(cls):
    # Get class name
    name = cls.__qualname__

    # Inspect __init__ function signature
    sig_init = _parse_signature(cls.__init__)
    sig_call = _parse_signature(cls.__call__)

    class Wrapped(cls):
        def __init__(self, name, **kwargs):
            self._pyrofit_instance_name = name
            if self._pyrofit_instance_name is None:
                cls.__init__(self, **kwargs)
            updates_set = {key: val for key, val in SETTINGS[name].items()}
            kwargs.update(updates_set)
            return scope(cls.__init__, prefix = self._pyrofit_instance_name)(self, **kwargs)

        def __call__(self, **kwargs):
            if self._pyrofit_instance_name is None:
                cls.__call__(self, **kwargs)
            updates_var = {key: val() for key, val in 
                    VARIABLES[self._pyrofit_instance_name].items()}
            kwargs.update(updates_var)
            return scope(cls.__call__, prefix = self._pyrofit_instance_name)(self, **kwargs)

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
            name, cls_name = split_name(name)
            #cls_name = YAML_CONFIG[name]['class']
            result[name] = get_wrapped_class(cls_name)(name, **kwargs)
    return result
