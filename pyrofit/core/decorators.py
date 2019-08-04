import inspect
from yaml_parser2 import yaml2actions, yaml2settings

YAML_CONFIG = None

def reg(obj):
    if inspect.isclass(obj):
        return _reg_cls(obj)
    else:
        return _reg_fn(obj)

def _reg_fn(fn):
    name = fn.__qualname__
    try:
        params = YAML_CONFIG[name]['params']
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
                params = YAML_CONFIG[name]['params']
            except KeyError:
                self.pyrofit_actions = {}
            else:
                self.pyrofit_actions = yaml2actions(name, YAML_CONFIG[name]['params'])
            try:
                yaml_settings = YAML_CONFIG[name]['settings']
            except KeyError:
                settings = {}
            else:
                settings = yaml2settings(YAML_CONFIG[name]['settings'])
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
    names = YAML_CONFIG.keys()
    result = {}
    for name in names:
        if name.startswith(start):
            cls_name = YAML_CONFIG[name]['class']
            result[name] = CLS_LIST[cls_name](name, **kwargs)
    return result
