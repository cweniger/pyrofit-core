# pyrofit
#
# version: v0.1
#
# author: Christoph Weniger <c.weniger@uva.nl>
# date: Jan - July 2019

import click
import numpy as np
import pickle
import collections
import importlib
import torch
import pyro
from pyro import poutine
from torch.distributions import constraints
from pyro.contrib.util import hessian
from torch.distributions.transforms import AffineTransform, ComposeTransform
from torch.distributions import biject_to
from pyro.contrib.autoguide import AutoDelta, AutoLowRankMultivariateNormal, AutoLaplaceApproximation, AutoDiagonalNormal, AutoMultivariateNormal, init_to_sample
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, JitTrace_ELBO
from pyro.infer.mcmc import MCMC, NUTS, HMC, util
from pyro.optim import Adam, SGD
from ruamel.yaml import YAML
yaml = YAML()
from tqdm import tqdm

from . import yaml_params

######################
# Auxilliary functions
######################

def get_components(yaml_entries, type_mapping, device='cpu'):
    """Instantiates objects defined in the lens plane.

    These may be defined in terms of convergences ('kappas' section in yaml) or
    deflection fields ('alphas' section).

    Parameters
    ----------
    X, Y : torch.Tensor, torch.Tensor
        Lens plane meshgrids.
    entries : dict
        Config entries.
    type_mapping : dict(str -> (Object, dict))
        Mapping from 'type' field in config to the corresponding object and a
        dict containing any extra initializer arguments.
    device : str

    Returns
    -------
    instances : list
        Instances of the relevant classes.
    """
    if yaml_entries is None: return []

    instances = []
    for name, entry in yaml_entries.items():
        entry_type = entry["type"]
        parameters = entry.get("parameters", {})
        options = entry.get("options", {})

        # Parse option values
        for key, val in options.items():
            options[key] = yaml_params._parse_val(val, device=device)

        sampler = yaml_params.yaml2sampler(name, parameters, device=device)

        cls, args, kwargs = type_mapping[entry_type]
        kwargs.update(options)
        kwargs.update({"device": device})
        instance = cls(sampler, *args, **kwargs)

        instances.append(instance)
    return instances

def listify(val):
    "Turns array into nested lists. Required for YAML output."
    val = np.array(val)
    if len(val.shape) == 0:
        return float(val)
    elif len(val.shape) == 1:
        return list(val)
    elif len(val.shape) == 2:
        ret = []
        for i in range(val.shape[0]):
            ret.append(list(val[i]))
        return str(ret)
    else:
        raise TypeError("Cannot write >2-dim tensors to yaml file")

def get_conditioned_model(yaml_section, model, device='cpu'):
    if yaml_section is None:
        return model
    conditions = {}
    for name , val in yaml_section.items():
        conditions[name] = yaml_params._parse_val(val, device = device)
    return pyro.condition(model, conditions)


##############
# I/O Routines
##############

def update_yaml(yaml_section, key, val):
    """Updates `init` values in yaml file.

    We assume that `key' = 'nob_name.param_name' OR 'auto_nob_name.param_name'.
    """
    # Handle autoguide parameters
    # TODO: Check absence of model parameter interference
    print(key)
    if "auto_" == key[:5]:
        key = key[5:]
    nob_name, param_name = key.split(".")
    if nob_name in yaml_section.keys():
        entry = yaml_section[nob_name]["parameters"][param_name]
        if isinstance(entry, dict):
            entry["init"] = listify(val)
        else:
            entry = listify(val)

def write_yaml(config, outfile):
    """Dump updated YAML file."""
    ps = pyro.get_param_store()
    for key in ps.keys():
        val = ps[key].detach().cpu().numpy()
        for section in ['alphas', 'kappas', 'sources']:
            if config[section] is not None:
                update_yaml(config[section], key, val)
    with open(outfile, "w") as outfile:
        yaml.dump(config, outfile)
    print("Dumped current parameters into YAML file:", outfile)

def load_param_steps(args):
    with open(args["fileroot"][:-3] + "init_infer-data.pkl", "rb") as f:
        infer_data = pickle.load(f)
    return infer_data

def save_posteriors(args, posteriors):
    """Saves empirical posteriors (ie, MCMC samples and weights).
    """
    with open(args["fileroot"] + "_posteriors.pkl", "wb") as f:
        pickle.dump(posteriors, f, pickle.HIGHEST_PROTOCOL)

def load_posteriors(args):
    """Loads posterior values.
    """
    with open(args["fileroot"] + "_posteriors.pkl", "rb") as f:
        posteriors = pickle.load(f)
    return posteriors

def save_param_store(filename):
    """Saves the parameter store so optimization can be resumed later.
    """

def load_param_store(paramfile, device = 'cpu'):
    """Loads the parameter store from the resume file.
    """
    pyro.clear_param_store()
    try:
        print(device)
        pyro.get_param_store().load(paramfile, map_location = device)
        print("Loading param_store file:", paramfile)
    except FileNotFoundError:
        print("...no resume file not found. Starting from scratch.")

def load_true_param(config):
    with open(config["data"]["true_yaml"], "r") as stream:
        config_true = yaml.load(stream)
    
    alphas = config_true['alphas']
    kappas = config_true['kappas']
    
    true_params = {}
    
    if alphas is not None:
        for i in range(0,len(alphas)):
            for key in alphas[i]["parameters"].keys():
                true_params[alphas[i]["name"]+"."+key] = alphas[i]["parameters"][key]
    
    if kappas is not None:
        for i in range(0,len(kappas),0):
            for key in kappas[i]["parameters"].keys():
                true_params[kappas[i]["name"]+"."+key] = kappas[i]["parameters"][key]

    return true_params

def init_guide(cond_model, guidetype, guidefile = None, device = 'cpu'):
    if guidefile is not None:
        load_param_store(guidefile, device = device)
    if guidetype == 'Delta':
        guide = AutoDelta(cond_model, init_loc_fn = init_to_sample)
    elif guidetype == 'DiagonalNormal':
        guide = AutoDiagonalNormal(cond_model, init_loc_fn = init_to_sample, init_scale = 0.01)
    elif guidetype == 'MultivariateNormal':
        guide = AutoMultivariateNormal(cond_model, init_loc_fn = init_to_sample)
    elif guidetype == 'LowRankMultivariateNormal':
        guide = AutoLowRankMultivariateNormal(cond_model, init_loc_fn = init_to_sample)
    elif guidetype == 'LaplaceApproximation':
        guide = AutoLaplaceApproximation(cond_model, init_loc_fn = init_to_sample)
    else:
        raise KeyError("Guide type unknown")
    return guide

def save_guide(guidetype, guidefile):
    pyro.get_param_store().save(guidefile)

def initial_params_from_guide(guide):
    median = guide.median()
    for key in median:
        median[key] = median[key].detach()
    return median


#######################
# Inference
#######################

def trace_to_cpu(trace):
    """Puts all the values in a trace on the CPU."""
    for key in trace.nodes.keys():
        if key != "_INPUT" and key != "_RETURN":
            try:
                trace.nodes[key]["value"] = trace.nodes[key]["value"].cpu()
            except KeyError:
                pass
    return trace


def make_transformed_pe(potential_fn, transform, unpack_fn):
    def transformed_potential_fn(arg):
        # NB: currently, intermediates for ComposeTransform is None, so this has no effect
        # see https://github.com/pyro-ppl/numpyro/issues/242
        z = arg["z"]
        u = transform(z)
        logdet = transform.log_abs_det_jacobian(z, u).sum()
        d = {s['name']: b for s, b in unpack_fn(u)}
        return potential_fn(d) + logdet

    return transformed_potential_fn
    

def infer_NUTS(cond_model, n_steps, warmup_steps, n_chains = 1, device = 'cpu', guidefile = None, guidetype = None):
    """Runs the NUTS HMC algorithm.

    Saves the samples and weights as well as a netcdf file for the run.

    Parameters
    ----------
    args : dict
        Command line arguments.
    cond_model : callable
        Model conditioned on an observed images.
    """
    initial_params, potential_fn, transforms, prototype_trace = util.initialize_model(cond_model)

    if guidefile is not None:
        guide = init_guide(cond_model, guidetype, guidefile = guidefile, device = device)
        sample = guide()
        for key in initial_params.keys():
            initial_params[key] = transforms[key](sample[key].detach())

    # FIXME: In the case of DiagonalNormal, results have to be mapped back onto unpacked latents
    if guidetype == "DiagonalNormal":
        transform = guide.get_transform()
        unpack_fn = lambda u: guide.unpack_latent(u)
        potential_fn = make_transformed_pe(potential_fn, transform, unpack_fn)
        initial_params = {"z": torch.zeros(guide.get_posterior().shape())}
        transforms = None

    def fun(*args, **kwargs):
        res = potential_fn(*args, **kwargs)
        print(res)
        return res


    nuts_kernel = NUTS(
        potential_fn = fun,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        jit_compile=False,
        max_tree_depth = 10,
        transforms = transforms,
        step_size = 1.)
    nuts_kernel.initial_params = initial_params

    # Run
    posterior = MCMC(
        nuts_kernel, n_steps, warmup_steps=warmup_steps,
        num_chains=n_chains).run()


def infer_VI(cond_model, guidetype, guidefile, n_steps, n_write=10, device = 'cpu'):
    """Runs MAP parameter inference.

    Regularly saves the parameter and loss values. Also saves the pyro
    parameter store so runs can be resumed.

    Parameters
    ----------
    args : dict
        Command line arguments
    cond_model : callable
        Lensing system model conditioned on an observed image.
    n_write : int
        Number of iterations between parameter store and loss and parameter
        saves.

    Returns
    -------
    loss : float
        Final value of loss.
    """

    # Initialize VI model and guide
    guide = init_guide(cond_model, guidetype, guidefile = guidefile, device = device)

    optimizer = Adam({"lr": 1e-2, "amsgrad": True})

    # For some reason, JitTrace_ELBO breaks for CPU
    if device == 'cpu':
        loss = Trace_ELBO(num_particles = 1)
    else:
        loss = Trace_ELBO(num_particles = 1)
    svi = SVI(cond_model, guide, optimizer, loss=loss)

    for i in tqdm(range(n_steps)):
        if i % n_write == 0:
            for name, value in pyro.get_param_store().items():
                tqdm.write(name + ": " + str(value))

            if i > 0:
                pyro.get_param_store().save(guidefile)
            tqdm.write("")

        loss = svi.step()
        if i % n_write == 0:
            tqdm.write("Loss: " + str(loss))

    save_guide(guidetype, guidefile)


def infer(args, config, cond_model):
    """Runs a parameter inference algorithm.

    Parameters
    ----------
    args : dict
        The value corresponding to 'mode' must be a string specifying the
        inference method.
    config : dict
        Config information.
    model : callable
        The unconditioned lensing system model.

    Returns
    -------
    float
        If finding a point-estimate of the lensing system parameters, the loss
        between observed and inferred images. Otherwise, 0.
    """

    if args["mode"] == "MAP":
        loss = _infer_VI(args, cond_model)
    elif args["mode"] == "NUTS":
        _infer_NUTS(args, cond_model)
        loss = 0.
    else:
        raise KeyError("Unknown mode (select MAP or NUTS).")

    return loss

def save_posterior_predictive(model, guide, filename):
    data = guide()
    pyro.clear_param_store()  # Don't save guide parameters in mock data
    trace = poutine.trace(poutine.condition(model, data = data)).get_trace()

    mock = {}
    for tag in trace:
        entry = trace.nodes[tag]
        if entry['type'] == 'sample':
            mock[tag] = entry['value'].detach().numpy()
    np.savez(filename, **mock)

def save_mock(model, filename, use_init_values = True):
    yaml_params.set_fix_all(use_init_values)

    traced_model = poutine.trace(model)
    trace = traced_model.get_trace()

    mock = {}
    for tag in trace:
        entry = trace.nodes[tag]
        # Only save sampled components
        if entry['type'] == 'sample':
            mock[tag] = entry['value'].cpu().numpy()

    np.savez(filename, **mock)



@click.group()
@click.option("--device", default = 'cpu', help="cpu (default) or cuda")
@click.argument("yamlfile")
@click.version_option(version=0.1)
@click.pass_context
def cli(ctx, device, yamlfile):
    """This is pyrofit."""
    ctx.ensure_object(dict)

    with open(yamlfile, "r") as stream:
        yaml_config = yaml.load(stream)

    # Generate model
    module_name = yaml_config['pyrofit_module']
    my_module = importlib.import_module("pyrofit."+module_name)
    model = my_module.get_model(yaml_config, device=device)

    # Pass on information
    ctx.obj['device'] = device
    ctx.obj['yaml_config'] = yaml_config
    ctx.obj['yamlfile'] = yamlfile
    ctx.obj['model'] = model

    # Standard filenames
    ctx.obj['default_guidefile'] = yamlfile[:-5]+"_guide.pt"


@cli.command()
@click.option("--n_steps", default = 1000)
@click.option("--guidetype", default = "Delta")
@click.option("--guidefile", default = None)
#@click.option("--quantfile", default = None)
@click.pass_context
def fit(ctx, n_steps, guidetype, guidefile):
    """Parameter inference with variational methods."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    infer_VI(cond_model, guidetype, guidefile, n_steps, device = device)

@cli.command()
@click.option("--n_steps", default = 300)
@click.option("--warmup_steps", default = 100)
@click.option("--guidetype", default = None)
@click.option("--guidefile", default = None)
@click.pass_context
def sample(ctx, warmup_steps, n_steps, guidetype, guidefile):
    """Sample posterior with Hamiltonian Monte Carlo."""
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model,
            device = device)
    infer_NUTS(cond_model, n_steps, warmup_steps, device = device, guidefile =
            guidefile, guidetype = guidetype)

@cli.command()
@click.argument("mockfile")
@click.pass_context
def mock(ctx, mockfile):
    """Create mock data based on yaml file."""
    model = ctx.obj['model']
    save_mock(model, filename = mockfile)

@cli.command()
@click.option("--guidetype", default = "Delta")
@click.option("--guidefile", default = None)
@click.argument("ppdfile")
@click.pass_context
def ppd(ctx, guidetype, guidefile, ppdfile):
    """Sample from posterior predictive distribution."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    guide = init_guide(cond_model, guidetype, guidefile = guidefile)
    save_posterior_predictive(model, guide, ppdfile)
