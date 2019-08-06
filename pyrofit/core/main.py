# pyrofit
#
# version: v0.1
#
# author: Christoph Weniger <c.weniger@uva.nl>
# date: Jan - July 2019

import click
import numpy as np
import importlib
import inspect
import torch
import pyro
from pyro import poutine
from pyro.contrib.autoguide import (AutoDelta, AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation, AutoDiagonalNormal, AutoMultivariateNormal,
        init_to_sample)
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.infer.mcmc import MCMC, NUTS, util
from pyro.optim import Adam
#from ruamel.yaml import YAML
#yaml = YAML()
import yaml
from tqdm import tqdm

from . import yaml_params2
from . import decorators


######################
# Auxilliary functions
######################

def get_conditioned_model(yaml_section, model, device='cpu'):
    if yaml_section is None:
        return model
    conditions = {}
    for name , val in yaml_section.items():
        conditions[name] = yaml_params2._parse_val(name, val, device = device)
    cond_model = pyro.condition(model, conditions)
    return cond_model

def load_param_store(paramfile):
    """Loads the parameter store from the resume file.
    """
    pyro.clear_param_store()
    try:
        pyro.get_param_store().load(paramfile)
        print("Loading param_store file:", paramfile)
    except FileNotFoundError:
        print("...no resume file not found. Starting from scratch.")

def init_guide(cond_model, guidetype, guidefile = None):
    if guidefile is not None:
        load_param_store(guidefile)
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


#######################
# Inference
#######################

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
        guide = init_guide(cond_model, guidetype, guidefile = guidefile)
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


def infer_VI(cond_model, guidetype, guidefile, n_steps, n_write=100, device = 'cpu'):
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
    guide = init_guide(cond_model, guidetype, guidefile = guidefile)

    optimizer = Adam({"lr": 1e-2, "amsgrad": True})

    # For some reason, JitTrace_ELBO breaks for CPU
    if device == 'cpu':
        loss = Trace_ELBO()
    else:
        loss = Trace_ELBO()
    svi = SVI(cond_model, guide, optimizer, loss=loss)

    print()
    print("##################")
    print("# Initial values #")
    print("##################")
    for name, value in pyro.get_param_store().items():
        print(name + ": " + str(value))
    print()

    print("##############")
    print("# Optimizing #")
    print("##############")
    with tqdm(total = n_steps) as t:
        for i in range(n_steps):
            if i % n_write == 0:
                pyro.get_param_store().save(guidefile)
            loss = svi.step()
            t.postfix = "loss=%.3f"%loss
            t.update()

    print()
    print("################")
    print("# Final values #")
    print("################")
    for name, value in pyro.get_param_store().items():
        print(name + ": " + str(value))
    print()

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
    yaml_params2.set_fix_all(use_init_values)

    traced_model = poutine.trace(model)
    trace = traced_model.get_trace()

    mock = {}
    for tag in trace:
        entry = trace.nodes[tag]
        # Only save sampled components
        if entry['type'] == 'sample':
            mock[tag] = entry['value'].cpu().detach().numpy()

    np.savez(filename, **mock)


########################
# Command line interface
########################

@click.group()
@click.option("--device", default = 'cpu', help="cpu (default) or cuda")
@click.argument("yamlfile")
@click.version_option(version=0.1)
@click.pass_context
def cli(ctx, device, yamlfile):
    """This is pyrofit."""
    ctx.ensure_object(dict)

#    with open(yamlfile, "r") as stream:
        #yaml_config = yaml.load(stream)
    yaml_config = decorators.load_yaml(yamlfile)

    # Import module
    module_name = yaml_config['pyrofit']['module']
    my_module = importlib.import_module("pyrofit."+module_name)

    # Get model...
    model_name = yaml_config['pyrofit']['model']
    try:
        model = getattr(my_module, model_name)
    except AttributeError:
        # And try to instantiate if necessary
        model = decorators.instantiate(model_name)[model_name]

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
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    save_mock(cond_model, filename = mockfile)

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
