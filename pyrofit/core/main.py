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
import pickle
from pyro import poutine
from pyro.contrib.autoguide import (AutoDelta, AutoLowRankMultivariateNormal,
        AutoLaplaceApproximation, AutoDiagonalNormal, AutoMultivariateNormal,
        init_to_sample)
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, JitTrace_ELBO
from pyro.infer.mcmc import MCMC, NUTS, util
from pyro.optim import Adam, SGD
#from ruamel.yaml import YAML
#yaml = YAML()
import yaml
from tqdm import tqdm

from . import yaml_params2
from . import decorators
from .guides import init_guide
from .utils import load_param_store


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

def save_guide(guidefile):
    print("Saving guide:", guidefile)
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


def infer_NUTS(cond_model, n_steps, warmup_steps, n_chains = 1, device = 'cpu',
               guidefile = None, guide_conf = None, mcmcfile=None):
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
        guide = init_guide(cond_model, guide_conf, guidefile = guidefile, device = device)
        sample = guide()
        for key in initial_params.keys():
            initial_params[key] = transforms[key](sample[key].detach())

    # FIXME: In the case of DiagonalNormal, results have to be mapped back onto unpacked latents
    if guide_conf['type'] == "DiagonalNormal":
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
    mcmc = MCMC(
        nuts_kernel, n_steps, warmup_steps=warmup_steps,
        initial_params=initial_params, num_chains=n_chains
    )
    mcmc.run()

    # This block lets the posterior be pickled
    mcmc.sampler = None
    mcmc.kernel.potential_fn = None
    mcmc._cache = {}

    print(f"Saving MCMC object to {mcmcfile}")
    with open(mcmcfile, "wb") as f:
        pickle.dump(mcmc, f, pickle.HIGHEST_PROTOCOL)


def infer_VI(cond_model, guide_conf, guidefile, n_steps, lr = 1e-3, n_write=300,
        device = 'cpu', n_particles = 1, conv_th = 0.):
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
    guide = init_guide(cond_model, guide_conf, guidefile = guidefile, device = device)

    optimizer = Adam({"lr": lr, "amsgrad": True})
    #optimizer = SGD({"lr": lr})

    # For some reason, JitTrace_ELBO breaks for CPU
    loss = Trace_ELBO(num_particles = n_particles)
    svi = SVI(cond_model, guide, optimizer, loss=loss)

    print()
    print("##################")
    print("# Initial values #")
    print("##################")
    print("Parameter store:")
    for name, value in pyro.get_param_store().items():
        print(name + ": " + str(value))
    print()
    print("Guide:")
    for name, value in guide().items():
        print(name + ": " + str(value))
    print()

    print("################################")
    print("# Maximizing ELBO. Hang tight. #")
    print("################################")
    losses = []
    with tqdm(total = n_steps) as t:
        for i in range(n_steps):
            if i % n_write == 0:
                pyro.get_param_store().save(guidefile)
            loss = svi.step()
            losses.append(loss)
            t.postfix = "loss=%.3f"%loss
            t.update()

            if len(losses) > 100:
                dl = (np.mean(losses[-100:-80]) - np.mean(losses[-20:]))/80
                if conv_th > 0 and dl < conv_th:
                    print("Convergence criterion reached: d_loss/d_step < %.3e"%conv_th)
                    break

    print()
    print("################")
    print("# Final values #")
    print("################")
    print("Parameter store:")
    for name, value in pyro.get_param_store().items():
        print(name + ": " + str(value))
    print()
    print("Guide:")
    for name, value in guide().items():
        print(name + ": " + str(value))
    print()

    save_guide(guidefile)


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


def save_posterior_predictive(model, guide, filename, N = 300):
    traces = [poutine.trace(poutine.condition(model, data = guide())).get_trace()
            for i in range(N)]
    pyro.clear_param_store()  # Don't save guide parameters in mock data

    mock = {}
    for tag in traces[0]:
        if traces[0].nodes[tag]['type'] == 'sample':
            if N == 1:
                mock[tag] = traces[0].nodes[tag]['value'].detach().cpu().numpy()
            else:
                mock[tag] = [trace.nodes[tag]['value'].detach().cpu().numpy() for trace in traces]
    np.savez(filename, **mock)
    print("Saved %i sample(s) from posterior predictive distribution to %s"%(N,filename))


def save_lossgrad(cond_model, guide, filename):
    # Calculate loss and gradients (implicitely done when evaluating loss_and_grads)
    with poutine.trace(param_only=True) as param_capture:
        loss = Trace_ELBO().loss_and_grads(cond_model, guide)

    print()
    print("Loss =", loss)

    # Zero grad (seems to be not necessary)
    #params = set(site["value"].unconstrained()
    #             for site in param_capture.trace.nodes.values())
    #pyro.infer.util.zero_grads(params_dict)

    print()
    print("Gradients:")
    param_dict = {site['name']: site["value"].unconstrained().grad.detach().numpy()
                 for site in param_capture.trace.nodes.values()}
    for name, param in param_dict.items():
        print(name + " :", param)

    param_dict["loss"] = loss

    np.savez(filename, **param_dict)
    print()
    print("Save loss and grads to %s"%(filename))


def save_mock(model, filename, use_init_values = True):
    yaml_params2.set_fix_all(use_init_values)

    traced_model = poutine.trace(model)
    trace = traced_model.get_trace()

    print(f"Mock data log prob sum: {trace.log_prob_sum()}")

    mock = {}
    for tag in trace:
        entry = trace.nodes[tag]
        # Only save sampled components
        if entry['type'] == 'sample':
            mock[tag] = entry['value'].cpu().detach().numpy()

    np.savez(filename, **mock)

# TODO: Rewrite Info Command
#def info(cond_model, guidetype, guidefile, device = 'cpu'):
#    # Initialize VI model and guide
#    guide = init_guide(cond_model, guidetype, guidefile = guidefile)
#
#    loss = Trace_ELBO()
#    svi = SVI(cond_model, guide, optimizer, loss=loss)
#    loss = svi.step()
#
#    print("LOSS =", loss)
#    print()
#
#    print("####################")
#    print("# Parameter values #")
#    print("####################")
#    for name, value in pyro.get_param_store().items():
#        print(name + ": " + str(value))


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
    print(
        """
         (                         (
         )\ )                      )\ )           )
        (()/(   (      (          (()/(   (    ( /(
         /(_))  )\ )   )(     (    /(_))  )\   )\())
        (_))   (()/(  (()\    )\  (_))_| ((_) (_))/
        | _ \   )(_))  ((_)  ((_) | |_    (_) | |_
        |  _/  | || | | '_| / _ \ | __|   | | |  _|
        |_|     \_, | |_|   \___/ |_|     |_|  \__|
                |__/

          high-dimensional modeling for everyone
        """
    )

    # Load the model's module and (re)parse all settings and variables
    yaml_config, my_module = decorators.load_yaml(yamlfile, device = device)

    # Get model...
    model_name = yaml_config['pyrofit']['model']
    try:
        model = getattr(my_module, model_name)
    except AttributeError:
        # And try to instantiate if necessary
        model = decorators.instantiate(model_name, device = device)[model_name]

    # Pass on information
    ctx.obj['device'] = device
    ctx.obj['yaml_config'] = yaml_config
    ctx.obj['yamlfile'] = yamlfile
    ctx.obj['model'] = model

    # Standard filenames
    ctx.obj['default_guidefile'] = yamlfile[:-5]+".guide.pt"


@cli.command()
@click.option("--n_steps", default = 1000)
#@click.option("--guide", default = "Delta", help = "Guide type (default Delta).")
@click.option("--guidefile", default = None, help = "Guide filename (default YAML.guide.pt.")
@click.option("--lr", default = 1e-3, help = "Learning rate (default 1e-3).")
@click.option("--n_write", default = 200, help = "Steps after which guide is written (default 200).")
@click.option("--n_particles", default = 1, help = "Particles used in optimization step (default 1).")
@click.option("--conv_th", default = 1e-3, help = "Convergence threshold (default 1e-3).")
#@click.option("--quantfile", default = None)
@click.pass_context
def fit(ctx, n_steps, guidefile, lr, n_write, n_particles, conv_th):
    """Parameter inference with variational methods."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    guide_conf = yaml_config['guide']
    infer_VI(cond_model, guide_conf, guidefile, n_steps, device = device, lr =
            lr, n_write = n_write, n_particles = n_particles, conv_th =
            conv_th)

@cli.command()
@click.option("--n_steps", default = 300)
@click.option("--warmup_steps", default = 100)
#@click.option("--guide", default = None)
@click.option("--guidefile", default = None)
@click.option("--mcmcfile", default=None)
@click.pass_context
def sample(ctx, warmup_steps, n_steps, guidefile, mcmcfile):
    """Sample posterior with Hamiltonian Monte Carlo."""
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model,
            device = device)
    guide_conf = yaml_config['guide']

    if mcmcfile is None:
        mcmcfile = ctx.obj['yamlfile'][:-5] + ".mcmc.pkl"

    infer_NUTS(cond_model, n_steps, warmup_steps, device=device,
               guidefile=guidefile, guide_conf=guide_conf, mcmcfile=mcmcfile)

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
    print("Save mock data to %s"%mockfile)

@cli.command()
#@click.option("--guide", default = "Delta")
@click.option("--guidefile", default = None)
@click.option("--n_samples", default = 1, help = "Number of samples (default 1).")
@click.argument("ppdfile")
@click.pass_context
def ppd(ctx, guidefile, ppdfile, n_samples):
    """Sample from posterior predictive distribution."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    guide_conf = yaml_config['guide']
    my_guide = init_guide(cond_model, guide_conf, guidefile = guidefile, device = device)
    save_posterior_predictive(model, my_guide, ppdfile, N = n_samples)

@cli.command()
#@click.option("--guide", default = "Delta")
@click.option("--guidefile", default = None)
@click.argument("outfile")
@click.pass_context
def lossgrad(ctx, guidefile, outfile):
    """Store model loss and gradient of guide parameters."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    guide_conf = yaml_config['guide']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    my_guide = init_guide(cond_model, guide_conf, guidefile = guidefile, device = device)
    save_lossgrad(cond_model, my_guide, outfile)

#@cli.command()
#@click.option("--guide", default = "Delta", help = "Guide type.")
#@click.option("--guidefile", default = None, help = "Guide filename.")
#@click.pass_context
#def info(ctx, guide, guidefile):
#    """Parameter inference with variational methods."""
#    if guidefile is None: guidefile = ctx.obj['default_guidefile']
#    model = ctx.obj['model']
#    device = ctx.obj['device']
#    yaml_config = ctx.obj['yaml_config']
#    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
#    info(cond_model, guide, guidefile, device = device)
