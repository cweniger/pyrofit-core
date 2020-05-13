"""
# pyrofit
#
# version: v0.1
#
# author: Christoph Weniger <c.weniger@uva.nl>
# date: Jan - July 2019
"""

import importlib
import inspect
import pickle
from collections import defaultdict

import click
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.contrib.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    init_to_sample,
)
from pyro.infer import SVI, Trace_ELBO, RenyiELBO
from pyro.infer.mcmc import MCMC, NUTS, util
from pyro.optim import SGD, Adam
from tqdm import tqdm

from . import decorators, yaml_params2
from .guides import init_guide
from .conlearn import ConLearn

# Modified ELBOs  (deprecated)
#from .trace_elbo import Trace_ELBO
#from .renyi_elbo import RenyiELBO

######################
# Auxilliary functions
######################


def get_conditioned_model(yaml_section, model, device="cpu"):
    if yaml_section is None:
        return model
    conditions = {}
    for name, val in yaml_section.items():
        conditions[name] = yaml_params2._parse_val(name, val, device=device)
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
        d = {s["name"]: b for s, b in unpack_fn(u)}
        return potential_fn(d) + logdet

    return transformed_potential_fn


def infer_sample(
    cond_model,
    n_steps,
    warmup_steps,
    n_chains=1,
    device="cpu",
    guidefile=None,
    guide_conf=None,
    mcmcfile=None,
):
    """Runs the NUTS HMC algorithm.

    Saves the samples and weights as well as a netcdf file for the run.

    Parameters
    ----------
    args : dict
        Command line arguments.
    cond_model : callable
        Model conditioned on an observed images.
    """
    initial_params, potential_fn, transforms, prototype_trace = util.initialize_model(
        cond_model
    )

    if guidefile is not None:
        guide = init_guide(cond_model, guide_conf, guidefile=guidefile, device=device)
        sample = guide()
        for key in initial_params.keys():
            initial_params[key] = transforms[key](sample[key].detach())

    # FIXME: In the case of DiagonalNormal, results have to be mapped back onto unpacked latents
    if guide_conf["type"] == "DiagonalNormal":
        transform = guide.get_transform()
        unpack_fn = lambda u: guide.unpack_latent(u)
        potential_fn = make_transformed_pe(potential_fn, transform, unpack_fn)
        initial_params = {"z": torch.zeros(guide.get_posterior().shape())}
        transforms = None

    def fun(*args, **kwargs):
        res = potential_fn(*args, **kwargs)
        return res

    nuts_kernel = NUTS(
        potential_fn=fun,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        jit_compile=False,
        max_tree_depth=10,
        transforms=transforms,
        step_size=1.0,
    )
    nuts_kernel.initial_params = initial_params

    # Run
    mcmc = MCMC(
        nuts_kernel,
        n_steps,
        warmup_steps=warmup_steps,
        initial_params=initial_params,
        num_chains=n_chains,
    )
    mcmc.run()

    # This block lets the posterior be pickled
    mcmc.sampler = None
    mcmc.kernel.potential_fn = None
    mcmc._cache = {}

    print(f"Saving MCMC object to {mcmcfile}")
    with open(mcmcfile, "wb") as f:
        pickle.dump(mcmc, f, pickle.HIGHEST_PROTOCOL)


LOSS_SUM = []


def infer_fit(
    cond_model,
    guide_conf,
    guidefile,
    n_steps,
    min_steps=1000,
    lr=1e-3,
    n_write=300,
    device="cpu",
    n_particles=1,
    conv_th=0.0,
    verbose=True,
):
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
    guide = init_guide(cond_model, guide_conf, guidefile=guidefile, device=device)

    optimizer = Adam({"lr": lr, "amsgrad": False, "weight_decay": 0.0})
    # optimizer = SGD({"lr": lr, "momentum": 0.9})

    # For some reason, JitTrace_ELBO breaks for CPU
    loss = Trace_ELBO(num_particles=n_particles)
    #    guide = poutine.trace(guide)
    svi = SVI(cond_model, guide, optimizer, loss=loss)

    if verbose:
        print()
        print("##################")
        print("# Initial values #")
        print("##################")
        print("Parameter store:")
        for name, value in pyro.get_param_store().items():
            print(name + ": " + str(value))
        print()
        print("Guide:")
        for name, value in guide()[1].items():
            print(name + ": " + str(value))
        print()

    print("################################")
    print("# Maximizing ELBO. Hang tight. #")
    print("################################")
    losses = []
    with tqdm(total=n_steps) as t:
        for i in range(n_steps):
            if i % n_write == 0:
                pyro.get_param_store().save(guidefile)

            loss = svi.step()
            losses.append(loss)
            minloss = min(losses)
            t.postfix = "loss=%.3f (%.3f)" % (loss, minloss)
            t.update()

            if len(losses) > min_steps and len(losses) > 100:
                dl = (np.mean(losses[-100:-80]) - np.mean(losses[-20:])) / 80
                if conv_th > 0.0 and dl < conv_th:
                    print(
                        "Convergence criterion reached: d_loss/d_step < %.3e" % conv_th
                    )
                    break
            # print(np.mean(losses[-500:]))

    if verbose:
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


# TODO: Merge with infer_fit if possibel
def infer_CS(
    cond_model,
    guide_conf,
    guidefile,
    n_wake,
    n_sleep,
    gen_samples = True,
    n_rounds = 1,
    n_simulate = 1000,
    sleep_batch_size = 3,
    wake_batch_size = 1,
    alpha = 1.,
    lr_wake=1e-3,
    lr_sleep=1e-3,
    n_write=300,
    device="cpu",
    verbose=True,
):
    """Contrastive inference.

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
    guide = init_guide(cond_model, guide_conf, guidefile=guidefile, device=device, default_observations = True)

    wake_optimizer = Adam({"lr": lr_wake, "amsgrad": False, "weight_decay": 0.0})
    sleep_optimizer = Adam({"lr": lr_sleep, "amsgrad": False, "weight_decay": 0.0})

    site_names = guide_conf['sleep_sites']

    conlearn= ConLearn(cond_model, guide, sleep_optimizer, training_batch_size = sleep_batch_size, site_names = site_names)

    # Default is standard ELBO loss
    if alpha == 1.:
        wake_loss = Trace_ELBO(num_particles=wake_batch_size)
    else:
        wake_loss = RenyiELBO(alpha=alpha, num_particles=wake_batch_size)

    svi = SVI(cond_model, guide, wake_optimizer, loss=wake_loss)

    if verbose:
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

    print("###################")
    print("# Wake and Sleep. #")
    print("###################")

    sleep_losses = []
    wake_losses = []

    # Rounds
    for r in range(n_rounds):
        print("\nRound %i:"%r) 

        # Wake phase
        with tqdm(total=n_wake, desc='Wake') as t:
            guide.wake()
            for i in range(n_wake):
                if (i+1) % n_write == 0:
                    pyro.get_param_store().save(guidefile)

                loss = svi.step()

                wake_losses.append(loss)
                minloss = min(wake_losses)
                t.postfix = "loss=%.3f (%.3f)" % (loss, minloss)
                t.update()

        # Sleep phase
        if n_sleep > 0:
            conlearn.simulate(n_simulate, replace = True, gen_samples = gen_samples)
            gen_samples = False  # Only in first round

        with tqdm(total=n_sleep, desc='Sleep') as t:
            guide.sleep()
            for i in range(n_sleep):
                if (i+1) % n_write == 0:
                    pyro.get_param_store().save(guidefile)

                loss = conlearn.step()
                sleep_losses.append(loss)
                minloss = min(sleep_losses)
                t.postfix = "loss=%.3f (%.3f)" % (loss, minloss)
                t.update()

    if verbose:
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


def save_posterior_predictive(model, guide, filename, N=300):
    if N == 1:
        mock = {}
        guide_trace = poutine.trace(guide).get_trace()
        trace = poutine.trace(poutine.condition(model, data=guide_trace)).get_trace()
        for tag in trace:
            if trace.nodes[tag]["type"] == "sample":
                mock[tag] = trace.nodes[tag]["value"].detach().cpu().numpy()
    else:
        mock = defaultdict(list)
        for i in range(N):
            # Faster way if we don't need `deterministic` statements.
            # Literally just samples from the guide.
            #
            # for tag, value in guide()[1].items():
            #     mock[tag].append(value.detach().cpu().numpy())
            # continue

            guide_trace = poutine.trace(guide).get_trace()
            trace = poutine.trace(poutine.condition(model, data=guide_trace)).get_trace()
            for tag in trace:
                if trace.nodes[tag]["type"] == "sample":
                    mock[tag].append(trace.nodes[tag]["value"].detach().cpu().numpy())

    np.savez(filename, **mock)
    print(
        "Saved %i sample(s) from posterior predictive distribution to %s"
        % (N, filename)
    )


def dictlist2listdict(L):
    """Take list of dictonaries and turn it into dictionary of lists."""
    if len(L) == 1:
        return L[0]  # Nothing to do in this case
    O = {}
    for key in L[0].keys():
        values = [L[i][key] for i in range(len(L))]
        O[key] = values
    return O


def save_lossgrad(cond_model, guide, verbose, filename, N=2, skip_grad=False):
    param_dict_list = []

    # Note: This is a hacky way of extracting guide samples while calculating the loss
    guide_ret = [None]

    def wrapped_guide(*args, **kwargs):
        guide_ret[0] = guide(*args, **kwargs)
        return guide_ret[0]

    for i in range(N):
        # Calculate loss and gradients (implicitely done when evaluating loss_and_grads)
        with poutine.trace(param_only=True) as param_capture:
            loss = Trace_ELBO().loss_and_grads(cond_model, wrapped_guide)

        guide_samples = guide_ret[0]
        for key, value in guide_samples.items():
            guide_samples[key] = value.detach().cpu().numpy()
        #        for site in guide.get_trace().nodes.values():
        #            if site['type'] == 'sample':
        #                guide_samples[site['name']] = site['value'].detach().numpy()

        if verbose:
            print()
            print("Loss =", loss)
            print()
            print("Guide samples:")
            for key, value in guide_samples.items():
                print(key + " :", value)

        # Zero grad (seems to be not necessary)
        # params = set(site["value"].unconstrained()
        #             for site in param_capture.trace.nodes.values())
        # pyro.infer.util.zero_grads(params_dict)

        param_dict = {}

        if not skip_grad:
            if verbose:
                print()
                print("Parameter gradients:")

            for site in param_capture.trace.nodes.values():
                grad = site["value"].unconstrained().grad
                if grad is not None:
                    param_dict[site["name"]] = grad.detach().cpu().numpy()

            if verbose:
                for name, param in param_dict.items():
                    print(name + " :", param)

        param_dict["loss"] = loss
        param_dict.update(guide_samples)

        param_dict_list.append(param_dict)

    param_dict = dictlist2listdict(param_dict_list)

    np.savez(filename, **param_dict)
    print()
    print("Save loss and grads to %s" % (filename))


def save_mock(model, filename, use_init_values=True):
    yaml_params2.set_fix_all(use_init_values)

    traced_model = poutine.trace(model)
    trace = traced_model.get_trace()

    print(f"Mock data log prob sum: {trace.log_prob_sum()}")

    mock = {}
    for tag in trace:
        entry = trace.nodes[tag]
        # Only save sampled components
        if entry["type"] == "sample":
            mock[tag] = entry["value"].cpu().detach().numpy()

    np.savez(filename, **mock)


########################
# Command line interface
########################


@click.group()
@click.option("--device", default="cpu", help="cpu (default) or cuda")
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
    yaml_config, my_module = decorators.load_yaml(yamlfile, device=device)

    # Get model...
    model_name = yaml_config["pyrofit"]["model"]
    try:
        model = getattr(my_module, model_name)
    except AttributeError:
        # And try to instantiate if necessary
        model = decorators.instantiate(model_name, device=device)[model_name]

    # Pass on information
    ctx.obj["device"] = device
    ctx.obj["yaml_config"] = yaml_config
    ctx.obj["yamlfile"] = yamlfile
    ctx.obj["model"] = model

    # Standard filenames
    ctx.obj["default_guidefile"] = yamlfile[:-5] + ".guide.pt"


@cli.command()
@click.option("--n_wake", default=0)
@click.option("--n_sleep", default=0)
@click.option("--n_rounds", default=1, help="Number of rounds (default 1).")
@click.option(
    "--guidefile", default=None, help="Guide filename (default YAML.guide.pt."
)
@click.option("--lr_wake", default=1e-3, help="Wake learning rate (default 1e-3).")
@click.option("--lr_sleep", default=1e-3, help="Sleep learning rate (default 1e-3).")
@click.option(
    "--n_write", default=200, help="Steps after which guide is written (default 200)."
)
@click.option(
    "--n_wake_particles", default=1, help="Particles used in wake step (default 1)."
)
@click.option(
    "--n_sleep_particles", default=3, help="Particles used in sleep step (default 3)."
)
@click.option(
    "--n_simulate", default=1000, help="Number of simulations used in sleep phase (default 1000)."
)
@click.option(
    "--alpha", default=1., help="The order of alpha-divergence (default 1., which corresponds to ELBO)."
)
@click.option(
    "--gen_samples/--no-gen_samples", default=True, help="Sample first simulations from generative model (default True)"
)
@click.option(
    "--verbose/--no-verbose", default=False, help="Print more messages (default False)"
)
@click.pass_context
def wakesleep(ctx, n_wake, n_sleep, n_rounds, guidefile, lr_wake, lr_sleep, n_write, n_wake_particles, n_sleep_particles, n_simulate, alpha, gen_samples, verbose):
    """Parameter inference with contrastive learning."""
    if guidefile is None:
        guidefile = ctx.obj["default_guidefile"]
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    guide_conf = yaml_config["guide"]
    infer_CS(
        cond_model,
        guide_conf,
        guidefile,
        n_wake,
        n_sleep,
        gen_samples=gen_samples,
        n_rounds=n_rounds,
        n_simulate=n_simulate,
        sleep_batch_size=n_sleep_particles,
        wake_batch_size=n_wake_particles,
        device=device,
        lr_wake=lr_wake,
        lr_sleep=lr_sleep,
        n_write=n_write,
        verbose=verbose,
        alpha=alpha
    )


@cli.command()
@click.option("--n_steps", default=1000)
@click.option(
    "--guidefile", default=None, help="Guide filename (default YAML.guide.pt."
)
@click.option("--min_steps", default=0, help="Do at least this number of steps.")
@click.option("--lr", default=1e-3, help="Learning rate (default 1e-3).")
@click.option(
    "--n_write", default=200, help="Steps after which guide is written (default 200)."
)
@click.option(
    "--n_particles", default=1, help="Particles used in optimization step (default 1)."
)
@click.option("--conv_th", default=0.0, help="Convergence threshold (default 0).")
@click.option(
    "--verbose/--no-verbose", default=False, help="Print more messages (default False)"
)
@click.pass_context
def fit(ctx, n_steps, guidefile, min_steps, lr, n_write, n_particles, conv_th, verbose):
    """Parameter inference with variational methods."""
    if guidefile is None:
        guidefile = ctx.obj["default_guidefile"]
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    guide_conf = yaml_config["guide"]
    infer_fit(
        cond_model,
        guide_conf,
        guidefile,
        n_steps,
        device=device,
        min_steps=min_steps,
        lr=lr,
        n_write=n_write,
        n_particles=n_particles,
        conv_th=conv_th,
        verbose=verbose,
    )


@cli.command()
@click.option("--n_steps", default=300)
@click.option("--warmup_steps", default=100)
# @click.option("--guide", default = None)
@click.option("--guidefile", default=None)
@click.option("--mcmcfile", default=None)
@click.pass_context
def sample(ctx, warmup_steps, n_steps, guidefile, mcmcfile):
    """Sample posterior with Hamiltonian Monte Carlo."""
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    guide_conf = yaml_config["guide"]

    if mcmcfile is None:
        mcmcfile = ctx.obj["yamlfile"][:-5] + ".mcmc.pkl"

    infer_sample(
        cond_model,
        n_steps,
        warmup_steps,
        device=device,
        guidefile=guidefile,
        guide_conf=guide_conf,
        mcmcfile=mcmcfile,
    )


@cli.command()
@click.argument("mockfile")
@click.pass_context
def mock(ctx, mockfile):
    """Create mock data based on yaml file."""
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    save_mock(cond_model, filename=mockfile)
    print("Save mock data to %s" % mockfile)


@cli.command()
# @click.option("--guide", default = "Delta")
@click.option("--guidefile", default=None)
@click.option("--n_samples", default=1, help="Number of samples (default 1).")
@click.argument("ppdfile")
@click.pass_context
def ppd(ctx, guidefile, ppdfile, n_samples):
    """Sample from posterior predictive distribution."""
    if guidefile is None:
        guidefile = ctx.obj["default_guidefile"]
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    guide_conf = yaml_config["guide"]
    my_guide = init_guide(cond_model, guide_conf, default_observations = True, guidefile=guidefile, device=device)
    save_posterior_predictive(model, my_guide, ppdfile, N=n_samples)


@cli.command()
# @click.option("--guide", default = "Delta")
@click.option("--guidefile", default=None)
@click.option("--n_samples", default=1, help="Number of samples (default 1).")
@click.option(
    "--verbose/--no-verbose", default=False, help="Print more messages (default False)"
)
@click.option(
    "--grads/--no-grads", default=True, help="Include gradients (default --grad)."
)
def lossgrad(ctx, guidefile, n_samples, verbose, outfile, grads):
    """Store model loss and gradient of guide parameters."""
    if guidefile is None:
        guidefile = ctx.obj["default_guidefile"]
    model = ctx.obj["model"]
    device = ctx.obj["device"]
    yaml_config = ctx.obj["yaml_config"]
    guide_conf = yaml_config["guide"]
    cond_model = get_conditioned_model(
        yaml_config["conditioning"], model, device=device
    )
    my_guide = init_guide(cond_model, guide_conf, guidefile=guidefile, device=device)
    save_lossgrad(
        cond_model, my_guide, verbose, outfile, N=n_samples, skip_grad=not grads
    )
