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
from pyro.contrib.autoguide import AutoDelta  #, AutoMultivariateNormal
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

from ruamel.yaml import YAML
yaml = YAML()

from tqdm import tqdm

from . import yaml_params
#from src.lens_model import get_model



######################
# Auxilliary functions
######################

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


##############
# I/O Routines
##############

def update_yaml(yaml_section, key, val):
    """Updates `init` values in yaml file.

    We assume that `key' = 'nob_name.param_name' OR 'auto_nob_name.param_name'.
    """
    # Handle autoguide parameters
    # TODO: Check absence of model parameter interference
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


def load_image_data(config_data):
    """Loads the observed image.
    """
    # Hack to avoid bit-errors when importing FITS-file based images
    raw = np.load(config_data["image"])
    img = np.zeros(raw.shape, dtype="float32")
    img[:] += raw
    return img

def set_default_filenames(args):
    """Sets the values of fileroot and the resume file.

    These can be set explicitly. If they're not, they are derived from the yaml
    filename.
    """
    if args["fileroot"] is None:
        args["fileroot"] = args["yamlfile"][:-5]
    if args["resumefile"] is None:
        args["resumefile"] = args["fileroot"] + "_resume.pt"


def save_param_steps(args, infer_data):
    """Saves lists of parameter values.
    """
    with open(args["fileroot"] + "_infer-data.pkl", "wb") as f:
        pickle.dump(infer_data, f, pickle.HIGHEST_PROTOCOL)
        
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

def save_param_store(args):
    """Saves the parameter store so optimization can be resumed later.
    """
    pyro.get_param_store().save(args["fileroot"] + "_resume.pt")

def load_param_store(args):
    """Loads the parameter store from the resume file.
    """
    pyro.clear_param_store()
    try:
        print("Trying to load resume file: " + args["resumefile"])
        pyro.get_param_store().load(args["resumefile"])
        print("...success!")
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

def _infer_NUTS(args, cond_model):
    """Runs the NUTS HMC algorithm.

    Saves the samples and weights as well as a netcdf file for the run.

    Parameters
    ----------
    args : dict
        Command line arguments.
    cond_model : callable
        Model conditioned on an observed images.
    """
    import arviz  # required for saving results as netcdf file and plotting

    # Set up NUTS kernel
    nuts_kernel = NUTS(
        cond_model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=True)
        #hide=["image"])
    # Must run model before get_init_values() since yaml parser is in the model
    nuts_kernel.setup(warmup_steps=args["warmup_steps"])

    # Set initial values
    init_values = yaml_params.get_init_values()
    initial_trace = nuts_kernel.initial_trace
    sites = []
    sites_vae = []
    for key in init_values.keys():
        print(key, init_values[key])
        initial_trace.nodes[key]["value"] = init_values[key]
        if init_values[key].numel() == 1:
            sites.append(key)
        else:
            sites_vae.append(key)
    nuts_kernel.initial_trace = initial_trace

    # Run
    posterior = MCMC(
        nuts_kernel,
        args["n_steps"],
        warmup_steps=args["warmup_steps"],
        num_chains=args["n_chains"]).run()

    # Move traces to CPU
    posterior.exec_traces = [
        trace_to_cpu(trace) for trace in posterior.exec_traces
    ]
    
    # ._get_samepls_and_weights() does NOT work if the parameters (sites)
    # have a different size. So, in case of the 'VAE' source class,
    # the parameters as to be divided into two classes, consequently
    # two differents 'sites' must be defided!

    # Save posteriors
    hmc_posterior = EmpiricalMarginal(
        posterior, sites)._get_samples_and_weights()[0].detach().numpy()
    if len(sites_vae) > 0:
        hmc_posterior_vae = EmpiricalMarginal(
            posterior, sites_vae)._get_samples_and_weights()[0].detach().numpy()
    dict_posteriors = {}
    for i, key in enumerate(sites):
        dict_posteriors[key] = hmc_posterior[:, i]
    for i, key in enumerate(sites_vae):
        dict_posteriors[key] = hmc_posterior_vae[:,i]
    save_posteriors(args, dict_posteriors)

    # Export chain netcdf
    data = arviz.from_pyro(posterior)
    arviz.to_netcdf(data, args["fileroot"] + "_chain.nc")

def _infer_MAP(args, cond_model, n_write=10):
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
    # Guide
#    if args["guide"] == "DIRAC":
    guide = AutoDelta(cond_model)
#    if args["guide"] == "NORM":
#        guide = AutoMultivariateNormal(cond_model)

    # Perform parameter fits
    if args["opt"] == "ADAM":
        optimizer = Adam({"lr": 1e-2, "amsgrad": False})
    elif args["opt"] == "SGD":
        optimizer = SGD({"lr": 1e-11, "momentum": 0.0})
    svi = SVI(cond_model, guide, optimizer, loss=Trace_ELBO())

    # Run YAML parsers inside model to populate initial values
    cond_model()

    # Initialize AutoDelta parameters before running guide first time
    init_values = yaml_params.get_init_values()
    for key in init_values.keys():
        pyro.param("auto_" + key, init_values[key])

    guide()  # Initialize remaining guide parameters
    print("Fitting parameters:", list(pyro.get_param_store()))

    print("Initial values:")
    for name, value in pyro.get_param_store().items():
        print(name + ": " + str(value))
    print()

    # Container for monitoring progress
    infer_data = collections.defaultdict(list)

    for i in tqdm(range(args["n_steps"])):
        if i % n_write == 0:
            for name, value in pyro.get_param_store().items():
                tqdm.write(name + ": " + str(value))

            if i > 0:
                tqdm.write("Saving resume file: " + args["resumefile"])
                save_param_store(args)
                save_param_steps(args, infer_data)
            tqdm.write("")

        # Save params step by step
        for name, value in pyro.get_param_store().items():
            infer_data[name].append(value.detach().clone().numpy())

        loss = svi.step()
        # Save losses step by step
        infer_data["losses"].append(loss)
        if i % n_write == 0:
            tqdm.write("Loss: " + str(loss))

    # Last save
    save_param_store(args)
    save_param_steps(args, infer_data)

    return loss

def infer(args, config, model):
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
    image = torch.tensor(load_image_data(config["data"])).to(args["device"])
    cond_model = pyro.condition(model, {"image": image})

    if args["mode"] == "MAP":
        loss = _infer_MAP(args, cond_model)
    elif args["mode"] == "NUTS":
        _infer_NUTS(args, cond_model)
        loss = 0.
    else:
        raise KeyError("Unknown mode (select MAP or NUTS).")

    return loss


###############
# Main PYROLENS
###############

@click.command()
@click.argument("command", required = True)
@click.argument("yamlfile", required = True)
@click.option(
    "--n_steps", default=100, help="Number of steps (default 100).")
@click.option(
    "--fileroot",
    default=None,
    help="File root, default is yamlfile name without extension.")
@click.option(
    "--resumefile",
    default=None,
    help="Resume file name, default is derived from fileroot.")
@click.option(
    "--device",
    default="cpu",
    help="Set to 'cuda' to run on GPU (default 'cpu').")
@click.option(
    "--mode",
    default="MAP",
    help="Inference mode: MAP or NUTS (default 'MAP').")
@click.option(
    "--opt",
    default="ADAM",
    help="Optimization mode: ADAM or SGD (default 'ADAM')")
@click.option(
    "--n_chains",
    default=1,
    help="Number of chains for NUTS sampler (default 1).")
@click.option(
    "--warmup_steps",
    default=50,
    help="Warmup steps for NUTS sampler (default 50).")
@click.option(
    "--outyaml", default=None, help="Output updated yaml file when using MAP.")
@click.option(
    "--n_pixel", default=None, help="Pixel dimensions for mock image output (use 'nx,ny').")
@click.version_option(version=0.1)
# TODO: Can be removed?
#@click.option(
#    "--guide",
#    default="DIRAC",
#    help="Guide used by SVI: DIRAC or NORM (default 'DIRAC')")
#@click.option("--image", default=None, help="Overwrite image data file.")
def cli(**kwargs):
    """This is PyroLens.

    COMMAND: mock infer
    """

    # If a resumefile was provided, presumably the user wants to resume running
    resume = ("resumefile" in kwargs)

    # Set default root
    set_default_filenames(kwargs)

    # Obtain YAML file
    with open(kwargs["yamlfile"], "r") as stream:
        config = yaml.load(stream)

    module_name = config['pyrofit_module']
    my_module = importlib.import_module("pyrofit."+module_name)

    if kwargs["command"] == "mock":
        model = my_module.get_model(config, device=kwargs["device"])
        my_module.save_mock(config, kwargs, model)
    elif kwargs["command"] == "infer":
        # Generate model
        model = my_module.get_model(config, device=kwargs["device"])
        # Try restoring param store from previous run
        if resume:
            load_param_store(kwargs)
        try:
            loss = infer(kwargs, config, model)
            config["data"]["loss"] = loss
        except KeyboardInterrupt:
            print("Interrupted. Saving resume file.")

        save_param_store(kwargs)

        if kwargs["outyaml"] is not None:
            write_yaml(config, kwargs["outyaml"])
    else:
        print("COMMAND unknown.  Run with --help for more info.")
