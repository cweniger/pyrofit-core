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
from pyro.contrib.autoguide import AutoDelta, AutoLaplaceApproximation, AutoDiagonalNormal, AutoMultivariateNormal, init_to_sample
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal
from pyro.infer.mcmc import MCMC, NUTS, HMC
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
        conditions[name] = yaml_params._parse_val(val, device = device, name = name)
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

#def load_image_data(config_data):
#    """Loads the observed image.
#    """
#    # Hack to avoid bit-errors when importing FITS-file based images
#    raw = np.load(config_data["image"])
#    img = np.zeros(raw.shape, dtype="float32")
#    img[:] += raw
#    return img

def set_default_filenames(kwargs):
    """Sets the values of fileroot and the resume file.

    These can be set explicitly. If they're not, they are derived from the yaml
    filename.
    """
    if kwargs["fileroot"] is None:
        kwargs["fileroot"] = kwargs["yamlfile"][:-5]
   # if kwargs["resumefile"] is None:
   # kwargs["resumefile"] = kwargs["fileroot"] + "_resume.pt"
    if kwargs["mockfile"] is None:
        kwargs["mockfile"] = kwargs["fileroot"] + "_mock.npz"
    if kwargs["guidefile"] is None:
        kwargs["guidefile"] = kwargs["fileroot"] + "_guide.npz"
#    if kwargs["quantfile"] is None:
#        kwargs["quantfile"] = kwargs["fileroot"] + "_quantiles.npz"

def save_quantfile(filename, data):
    out = {}
    for x, y in data.items():
        out[x] = y#.detach().cpu().numpy()
    np.savez(filename, **out)

def get_transforms_initial_params(filename, cond_model):
    # Transformation to standard normal
    try:
        data = np.load(filename)
    except IOError:
        print("Cannot lead quantfile")
        return None, None

    # Transformation to unconstrained parameters
    trace = poutine.trace(cond_model).get_trace()
    transforms = {}
    initial_params = {}
    for name in trace.stochastic_nodes:
        trans1 = biject_to(trace.nodes[name]['fn'].support).inv
        loc = trans1(data[name][1])
        scale = (trans1(data[name][2])-trans1(data[name][0]))/2
        trans2 = AffineTransform(loc, scale).inv
        #trans2 = AffineTransform(0., 1.)
        transforms[name] = ComposeTransform((trans1, trans2))
        initial_params[name] = torch.zeros_like(loc)
    return transforms, initial_params
#
#    for name, val in data.items():
#        if name[:6] == "scale_":
#            continue
#        scale = 1/torch.tensor(data["scale_"+name[4:]])
#        loc = -torch.tensor(val)*scale
#        transform = AffineTransform(loc, scale)
#        transforms[name[4:]] = ComposeTransform((transforms[name[4:]], transform))
#    return transforms

#def save_param_steps(args, infer_data):
#    """Saves lists of parameter values.
#    """
#    with open(args["fileroot"] + "_infer-data.pkl", "wb") as f:
#        pickle.dump(infer_data, f, pickle.HIGHEST_PROTOCOL)
        
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

def load_param_store(guidefile):
    """Loads the parameter store from the resume file.
    """
    pyro.clear_param_store()
    try:
        #print("Trying to load resume file: ", args["resumefile"])
        pyro.get_param_store().load(guidefile)
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

def init_guide(cond_model, filename, method):
    if filename is not None:
        load_param_store(filename)
    if method == 'Delta':
        guide = AutoDelta(cond_model, init_loc_fn = init_to_sample)
    elif method == 'DiagonalNormal':
        guide = AutoDiagonalNormal(cond_model, init_loc_fn = init_to_sample)
    return guide


#def init_guide(model, filename = None, method = None):
#    # Load guide file, if provided, and check argument consistency
#    data = None
#    if filename is not None:
#        try:
#            data = np.load(filename)
#            if method is not None:
#                assert data['method'].item() == method
#            else:
#                method = data['method'].item()
#        except FileNotFoundError:
#            print("No guidefile not found. Initializing guide from scratch.")
#
#    # Instantiate guide
#    if method == "AutoDelta":
#        guide = AutoDelta(model)
#        # TODO: Setup initialization
#    elif method == "AutoDiagonalNormal":
#        guide = AutoDiagonalNormal(model)
#        if data is not None:
#            pyro.clear_param_store()
#            pyro.param("auto_scale", torch.tensor(data['auto_scale']), constraint=constraints.positive)
#            pyro.param("auto_loc", torch.tensor(data['auto_loc']))
#        else:
#            guide()
#            pyro.clear_param_store()
#            pyro.param("auto_loc", torch.zeros(guide.latent_dim))
#            pyro.param("auto_scale", torch.ones(guide.latent_dim)*1e-1, constraint=constraints.positive)
#    else:
#        raise KeyError("Invalid guide id.")
#
#    return guide

#def save_guide(guide, filename):
#    method = guide.__class__.__name__
#    data = {}
#    data['method'] = method
#
#    if method == 'AutoDelta':
#        for 'auto_'
#    elif method == 'AutoDiagonalNormal':
#        data['auto_loc'] = pyro.param('auto_loc').detach().cpu().numpy()
#        data['auto_scale'] = pyro.param('auto_scale').detach().cpu().numpy()
#    else:
#        raise NotImplementedError
#
#    np.savez(filename, **data)


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

    #loc = torch.zeros(10000)
    #scale = torch.ones(10000)
    #transforms = {"params.mu": AffineTransform(loc, scale)}
    transforms, initial_params = get_transforms_initial_params(args["quantfile"], cond_model)
    #transforms = None

    # Set up NUTS kernel
    nuts_kernel = NUTS(
        cond_model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        use_multinomial_sampling=True,
        jit_compile=False,
        max_tree_depth = 10,
        transforms = transforms,
        step_size = 1.)

    #initial_params = {name: torch.tensor(0.) for name in transforms.keys()}
    nuts_kernel.initial_params = initial_params

    # TODO: Update to use initial_params
#    # Must run model before get_init_values() since yaml parser is in the model
#    nuts_kernel.setup(warmup_steps=args["warmup_steps"])
#
#    # Set initial values
#    init_values = yaml_params.get_init_values()
#    initial_trace = nuts_kernel.initial_trace
#    sites = []
#    sites_vae = []
#    for key in init_values.keys():
#        print(key, init_values[key])
#        initial_trace.nodes[key]["value"] = init_values[key]
#        if init_values[key].numel() == 1:
#            sites.append(key)
#        else:
#            sites_vae.append(key)
#    nuts_kernel.initial_trace = initial_trace

    # Run
    posterior = MCMC(
        nuts_kernel,
        args["n_steps"],
        warmup_steps=args["warmup_steps"],
        num_chains=args["n_chains"]).run()

# FIXME: Fix chain export
#    # Move traces to CPU
#    posterior.exec_traces = [
#        trace_to_cpu(trace) for trace in posterior.exec_traces
#    ]
#    
#    # ._get_samepls_and_weights() does NOT work if the parameters (sites)
#    # have a different size. So, in case of the 'VAE' source class,
#    # the parameters as to be divided into two classes, consequently
#    # two differents 'sites' must be defided!
#
#    # Save posteriors
#    hmc_posterior = EmpiricalMarginal(
#        posterior, sites)._get_samples_and_weights()[0].detach().numpy()
#    if len(sites_vae) > 0:
#        hmc_posterior_vae = EmpiricalMarginal(
#            posterior, sites_vae)._get_samples_and_weights()[0].detach().numpy()
#    dict_posteriors = {}
#    for i, key in enumerate(sites):
#        dict_posteriors[key] = hmc_posterior[:, i]
#    for i, key in enumerate(sites_vae):
#        dict_posteriors[key] = hmc_posterior_vae[:,i]
#    save_posteriors(args, dict_posteriors)
#
#    # Export chain netcdf
#    data = arviz.from_pyro(posterior)
#    arviz.to_netcdf(data, args["fileroot"] + "_chain.nc")

def _infer_VI(cond_model, guide, guidefile, n_steps, n_write=10, quantfile = None):
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
    guide = init_guide(cond_model, guidefile, guide)
#
#        # Run YAML parsers attached to model to populate initial values
#        cond_model()
#
#        # Initialize AutoDelta parameters before running guide first time
#        init_values = yaml_params.get_init_values()
#        for key in init_values.keys():
#            pyro.param("auto_" + key, init_values[key])
#
#        guide()  # Initialize remaining guide parameters
#        print("Fitting parameters:", list(pyro.get_param_store()))
#
#        print("Initial values:")
#        for name, value in pyro.get_param_store().items():
#            print(name + ": " + str(value))
#        print()

    #guide = AutoDelta(cond_model)
    #guide = init_guide(cond_model, filename = args['guidefile'], method = "AutoDelta")
    #guide = AutoDiagonalNormal(cond_model)
    #guide = AutoLaplaceApproximation(cond_model)
    #pyro.clear_param_store()
    #pyro.clear_param_store()
    #guide()
    #auto_scale = pyro.param("auto_scale").detach()
    #pyro.clear_param_store()
    #pyro.param("auto_scale", auto_scale*1e-2)
    #pyro.param("auto_scale", torch.ones(1000), constraint=constraints.positive)
    #print pyro.param("auto_loc", torch.ones(latent_dim),
    #                       constraint=constraints.positive)
    #guide()
    #print(pyro.param("auto_loc"))
    #print(pyro.param("auto_scale"))
    #guide = AutoMultivariateNormal(cond_model)
    #print(guide())
    #quit()

    # Perform parameter fits
#    if args["opt"] == "ADAM":
    optimizer = Adam({"lr": 1e-2, "amsgrad": True})
    #elif args["opt"] == "SGD":
    #    optimizer = SGD({"lr": 1e-11, "momentum": 0.0})
    svi = SVI(cond_model, guide, optimizer, loss=Trace_ELBO())


    # Container for monitoring progress
#    infer_data = collections.defaultdict(list)

    for i in tqdm(range(n_steps)):
        if i % n_write == 0:
            for name, value in pyro.get_param_store().items():
                tqdm.write(name + ": " + str(value))

            #if args["resumefile"] is not None:
            if i > 0:
                #tqdm.write("Saving resume file: " + args["resumefile"])
                pyro.get_param_store().save(guidefile)
                    #save_param_steps(args, infer_data)
            tqdm.write("")

        # Save params step by step
#        for name, value in pyro.get_param_store().items():
#            infer_data[name].append(value.detach().clone().numpy())

        loss = svi.step()
        # Save losses step by step
#        infer_data["losses"].append(loss)
        if i % n_write == 0:
            tqdm.write("Loss: " + str(loss))

#    # Estimate parameter variance (Laplace approximation, based on diagonals only)
#    guide_trace = poutine.trace(guide).get_trace()
#    model_trace = poutine.trace(
#        poutine.replay(cond_model, trace=guide_trace)).get_trace()
#    loss = guide_trace.log_prob_sum() - model_trace.log_prob_sum()

#    
#    # TODO: Add option to use full laplacian instead
#    var_store = {}
#    for name in guide().keys():
#        print(name)
#        loc = pyro.param("auto_"+name)
#        try:
#            H = hessian(loss, loc.unconstrained())
#        except RuntimeError:
#            H = torch.ones(1)
#        var_store[name] = 1/torch.diag(H)


    pyro.get_param_store().save(guidefile)

    if quantfile is not None:
        try:
            data = guide.quantiles([0.16, 0.5, 0.84])
            data = {key: [val.detach().numpy() for val in vals] for key, vals in data.items()}
        except:
            data = {key: val.detach().numpy() for key, val in guide.median().items()}
        save_quantfile(quantfile, data)

    # Last save
#    if args["resumefile"] is not None:
#        save_param_store(args)
    #save_param_steps(args, infer_data)


# Strategy:
# - AutoDiagNormal --> MAP, and unconstrained loc and scale
# - transforms

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

###############
# Main PYROLENS
###############

#@click.command()
#@click.argument("command", required = True)
#@click.argument("yamlfile", required = True)
#@click.option(
#    "--n_steps", default=100, help="Number of steps (default 100).")
#@click.option(
#    "--fileroot",
#    default=None,
#    help="File root, default is yamlfile name without extension.")
#@click.option(
#    "--resumefile",
#    default=None,
#    help="Resume file name, default is derived from fileroot.")
#@click.option(
#    "--device",
#    default="cpu",
#    help="Set to 'cuda' to run on GPU (default 'cpu').")
#@click.option(
#    "--mode",
#    default="MAP",
#    help="Inference mode: MAP or NUTS (default 'MAP').")
#@click.option(
#    "--opt",
#    default="ADAM",
#    help="Optimization mode: ADAM or SGD (default 'ADAM')")
#@click.option(
#    "--n_chains",
#    default=1,
#    help="Number of chains for NUTS sampler (default 1).")
#@click.option(
#    "--warmup_steps",
#    default=50,
#    help="Warmup steps for NUTS sampler (default 50).")
#@click.option(
#    "--outyaml", default=None, help="Output updated yaml file when using MAP.")
#@click.option(
#    "--mockfile", default=None, help="Mock file name (*.npz).")
#@click.option(
#    "--guidefile", default=None, help="Optimization file name (*.npz).")
#@click.option(
#    "--quantfile", default=None, help="Quantifle file name (*.npz).")
#@click.option(
#    "--n_pixel", default=None, help="Pixel dimensions for mock image output (use 'nx,ny').")
#@click.option(
#    "--guide", default="Delta", help="Pixel dimensions for mock image output (use 'nx,ny').")
#@click.version_option(version=0.1)
def cli2(**kwargs):
    """This is PyroLens.

    COMMAND: mock ppd fit sample
    """
    set_default_filenames(kwargs)

    # Obtain YAML file
    with open(kwargs["yamlfile"], "r") as stream:
        config = yaml.load(stream)

    # Generate model
    module_name = config['pyrofit_module']
    my_module = importlib.import_module("pyrofit."+module_name)
    model = my_module.get_model(config, device=kwargs["device"])

    if kwargs["command"] == "mock":
        save_mock(model, filename = kwargs['mockfile'])
    elif kwargs["command"] == "ppd":
        cond_model = get_conditioned_model(config["conditioning"], model, device = kwargs["device"])
        guide = init_guide(cond_model, kwargs['guidefile'], method = kwargs['guide'])
        save_posterior_predictive(model, guide, kwargs['mockfile'])
    elif kwargs["command"] == "infer":
        # Try restoring param store from previous run
#        if resume:
#            load_param_store(kwargs)
        try:
            cond_model = get_conditioned_model(config["conditioning"], model, device = kwargs["device"])
            loss = infer(kwargs, config, cond_model)
            #config["data"]["loss"] = loss
        except KeyboardInterrupt:
            print("Interrupted.")

#        if resume:
#            save_param_store(kwargs)

        if kwargs["outyaml"] is not None:
            write_yaml(config, kwargs["outyaml"])
    else:
        print("COMMAND unknown.  Run with --help for more info.")

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
    ctx.obj['default_guidefile'] = yamlfile[:-5]+"_guide.npz"


@cli.command()
@click.option("--n_steps", default = 1000)
@click.option("--guide", default = "Delta")
@click.option("--guidefile", default = None)
@click.option("--quantfile", default = None)
@click.pass_context
def fit(ctx, n_steps, guide, guidefile, quantfile):
    """Parameter inference with variational methods."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    _infer_VI(cond_model, guide, guidefile, n_steps, quantfile)

@cli.command()
@click.option("--warmup_steps", default = 100)
@click.option("--n_steps", default = 300)
def sample():
    """Sample posterior with Hamiltonian Monte Carlo."""
    raise NotImplementedError

@cli.command()
@click.argument("mockfile")
@click.pass_context
def mock(ctx, mockfile):
    """Create mock data based on yaml file."""
    model = ctx.obj['model']
    save_mock(model, filename = mockfile)

@cli.command()
@click.option("--guide", default = "Delta")
@click.option("--guidefile", default = None)
@click.argument("ppdfile")
@click.pass_context
def ppd(ctx, guide, guidefile, ppdfile):
    """Sample from posterior predictive distribution."""
    if guidefile is None: guidefile = ctx.obj['default_guidefile']
    model = ctx.obj['model']
    device = ctx.obj['device']
    yaml_config = ctx.obj['yaml_config']
    cond_model = get_conditioned_model(yaml_config["conditioning"], model, device = device)
    guide = init_guide(cond_model, guidefile, method = guide)
    save_posterior_predictive(model, guide, ppdfile)
