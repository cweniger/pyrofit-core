# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import itertools
import numpy as np
import torch
import pyro
import pyro.poutine as poutine
from pyro.infer.util import torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, warn_if_nan
from pyro.infer.importance import Importance

class ConLearn(Importance):
    """
    Contrastive Learning, allowing sequential compilation of inference network q(z|x) using atomic proposals.

    **Reference**
    FIXME

    :param model: probabilistic model defined as a function. Must accept a
        keyword argument named `observations`, in which observed values are
        passed as, with the names of nodes as the keys.
    :param guide: guide function which is used as an approximate posterior. Must
        also accept `observations` as keyword argument.
    :param optim: a Pyro optimizer
    :type optim: pyro.optim.PyroOptim
    :param num_inference_samples: The number of importance-weighted samples to
        draw during inference.
    :param training_batch_size: Number of samples to use to approximate the loss
        before each gradient descent step during training.
    :param validation_batch_size: Number of samples to use for calculating
        validation loss (will only be used if `.validation_loss` is called).
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 training_batch_size=4,
                 num_inference_samples=10,
                 validation_batch_size=20, site_names = None):
        super().__init__(model, guide, num_inference_samples)
        self.model = model
        self.guide = guide
        self.optim = optim
        self.observations = self._get_observations(model)
        self.training_batch_size = training_batch_size
        self.validation_batch_size = validation_batch_size
        self.validation_batch = None
        self.site_names = site_names
        self.simulations = []
        self.R = 0  # round counter

    def set_validation_batch(self, *args, **kwargs):
        """
        Samples a batch of model traces and stores it as an object property.

        Arguments are passed directly to model.
        """
        # TODO: Needs to be updated
        self.validation_batch = [self._sample_from_joint(0, *args, **kwargs)
                                 for _ in range(self.validation_batch_size)]

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function. Arguments are passed to the
        model and guide.
        """
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(True, None, *args, **kwargs)

        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values()
                     if site["value"].grad is not None)

        self.optim(params)

        pyro.infer.util.zero_grads(params)

        return torch_item(loss)

    def simulate(self, n_simulations, *args, replace = False, **kwargs):
        """Simulate more training data: x, z ~ P(x|z) Q(z|x_0)"""
        new_simulations = [self._sample_from_joint(self.R, *args, **kwargs)
                    for _ in range(n_simulations)]
        self.R += 1
        if replace:
          self.simulations = new_simulations
        else:
          self.simulations += new_simulations

    def loss_and_grads(self, grads, batch, *args, **kwargs):
        """
        :returns: an estimate of the loss (expectation over p(x, y) of
            -log q(x, y) ) - where p is the model and q is the guide
        :rtype: float

        If a batch is provided, the loss is estimated using these traces
        Otherwise, a fresh batch is generated from the model.

        If grads is True, will also call `backward` on loss.

        `args` and `kwargs` are passed to the model and guide.
        """
        if batch is None:
            indices = np.random.choice(len(self.simulations),
                                       size = self.training_batch_size,
                                       replace = False)
            batch = [self.simulations[i] for i in indices]
            batch_size = self.training_batch_size
        else:
            batch_size = len(batch)

        # Collect all cross matched guide traces
        with poutine.trace(param_only=True) as particle_param_capture:
            guide_traces = []

            for i in range(batch_size):
                # model_x: True model against which we contrast the rest
                model_x_trace = batch[i]

                guide_traces.append([])

                for j in range(batch_size):
                    # model_z: Contrasting model parameters
                    model_z_trace = batch[j]
    
                    # Evaluate matched guide
                    guide_trace = self._get_matched_cross_trace(
                        model_x_trace, model_z_trace, *args, **kwargs)  

                    guide_traces[-1].append(guide_trace)

        loss = torch.tensor(0.)
        
        # Calculate losses per site
        for site_name in self.site_names:
            for i in range(batch_size):
                model_x_trace = batch[i]  

                log_prob_priors = []
                for j in range(batch_size):
                    model_z_trace = batch[j]
                    log_prob_prior = (
                        model_x_trace.nodes[site_name]['fn'].log_prob(
                        model_z_trace.nodes[site_name]['value']))
                    log_prob_priors.append(log_prob_prior.unsqueeze(0))
                log_prob_priors = torch.cat(log_prob_priors, 0)

                guide_losses = torch.cat(
                    [self._differentiable_loss_particle(
                        guide_trace, site_name = site_name).unsqueeze(0)
                        for guide_trace in guide_traces[i]], 0)

                f_phis = guide_losses + log_prob_priors
                r = -torch.log_softmax(-f_phis, 0)
                particle_loss = r[i].sum()/batch_size

                loss += particle_loss

        warn_if_nan(loss, "loss")

        if grads:
            guide_params = set(site["value"].unconstrained()
                            for site in particle_param_capture.trace.nodes.values())
            guide_params = list(guide_params)
            torch.autograd.set_detect_anomaly(True)
            guide_grads = torch.autograd.grad(loss, guide_params, allow_unused=True, retain_graph=True)
            for guide_grad, guide_param in zip(guide_grads, guide_params):
                if guide_param.grad is None:
                    guide_param.grad = guide_grad
                else:
                    if guide_grad is not None:
                        guide_param.grad =  guide_param.grad + guide_grad 

        return torch_item(loss)

    def _differentiable_loss_particle(self, guide_trace, site_filter = lambda name, site: True, site_name = None):
        if site_name is None:
            return -guide_trace.log_prob_sum(site_filter = site_filter)
        else:
            guide_trace.compute_log_prob(site_filter = lambda name, site: name == site_name)
            log_prob = guide_trace.nodes[site_name]['log_prob']
            print(log_prob)
            return -log_prob

    def validation_loss(self, *args, **kwargs):
        """
        :returns: loss estimated using validation batch
        :rtype: float

        Calculates loss on validation batch. If no validation batch is set,
        will set one by calling `set_validation_batch`. Can be used to track
        the loss in a less noisy way during training.

        Arguments are passed to the model and guide.
        """
        if self.validation_batch is None:
            self.set_validation_batch(*args, **kwargs)

        return self.loss_and_grads(False, self.validation_batch, *args, **kwargs)

    def _get_matched_cross_trace(self, model_x_trace, model_z_trace,*args, **kwargs):
        kwargs["observations"] = {}
        kwargs["truth"] = {}
        for node in itertools.chain(model_x_trace.stochastic_nodes, model_x_trace.observation_nodes):
            if "was_observed" in model_x_trace.nodes[node]["infer"]:
                model_x_trace.nodes[node]["is_observed"] = True
                model_z_trace.nodes[node]["is_observed"] = True
                kwargs["observations"][node] = model_x_trace.nodes[node]["value"]
            else:
                kwargs["truth"][node] = model_x_trace.nodes[node]["value"]

        guide_trace = poutine.trace(poutine.replay(self.guide,
                                                   model_z_trace)
                                    ).get_trace(*args, **kwargs)

        check_model_guide_match(model_x_trace, guide_trace)
        check_model_guide_match(model_z_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace


    def _get_matched_trace(self, model_trace, *args, **kwargs):
        """
        :param model_trace: a trace from the model
        :type model_trace: pyro.poutine.trace_struct.Trace
        :returns: guide trace with sampled values matched to model_trace
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a guide trace with values at sample and observe statements
        matched to those in model_trace.

        `args` and `kwargs` are passed to the guide.
        """
        kwargs["observations"] = {}
        for node in itertools.chain(model_trace.stochastic_nodes, model_trace.observation_nodes):
            if "was_observed" in model_trace.nodes[node]["infer"]:
                model_trace.nodes[node]["is_observed"] = True
                kwargs["observations"][node] = model_trace.nodes[node]["value"]
            else:
                model_trace.nodes[node][""]

        guide_trace = poutine.trace(poutine.replay(self.guide,
                                                   model_trace)
                                    ).get_trace(*args, **kwargs)

        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace

    def _sample_from_joint(self, R, *args, **kwargs):
        """
        :returns: a sample from the joint distribution over unobserved and
            observed variables
        :rtype: pyro.poutine.trace_struct.Trace

        Returns a trace of the model without conditioning on any observations.

        Arguments are passed directly to the model.
        """
        if R == 0:  # first round, sample from original model
            unconditioned_model = pyro.poutine.uncondition(self.model)
            return poutine.trace(unconditioned_model).get_trace(*args, **kwargs)
        else:
            kwargs_guide = kwargs.copy()
            kwargs_guide['observations'] = self.observations
            with pyro.poutine.block(expose = []):
                with pyro.poutine.trace() as guide_trace_msg:
                    self.guide(*args, **kwargs_guide)
            guide_trace = guide_trace_msg.get_trace()
    
            unconditioned_model = pyro.poutine.uncondition(self.model)
            
            with pyro.poutine.trace() as model_trace_msg:
                pyro.poutine.replay(unconditioned_model, trace = guide_trace)(*args, **kwargs)
            trace = model_trace_msg.get_trace()
            trace.detach_()
            return trace


    def _get_observations(self, cond_model):
        trace = pyro.poutine.trace(cond_model).get_trace()
        observations = {}
        for name in trace.observation_nodes:
            value = trace.nodes[name]['value']
            observations[name] = value
        return observations
