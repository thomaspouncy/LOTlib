# -*- coding: utf-8 -*-

from LOTlib.Miscellaneous import q, qq, Infinity, self_update
from LOTlib.Inference.Samplers.Sampler import Sampler, MH_acceptance

from math import log, exp
from random import random


class NumpyMH(Sampler):
    """A class to implement MH sampling.

    You can create a sampler object::
        from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
        data = generate_data(500)
        h0 = NumberExpression(grammar)
        sampler = MHSampler(h0, data, 10000)
        for h in sampler:
            print sampler.acceptance_ratio(), h

    Or implicitly::
        from LOTlib.Examples.Number.Shared import generate_data, NumberExpression, grammar
        data = generate_data(500)
        h0 = NumberExpression(grammar)
        for h in  MHSampler(h0, data, 10000):
            print h

    Note
    ----
    A wrapper below called mh_sample is provided to maintain backward compatibility. But mh_sample
    will be removed in the future.

    Returns
    -------
    If a proposer is specific in __init__, it should return a *new copy* of the object


    Parameters
    ----------
    current_sample : doc?
        if None, we don't compute it's posterior (via set_state); otherwise we do.
    data : doc?
        doc?
    steps : int
        Number of steps to generate before stopping.
    proposer : function
        Defaultly this calls the sample Hypothesis's propose() function.
    skip : int
        Throw out this many samples each time MHSampler yields a sample.
    prior_temperature : float
        How much do we weight our prior relative to likelihood?
    likelihood_temperature : float
        How much do we weight our likelihood relative to prior?
    acceptance_temperature : float
        This weights the probability of accepting proposals.
    trace : bool
        If true, print stuff as we sample.
    shortcut_likelihood : bool
        If true, we allow for short-cut evaluation of the likelihood, rejecting when we can if the ll
        drops below the acceptance value

    Attributes
    ----------
    was_accepted : bool
        Was the last proposal accepted?
    samples_yielded : int
        How many samples have I yielded? This doesn't count skipped samples.


    """

    def __init__(self, current_sample, data, steps=Infinity, proposer=None, skip=0,
                 prior_temperature=1.0, likelihood_temperature=1.0, acceptance_temperature=1.0, trace=False,
                 shortcut_likelihood=True):
        self_update(self, locals())
        self.was_accepted = None

        if proposer is None:
            self.proposer = lambda x: x.propose()

        self.samples_yielded = 0
        self.set_state(current_sample, compute_posterior=(current_sample is not None))
        self.reset_counters()

    def at_temperature(self, t, whichtemperature):
        """Set temperature for prior, likelihood, or acceptance.

        Args
        ----
        t : float
            Set temperature to this value.
        whichtemperature : {'prior_temperature', 'likelihood_temperature', 'acceptance_temperature'}
            Which temperature?

        """
        pt, lt, at = self.prior_temperature, self.likelihood_temperature, self.acceptance_temperature
        if whichtemperature == 'prior_temperature':
            pt = t
        elif whichtemperature == 'likelihood_temperature':
            lt = t
        elif whichtemperature == 'acceptance_temperature':
            at = t

        return (self.current_sample.prior / pt + self.current_sample.likelihood / lt) / at

    def reset_counters(self):
        """
        Reset acceptance and proposal counters.

        """
        self.acceptance_count = 0
        self.proposal_count = 0
        self.posterior_calls = 0

    def acceptance_ratio(self):
        """
        Returns the proportion of proposals that have been accepted.

        """
        if self.proposal_count > 0:
            return float(self.acceptance_count) / float(self.proposal_count)
        else:
            return float("nan")

    def next(self):
        """Generate another sample."""
        if self.samples_yielded >= self.steps:
            raise StopIteration
        else:
            for _ in xrange(self.skip + 1):

                self.proposal, prop_probs = self.proposer(self.current_sample)

                assert self.proposal is not self.current_sample, "*** Proposal cannot be the same as the current sample!"
                assert self.proposal.value is not self.current_sample.value, "*** Proposal cannot be the same as the current sample!"

                # Call myself so memoized subclasses can override
                self.compute_posterior(self.proposal, self.data)

                # if im not mistaken, under a PCFGPrior, these variables
                # are already weighted by their respective temperatures, so
                # reweighting them will result in them being weighted twice
                cur = self.current_sample.prior + self.current_sample.likelihood
                prop = self.proposal.prior + self.proposal.likelihood

                if MH_acceptance(cur, prop, prop_probs, acceptance_temperature=self.acceptance_temperature): # this was the old form
                    self.current_sample = self.proposal
                    self.was_accepted = True
                    self.acceptance_count += 1
                else:
                    self.was_accepted = False

                self.proposal_count += 1

            self.samples_yielded += 1
            return self.current_sample

