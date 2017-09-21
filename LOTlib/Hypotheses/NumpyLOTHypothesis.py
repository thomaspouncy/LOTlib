
from LOTlib.Eval import * # Necessary for compile_function eval below
from LOTlib.Hypotheses.Proposers import np_regeneration_proposal, ProposalFailedException
from LOTlib.Miscellaneous import self_update
from LOTlib.Hypotheses.FunctionHypothesis import FunctionHypothesis
from LOTlib.Miscellaneous import attrmem
from LOTlib.Primitives import *

from copy import deepcopy

import numpy as np

class NumpyLOTHypothesis(FunctionHypothesis):
    """
    A LOTHypothesis that permits recursive calls to itself via the primitive "recurse" (previously, L).

    Here, RecursiveLOTHypothesis.__call__ does essentially the same thing as LOTHypothesis.__call__, but it binds
    the symbol "recurse" to RecursiveLOTHypothesis.recursive_call so that recursion is processed internally.

    This bind is done in compile_function, NOT in __call__

    For a Demo, see LOTlib.Examples.Number
    """

    def __init__(self, grammar, value=None, f=None, node_counts=None, maxnodes=25, recurse_bound=25, display="lambda recurse_, x: %s", **kwargs):
        """
        Initializer. recurse gives the name for the recursion operation internally.
        """
        assert "lambda recurse_" in display, "*** RecursiveLOTHypothesis must have 'recurse_' as first display element." # otherwise it can't eval

        # save recurse symbol
        self.recursive_depth_bound = recurse_bound # how deep can we recurse?
        self.recursive_call_depth = 0 # how far down have we recursed?

        if 'args' in kwargs:
            assert False, "*** Use of 'args' is deprecated. Use display='...' instead."

        # Save all of our keywords
        self_update(self, locals())
        grammar.update_alphas()         # make sure alpha/sigma arrays are initialized
        if value is None:
            value, self.node_counts = grammar.generate_with_counts()
        else:
            self.node_counts = node_counts
        self.tree_size = np.sum(self.node_counts)

        FunctionHypothesis.__init__(self, value=value, f=f, display=display, **kwargs)

        self.likelihood = 0.0
        self.compute_prior()
        self.rules_vector = None

    def __copy__(self, value=None, node_counts=None):
        """Returns a copy of the Hypothesis. Allows you to pass in value to set to that instead of a copy."""

        thecopy = type(self)()  # Empty initializer

        # copy over all the relevant attributes and things.
        # Note objects like Grammar are not copied
        thecopy.__dict__.update(self.__dict__)

        # and then we need to explicitly *deepcopy* the value (in case its a dict or tree, or whatever)
        if value is None:
            value = deepcopy(self.value)

        if node_counts is not None:
            thecopy.node_counts = node_counts
            thecopy.tree_size = np.sum(node_counts)

        thecopy.set_value(value)

        thecopy.compute_prior()

        return thecopy

    def recursive_call(self, *args):
        """
        This gets called internally on recursive calls. It keeps track of the depth and throws an error if you go too deep
        """

        self.recursive_call_depth += 1

        if self.recursive_call_depth > self.recursive_depth_bound:
            raise RecursionDepthException

        # Call with sending myself as the recursive call
        return FunctionHypothesis.__call__(self, self.recursive_call, *args)

    def __call__(self, *args):
        """
        The main calling function. Resets recursive_call_depth and then calls
        """
        self.recursive_call_depth = 0

        try:
            # call with passing self.recursive_Call as the recursive call
            return FunctionHypothesis.__call__(self, self.recursive_call, *args)
        except TypeError as e:
            print "TypeError in function call: ", e, str(self), "  ;  ", type(self), args
            raise TypeError
        except NameError as e:
            print "NameError in function call: ", e, " ; ", str(self), args
            raise NameError

    def type(self):
        return self.value.type()

    def compile_function(self):
        """Called in set_value to compile into a function."""
        if self.value.count_nodes() > self.maxnodes:
            return lambda *args: raise_exception(TooBigException)
        else:
            try:
                return eval(str(self)) # evaluate_expression(str(self))
            except Exception as e:
                print "# Warning: failed to execute evaluate_expression on " + str(self)
                print "# ", e
                return lambda *args: raise_exception(EvaluationException)

    def propose(self, **kwargs):
        proposal, fb = None, None
        while True: # keep trying to propose
            try:
                proposal, fb = np_regeneration_proposal(self.grammar, self, **kwargs)
                break
            except ProposalFailedException:
                pass

        return proposal, fb

    @attrmem('prior')
    def compute_prior(self):
        if self.tree_size > getattr(self, 'maxnodes', Infinity):
            return -Infinity
        else:
            # NOTE: if we are inferring the sigmas as well, then we should compute this
            # prior by marginalizing over the sigmas under the hyperprior
            # basically, replace the following with:
            # sum(mvbeta(x.node_counts + self.grammar.alphas)),
            # where mv_beta(alphas):
            #   filtered = alphas[alphas > 0]
            #   return sum(map(gammaln, filtered)) - gammaln(sum(filtered))
            return np.sum(self.grammar.log_sigmas*self.node_counts) / self.prior_temperature

    @attrmem('posterior_score')
    def compute_posterior(self, d, **kwargs):
        """Computes the posterior score by computing the prior and likelihood scores.
        Defaultly if the prior is -inf, we don't compute the likelihood (and "pretend" it's -Infinity).
        This saves us from computing likelihoods on hypotheses that we know are bad.
        """

        p = self.prior  # this is calculated on initialization for use in the proposal,
                        # so we shouldn't need to recalculate it here

        if p > -Infinity:
            l = self.compute_likelihood(d, **kwargs)
            return p + l
        else:
            self.likelhood = None  # We haven't computed this
            return -Infinity