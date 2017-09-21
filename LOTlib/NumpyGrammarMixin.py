from LOTlib.Miscellaneous import *
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import FunctionNode, BVAddFunctionNode

import numpy as np

class NumpyGrammarMixin(object):
    def update_alphas(self):
        n = len(self.nonterminals())
        m = max([len(rhs) for rhs in self.rules.values()])

        self.alphas = np.zeros((n, m))
        self.sigmas = np.ones(np.shape(self.alphas))        # we're doing this to allow us to
                                                            # take the log of this array without issue
        self.sorted_rules = sorted(self.rules.keys())

        for ix, key in enumerate(self.sorted_rules):
            self.alphas[ix, :len(self.rules[key])] = 1.0
            ps = np.asarray([x.p for x in self.rules[key]])
            self.sigmas[ix, :len(self.rules[key])] = ps/sum(ps)

        # we're going to assume here that our rule and symbol indexing works
        # correctly and we never end up trying to get the log(sigma) of a
        # value that doesnt actually exist
        self.log_sigmas = np.log(self.sigmas)

    def generate_with_counts(self, x=None, nc=None):
        if x is None:
            x = self.start

            assert self.start in self.nonterminals(), \
                "The default start symbol %s is not a defined nonterminal" % self.start

            if nc is None:
                nc = np.zeros(np.shape(self.alphas))
            x = self.generate_with_counts(x, nc)

            return x, nc

        try:
            if isinstance(x, list):
                return map(lambda xi: self.generate_with_counts(x=xi, nc=nc), x)
            elif self.is_nonterminal(x):
                rules = self.get_rules(x)
                assert len(rules) > 0, "*** No rules in x=%s"%x

                r = weighted_sample(rules, probs=lambda x: x.p, log=False)

                rule_ix = self.sorted_rules.index(x)
                sym_ix = rules.index(r)
                nc[rule_ix, sym_ix] += 1.0

                fn = r.make_FunctionNodeStub(self, None)

                with BVRuleContextManager(self, fn, recurse_up=False):
                    if fn.args is not None:
                        try:
                            fn.args = self.generate_with_counts(fn.args, nc)
                        except RuntimeError as e:
                            print "*** Runtime error in %s" %fn
                            raise e

                    for a in fn.argFunctionNodes():
                        a.parent = fn

                return fn
            elif isinstance(x, FunctionNode):
                x.args = [self.generate_with_counts(a, nc) for a in x.args]

                for a in x.argFunctionNodes():
                    a.parent = x

                return x
            else:
                assert isinstance(x, str), ("*** Terminal must be a string! x="+x)
                return x
        except AssertionError as e:
            print "***Assertion error in %s" % x
            raise e