"""
    Regeneration Proposer - choose a node of type X and replace it with
    a newly sampled value of type X.

    NOTE: ERGODIC!
"""

from LOTlib.Hypotheses.Proposers.Proposer import *
from LOTlib.Miscellaneous import lambdaOne, nicelog
from LOTlib.FunctionNode import isFunctionNode
from random import random
from copy import copy
import numpy as np

class NumpyRegenerationProposer(Proposer):
    def proposal_content(self, grammar, hypothesis, resampleProbability=lambdaOne):
        t = self.propose_tree(grammar, hypothesis, resampleProbability)
        fb = self.compute_fb(grammar, hypothesis, t, resampleProbability)
        return t, fb

    def propose_tree(self, grammar, h, resampleProbability=lambdaOne):
        """Propose, returning the new tree"""

        # 1) sample subnode
        # 2) create a new value by:
        #   a) starting from root, create new nodes with same properties until
        #      selected subnode is reached, then generate new nodes
        # 3) create a new hypothesis with this value and node counts
        # 4) return the new hypothesis

        # NOTE: we are getting rid of resampleProbs here to save on iterations through the tree

        node_prob = 1.0 / float(h.tree_size)
        r = random()

        value, node_counts, r = self.copy_and_permute_tree(grammar, r, node_prob, h.value, np.zeros(np.shape(grammar.alphas)))

        assert r <= 0, "***Failed to select a node for proposal"
        return h.__copy__(value=value, node_counts=node_counts)

    def copy_and_permute_tree(self, grammar, r, node_prob, curr_node, nc):
        new_r = r - node_prob
        rs = curr_node.get_rule_signature()

        if r > 0 and new_r <= 0:
            # print "generating new subtree from: {}".format(rs)
            new_node = grammar.generate_with_counts(x=rs[0], nc=nc)
            # print "\tnew node is: {}".format(new_node.get_rule_signature())

            return new_node, nc, new_r
        else:
            new_node = curr_node.__copy__(shallow=True)            # leaves children pointing to old nodes
            rule_ix = grammar.sorted_rules.index(rs[0])
            sym_ix = next(i for i, x in enumerate(grammar.rules[rs[0]]) if x.get_rule_signature() == rs)
            nc[rule_ix, sym_ix] += 1.0

            if curr_node.args is not None:
                new_node.args = []
                for i, a in enumerate(curr_node.args):
                    if isFunctionNode(a):
                        new_child, nc, new_r = self.copy_and_permute_tree(grammar, new_r, node_prob, a, nc)
                        new_node.args.append(new_child)
                        new_child.parent = new_node
                    else:
                        # print "non function node: {}".format(a)
                        new_node.args.append(copy(a))

            return new_node, nc, new_r

    def compute_proposal_probability(self, grammar, h1, h2, resampleProbability=lambdaOne, recurse=True):
        """Compute probability of proposing t2 given t1 and resampleProb"""

        # we can just use the probability of the whole tree here because
        # the probability of the parent nodes will cancel out for the
        # nodes above the sampled point

        # NOTE: we are getting rid of the ability to have differential resampleProbs
        # here to save on another parse through each tree.

        return h1.prior - nicelog(h2.tree_size)

if __name__ == "__main__":  # test code
    test_proposer(NumpyRegenerationProposer)
