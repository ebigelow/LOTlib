"""

TODO: We need to add replicatingrules and apply for the other proposal methods to be tested!

"""


from collections import Counter
from math import exp
from scipy.stats import chisquare

from TreeTesters import FiniteTreeTester # defines check_tree and setUp
from LOTlib.Hypotheses.Proposers import ProposalFailedException

NSAMPLES = 1000

class ProposalTest(FiniteTreeTester):
    """
    This tests if proposals return well-formed trees.
    """

    def test_RegenerationProposal(self):
        from LOTlib.Hypotheses.Proposers.RegenerationProposer import RegenerationProposer
        rp = RegenerationProposer(grammar=self.grammar)

        for tree in self.trees:
            cnt = Counter()
            for _ in xrange(NSAMPLES):
                p, fb = rp.propose_tree(tree)
                cnt[p] += 1

                # Check the proposal
                self.check_tree(p)

            ## check that the proposals are what they should be -- rp.lp_propose is correct!
            obsc = [cnt[t] for t in self.trees]
            expc = [exp(self.grammar.log_probability(t))*sum(obsc) for t in self.trees]
            csq, pv = chisquare([cnt[t] for t in self.trees],
                                [exp(rp.lp_propose(tree, x))*NSAMPLES for x in self.trees])

            # Look at some
            # print ">>>>>>>>>>>", tree
            # for p in self.trees:
            #     print "||||||||||", p
            #     v = rp.lp_propose(tree,p)
            #     print "V=",v

            # for c, e, tt in zip([cnt[t] for t in self.trees],
            #                    [exp(rp.lp_propose(tree, x))*NSAMPLES for x in self.trees],
            #                    self.trees):
            #     print c, e, tt, rp.lp_propose(tree,tt)

            self.assertGreater(pv, 0.001, msg="Sampler failed chi squared!")

    def test_InsertDeleteProposal(self):
        from LOTlib.Hypotheses.Proposers.InsertDeleteProposer import InsertDeleteProposer
        rp = InsertDeleteProposer(grammar=self.grammar)

        for tree in self.trees:
            for _ in xrange(100):
                try:
                    p, fb = rp.propose_tree(tree)
                    self.check_tree(p)
                except ProposalFailedException:
                    pass

if __name__ == '__main__':

    import unittest

    unittest.main()
