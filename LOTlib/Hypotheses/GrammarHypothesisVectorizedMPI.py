"""
Vectorized GrammarHypothesis class.

This assumes:
    - domain hypothesis (e.g. NumberGameHypothesis) evaluates to a set
    - `data` is a list of HumanData objects
    - HumanData.data can be applied to hypothesis.compute_likelihood (for domain-level hypothesis)
    - HumanData.responses is a list of (y, n) pairs,  where y is the number of positive and n is the number
      of negative responses for the given output key, conditioned on the input

To use this for a different format, change init_R & compute_likelihood.

"""



# same for all GrammarHypotheses
# ------------------------------
#
# C = get rule counts for each grammar rule, for each hypothesis    |hypotheses| x |rules|
# for each FunctionData:
# Li = for FuncData i, for ea hypothesis, get likelihood of i.input in concept   |hypotheses| x 1
# Ri = for FuncData i, for ea hypothesis, is each i.output in the concept (1/0)  |hypotheses| x |output|


# compute_likelihood
# ------------------
#
# x = get rule probabilities for each rule    1 x |rules|
# P = x * C     |hypotheses| x 1
#
# for each FunctionData i:
#   v = Li + P      1 x |hypotheses|
#   Z = logsumexp(v)
#   v = exp(v-Z)        # weighted probability of each hypothesis given input data
#   p_in_concept = rowsum(v * Ri_j) for Ri_j in Ri   # i.e. multiply ea. col in Ri by v


import copy
from math import exp, log
import numpy as np
from LOTlib.GrammarRule import BVUseGrammarRule
from LOTlib.Hypotheses.GrammarHypothesis import GrammarHypothesis
from LOTlib.Miscellaneous import logsumexp, gammaln, log1mexp
from LOTlib.MPI.MPI_map import MPI_unorderedmap


class GrammarHypothesisVectorizedMPI(GrammarHypothesis):

    def __init__(self, grammar, hypotheses, H=None, C=None, L=None, R=None, **kwargs):
        GrammarHypothesis.__init__(self, grammar, hypotheses, **kwargs)
        if H is None:
            self.init_H()
        else:
            self.H = H
        if H is None:
            self.init_C()
        else:
            self.C = C
        self.L = L if L else {}
        self.R = R if R else {}

    def init_C(self):
        """
        Initialize our rule count vector `self.C`.

        """
        self.C = np.zeros((len(self.hypotheses), len(self.rules)))
        rule_idxs = {str([r.name, r.nt, r.to]): i for i, r in enumerate(self.rules)}

        for j, h in enumerate(self.hypotheses):
            grammar_rules = [self.grammar.get_matching_rule(fn) for fn in h.value.iterate_subnodes(self.grammar)]
            for rule in grammar_rules:
                try:
                    self.C[j, rule_idxs[str([rule.name, rule.nt, rule.to])]] += 1
                except Exception as e:
                    if isinstance(rule, BVUseGrammarRule):
                        pass
                    else:
                        print str(h)
                        raise e

    def init_H(self):
        """Initialize hypothesis concept list `self.H`."""
        self.H = [h() for h in self.hypotheses]

    def init_L(self, d, d_index):
        """Initialize `self.L` dictionary."""
        self.L[d_index] = np.array([h.compute_likelihood(d.data) for h in self.hypotheses])

    def init_R(self, d, d_index):
        """Initialize `self.R` dictionary."""
        self.R[d_index] = np.zeros((len(self.hypotheses), d.q_n))

        for q, r, m in d.get_queries():
            self.R[d_index][:, m] = [int(q in h_concept) for h_concept in self.H]

    def compute_likelihood(self, data, update_post=True, **kwargs):
        """
        Compute the likelihood of producing human data, given:  H (self.hypotheses)  &  x (self.value)

        """
        # Initialize unfilled values for L[data] & R[data]
        for d_index, d in enumerate(data):
            if d_index not in self.L:
                self.init_L(d, d_index)
            if d_index not in self.R:
                self.init_R(d, d_index)

        x = self.normalized_value()         # vector of rule probabilites
        P = np.dot(self.C, x)               # prior for each hypothesis

        # Compute each likelihood; "wow very parallel such MPI wow"
        likelihood = sum(MPI_unorderedmap(self.compute_single_likelihood_MPI,
                                          [(d_index, d, P) for d_index, d in enumerate(data)]))
        if update_post:
            self.likelihood = likelihood
            self.update_posterior()
        return likelihood

    def compute_single_likelihood_MPI(self, input_args):
        d_index, d, P = input_args
        posteriors = self.L[d_index] + P
        Z = logsumexp(posteriors)
        w = np.exp(posteriors - Z)              # weights for each hypothesis
        r_i = np.transpose(self.R[d_index])
        w_times_R = w * r_i

        likelihood = 0.0

        # Compute likelihood of producing same output (yes/no) as data
        for q, r, m in d.get_queries():
            # col `m` of boolean matrix `R[i]` weighted by `w`
            query_col = w_times_R[m, :]
            exp_p = query_col.sum()
            p = log(exp_p)
            ## p = log((np.exp(w) * self.R[d_index][:, m]).sum())

            # NOTE: with really small grammars sometimes we get p > 0
            if p >= 0:
                print 'P ERROR!'

            yes, no = r
            k = yes             # num. yes responses
            n = yes + no        # num. trials
            bc = gammaln(n+1) - (gammaln(k+1) + gammaln(n-k+1))     # binomial coefficient
            l1mp = log1mexp(p)
            likelihood += bc + (k*p) + (n-k)*l1mp                   # likelihood we got human output

    def normalized_value(self):
        """Return a rule probabilities, each normalized relative to other rules with same nt.

        Note
        ----
        This is the only time where we need to call `self.update()`, since this is the
        only time where we reference `self.rules`.

        """
        # self.update()

        # Make dictionary of normalization constants for each nonterminal
        nt_Z = {}
        for nt in self.grammar.nonterminals():
            Z = sum([self.value[i] for i in self.get_rules(rule_nt=nt)[0]])
            nt_Z[nt] = Z

        # Normalize each probability in `self.value`
        normalized = np.zeros(len(self.rules))
        for i, r in enumerate(self.rules):
            normalized[i] = self.value[i] / nt_Z[self.rules[i].nt]

        return np.log(normalized)

    def update(self):
        """Update `self.rules` relative to `self.value`."""
        # Set probability for each rule corresponding to value index
        for i in range(0, self.n):
            self.rules[i].p = self.value[i]
