"""
An example of a "factorized data" model where instead of having a function F generate the data,
we have a family of functions, each of which generates part of the data from the previous parts.
"""

import random
from copy import deepcopy

from LOTlib.Hypotheses.Lexicon.SimpleLexicon import SimpleLexicon
from LOTlib.Hypotheses.Likelihoods.StochasticFunctionLikelihood import StochasticFunctionLikelihood
from LOTlib.Hypotheses.RecursiveLOTHypothesis import RecursiveLOTHypothesis, RecursionDepthException
from LOTlib.Hypotheses.Proposers.RegenerationProposer import RegenerationProposer
from LOTlib.Hypotheses.Proposers.InsertDeleteProposer import InsertDeleteProposer
from LOTlib.Evaluation.EvaluationException import TooBigException


class InnerHypothesis(StochasticFunctionLikelihood, RecursiveLOTHypothesis, RegenerationProposer, InsertDeleteProposer):
    """
    The type of each function F.
    """
    def __init__(self, grammar=None, **kwargs):
        RecursiveLOTHypothesis.__init__(self, grammar=grammar, **kwargs)

    def __call__(self, *args):
        try:
            return RecursiveLOTHypothesis.__call__(self, *args)
        except RecursionDepthException:
            return ''

    def propose(self):
        if random.random() < 0.5:
            return RegenerationProposer.propose(self)
        else:
            return InsertDeleteProposer.propose(self)


class FactorizedDataHypothesis(SimpleLexicon):
    """
        An abstract class where we write the data as a composition of functions.

        self.__call__ calls using a compositional structure (which we may want to change in the future) of
        the ith function takes all the previous i outputs as arguments, and we return the last one.

        A SimpleLexicon associating each integer n with an InnerHypothesis. Each InnerHypothesis' grammar
        must be augmented with the arguments for the previous f_i

        This requires self.make_hypothesis to be defined and take a grammar argument.
    """
    def __init__(self, N=3, grammar=None, argument_type='LIST', variable_weight=2.0, value=None, **kwargs):

        SimpleLexicon.__init__(self, value=value)

        self.N = N

        if grammar is not None: # else we are in a copy initializer, and the rest will get copied
            for w in xrange(N):
                nthgrammar = deepcopy(grammar)

                # Add all the bound variables
                args = [  ]
                for xi in xrange(w):  # no first argument
                    argi = 'x%s'%xi

                    # Add a rule for the variable
                    nthgrammar.add_rule(argument_type, argi, None, variable_weight)

                    args.append(argi)

                # and add a rule for the n-ary recursion
                nthgrammar.add_rule('LIST', 'recurse_', [argument_type]*(w), 1.)

                self.set_word(w, self.make_hypothesis(grammar=nthgrammar, args=args))

    def __call__(self):
        # The call here must take no arguments. If this changes, alter x%si above
        theargs = []

        for w in xrange(self.N):
            try:
                v = self.get_word(w)(*theargs) # call with all prior args
                theargs.append(v)
            except TooBigException:
                theargs.append('')
            # print "V=", v, theargs

        return v # return the last one

    def make_hypothesis(self, **kwargs):
        raise NotImplementedError
