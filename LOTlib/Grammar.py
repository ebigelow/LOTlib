# *- coding: utf-8 -*-
try: import numpy as np
except ImportError: import numpypy as np

from copy import copy
from collections import defaultdict
import itertools

from LOTlib import break_ctrlc
from LOTlib.Miscellaneous import *
from LOTlib.GrammarRule import GrammarRule, BVAddGrammarRule
from LOTlib.BVRuleContextManager import BVRuleContextManager
from LOTlib.FunctionNode import FunctionNode

class Grammar:
    """
    A PCFG-ish class that can handle rules that introduce bound variables
    """
    def __init__(self, BV_P=10.0, start='START'):
        self.__dict__.update(locals())
        self.rules = defaultdict(list)  # A dict from nonterminals to lists of GrammarRules.
        self.rule_count = 0
        self.bv_count = 0   # How many rules in the grammar introduce bound variables?

    def __str__(self):
        """Display a grammar."""
        return '\n'.join([str(r) for r in itertools.chain(*[self.rules[nt] for nt in self.rules.keys()])])

    def nrules(self):
        """ Total number of rules
        """
        return sum([len(self.rules[nt]) for nt in self.rules.keys()])

    def get_rules(self, nt):
        """
        The possible rules for any nonterminal
        """
        return self.rules[nt]

    def is_nonterminal(self, x):
        """A nonterminal is just something that is a key for self.rules"""
        # if x is a string  &&  if x is a key
        return isinstance(x, str) and (x in self.rules)

    def display_rules(self):
        """Prints all the rules to the console."""
        for rule in self:
            print rule

    def __iter__(self):
        """Define an iterator over all rules so we can say 'for rule in grammar...'."""
        for k in self.rules.keys():
            for r in self.rules[k]:
                yield r

    def nonterminals(self):
        """Returns all non-terminals."""
        return self.rules.keys()

    def get_matching_rule(self, t):
        """
        Get the rule matching t's signature. Note: We could probably speed this up with a hash table if need be.
        """
        rules = self.get_rules(t.returntype)
        matching_rules = [r for r in rules if (r.get_rule_signature() == t.get_rule_signature())]
        assert len(matching_rules) == 1, \
            "Grammar Error: " + str(len(matching_rules)) + " matching rules for this FunctionNode!"
        return matching_rules[0]

    def log_probability(self, t):
        """
        Returns the log probability of t, recomputing everything (as we do now)

        This is overall about half as fast, but it means we don't have to store generation_probability
        """
        assert isinstance(t, FunctionNode)

        z = log(sum([ r.p for r in self.get_rules(t.returntype) ]))

        # Find the one that matches. While it may seem like we should store this, that is hard to make work
        # with multiple grammar objects across loading/saving, because the objects will change. This way,
        # we always look it up.
        lp = -Infinity
        r = self.get_matching_rule(t)
        assert r is not None, "Failed to find matching rule at %s %s" % (t, r)

        lp = log(r.p) - z

        with BVRuleContextManager(self, t):
            for a in t.argFunctionNodes():
                lp += self.log_probability(a)

        return lp

    def add_rule(self, nt, name, to, p, bv_type=None, bv_args=None, bv_prefix='y', bv_p=None):
        """Adds a rule and returns the added rule.

        Arguments
            nt (str): The Nonterminal. e.g. S in "S -> NP VP"
            name (str): The name of this function. NOTE: If you are introducing a bound variable,
              the name of this function must reflect that it is a lambda node! Currently, the only way to
              do this is to name it 'lambda'.
            to (list<str>): What you expand to (usually a FunctionNode).
            p (float): Unnormalized probability of expansion
            bv_type (str): What bound variable was introduced
            bv_args (list): What are the args when we use a bv (None is terminals, else a type signature)

        """
        self.rule_count += 1
        assert name is not None, "To use null names, use an empty string ('') as the name."
        if bv_type is not None:
            assert name.lower() == 'lambda', \
                "When introducing bound variables, the name of the expanded function must be 'lambda'."

            newrule = BVAddGrammarRule(nt, name,to, p=p, bv_type=bv_type, bv_args=bv_args, bv_prefix=bv_prefix, bv_p=bv_p)
        else:
            newrule = GrammarRule(nt,name,to, p=p)

        self.rules[nt].append(newrule)
        return newrule
    
    def is_terminal_rule(self, r):
        """
        Check if a rule is "terminal" - meaning that it doesn't contain any nonterminals in its expansion.
        """ 
        return not any([self.is_nonterminal(a) for a in None2Empty(r.to)])  


    # --------------------------------------------------------------------------------------------------------
    # Generation
    # --------------------------------------------------------------------------------------------------------

    def generate(self, x=None):
        """Generate from the grammar

        Arguments:
            x (string): What we start from -- can be None and then we use Grammar.start.

        """
        # print "# Calling Grammar.generate", type(x), x

        # Decide what to start from based on the default if start is not specified
        if x is None:
            x = self.start
            assert self.start in self.nonterminals(), \
                "The default start symbol %s is not a defined nonterminal" % self.start

        # Dispatch different kinds of generation
        if isinstance(x,list):            
            # If we get a list, just map along it to generate.
            # We don't count lists as depth--only FunctionNodes.
            return map(lambda xi: self.generate(x=xi), x)
        elif self.is_nonterminal(x):

            # sample a grammar rule
            rules = self.get_rules(x)
            assert len(rules) > 0, "*** No rules in x=%s"%x

            # sample the rule
            r = weighted_sample(rules, probs=lambda x: x.p, log=False)

            # Make a stub for this functionNode 
            fn = r.make_FunctionNodeStub(self, None)

            # Define a new context that is the grammar with the rule added
            # Then, when we exit, it's still right.
            with BVRuleContextManager(self, fn, recurse_up=False):      # not sure why we can't use with/as:
                # Can't recurse on None or else we genreate from self.start
                if fn.args is not None:
                    # and generate below *in* this context (e.g. with the new rules added)
                    try:
                        fn.args = self.generate(fn.args)
                    except RuntimeError as e:
                        print "*** Runtime error in %s" % fn
                        raise e


                # and set the parents
                for a in fn.argFunctionNodes():
                    a.parent = fn

            return fn

        else:  # must be a terminal
            assert isinstance(x, str), ("*** Terminal must be a string! x="+x)
            return x

    def enumerate(self, d=20, nt=None, leaves=True):
        """Enumerate all trees up to depth n.

        Parameters:
            d (int): how deep to go? (defaults to 20 -- if Infinity, enumerate() runs forever)
            nt (str): the nonterminal type
            leaves (bool): do we put terminals in the leaves or leave nonterminal types? This is useful in
              PartitionMCMC

        """
        for i in infrange(d):
            for t in self.enumerate_at_depth(i, nt=nt, leaves=leaves):
                yield t

    def enumerate_at_depth(self, d, nt=None, leaves=True):
        """Generate trees at depth d, no deeper or shallower.

        Parameters
            d (int): the depth of trees you want to generate
            nt (str): the type of the nonterminal you want to return (None reverts to self.start)
            leaves (bool): do we put terminals in the leaves or leave nonterminal types? This is useful in
              PartitionMCMC. This returns trees of depth d-1!

        Return:
            yields the ...

        """
        if nt is None:
            nt = self.start

        # handle garbage that may be passed in here
        if not self.is_nonterminal(nt):
            yield nt
            raise StopIteration

        if d == 0:
            if leaves:
                # Note: can NOT use filter here, or else it doesn't include added rules
                for r in self.rules[nt]:
                    if self.is_terminal_rule(r):
                        yield r.make_FunctionNodeStub(self, None)
            else:
                # If not leaves, we just put the nonterminal type in the leaves
                yield nt
        else:
            # Note: can NOT use filter here, or else it doesn't include added rules. No sorting either!
            for r in self.rules[nt]:

                # No good since it won't be deep enough
                if self.is_terminal_rule(r):
                    continue


                fn = r.make_FunctionNodeStub(self, None)

                # The possible depths for the i'th child
                # Here we just ensure that nonterminals vary up to d, and otherwise
                child_i_depths = lambda i: xrange(d) if self.is_nonterminal(fn.args[i]) else [0]

                # The depths of each kid
                for cd in lazyproduct(map(child_i_depths, xrange(len(fn.args))), child_i_depths):

                    # One must be equal to d-1
                    # TODO: can be made more efficient via permutations. Also can skip terminals in args.
                    if max(cd) < d-1:
                        continue
                    assert max(cd) == d-1

                    myiter = lazyproduct(
                        [self.enumerate_at_depth(di, nt=a, leaves=leaves) for di, a in zip(cd, fn.args)],
                        lambda i: self.enumerate_at_depth(cd[i], nt=fn.args[i], leaves=leaves))
                    try:
                        while True:
                            # Make a copy so we don't modify anything
                            yieldfn = copy(fn)

                            # BVRuleContextManager here makes us remove the rule BEFORE yielding,
                            # or else this will be incorrect. Wasteful but necessary.
                            with BVRuleContextManager(self, fn, recurse_up=False):
                                yieldfn.args = myiter.next()
                                for a in yieldfn.argFunctionNodes():
                                    # Update parents
                                    a.parent = yieldfn

                            yield copy(yieldfn)

                    except StopIteration:
                        # Catch this here so we continue in this loop over rules
                        pass

    def depth_to_terminal(self, x, openset=None, current_d=None):
        """
        Return a dictionary that maps both this grammar's rules and its nonterminals to a number,
        giving how quickly we can go from that nonterminal or rule to a terminal.

        Arguments:
            openset(doc?): stores the set of things we're currently trying to compute for. We must skip rules
              that contain anything in there, since they have to be defined still, and so we want to avoid
              a loop.

        """
        if current_d is None: 
            current_d = dict()
            
        if openset is None:
            openset = set()
            
        openset.add(x)
        
        if isinstance(x, GrammarRule):
            if x.to is None or len(x.to) == 0:
                current_d[x] = 0 # we are a terminal
            else:
                current_d[x] = 1 + max([(self.depth_to_terminal(a, openset=openset, current_d=current_d)
                                        if a not in openset else 0) for a in x.to])
        elif isinstance(x, str):
            if x not in self.rules:
                current_d[x] = 0    # A terminal
            else:
                current_d[x] = min([(self.depth_to_terminal(r, openset=openset, current_d=current_d)
                                    if r not in openset else Infinity) for r in self.rules[x]])
        else:
            assert False, "Shouldn't get here!"

        openset.remove(x)
        return current_d[x]

    def renormalize(self):
        """ go through each rule in each nonterminal, and renormalize the probabilities """

        for nt in self.nonterminals():
            z = sum([r.p for r in self.get_rules(nt)])
            for r in self.get_rules(nt):
                r.p = r.p / z

