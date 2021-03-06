"""
        A small finite grammar with bound variables that have bv_args.
"""


from LOTlib.Grammar import Grammar
import math

g = Grammar()

g.add_rule("START", 'lambda', ["A"], 1.0, bv_type="A", bv_args=["B"], bv_prefix="p")
g.add_rule("A", 'a', "B", 1.0)
g.add_rule("A", 'a', "w", 1.0)
g.add_rule("A", 'a', "x", 1.0)
g.add_rule("B", 'b', "y", 1.0)
g.add_rule("B", 'b', "z", 1.0)

# manually computes the log probability of a given tree being generated by the above grammar
def log_probability(tree):
    ls = tree.as_list()
    # print "tree is", ls
    # the total log probability is just the log probability of what's generated under the START node
    return log_probability_A(ls[1])

# the log probability of a given subtree with root A being generated
def log_probability_A(ls):
    # compute the sum of unnormalized probabilities for rules of the form A --> x
    #                                                                               see what I did there Samay?
    #                                                                                                               |
    #                                                                                                               |
    #                                                                                                               |
    #                                                                                                               !
    sum_prob_A = 13.0 # NOTE THAT BOUND VARIABLE RULES DEFAULTLY HAVE PROBABILITY 10.0!!!!!!
    # see if we have a bound variable or not
    if ls[0] == 'a':
        if ls[1][0] == 'b': return math.log(1.0/sum_prob_A) + log_probability_B(ls[1])
        else: return math.log(1.0/sum_prob_A)
    else: return math.log(10.0/sum_prob_A) + log_probability_B(ls[1])

# the log probability of a subtree with root B being generated
def log_probability_B(ls):
    # this is easy
    return math.log(1.0/2)


if __name__ == "__main__":
    for i in xrange(100):
        print(g.generate())
