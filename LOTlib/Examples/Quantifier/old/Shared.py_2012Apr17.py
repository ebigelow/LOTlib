# -*- coding: utf-8 -*-
"""
	Shared functions for the quantifier learning model. 
"""

import numpy

from LOTlib.bvPCFG import *
from LOTlib.BasicPrimitives import *
import LOTlib.MetropolisHastings
from LOTlib.Miscellaneous import *
from LOTlib.Lexicon import *
from LOTlib.Objects import *
from LOTlib.Hypothesis import *
from LOTlib.Memoization import *

############################################################
# Manipulate lexica



def load_finite_trees(f):
	"""
		Load in either a list of fintie trees 
		This can come in either two formats -- a "finite sample" of hypotheses, or a "finite sample" of lexica.
		In the latter case, we take the top for *each word* and just use them
	"""
	inh = open(f)
	fs = pickle.load(inh)
	if isinstance(fs.peek()[0], FunctionNode): # if we store trees in each
		return fs.get_all()
	elif isinstance(fs.peek()[0], VectorizedLexicon) or isinstance(fs.peek()[0], SimpleLexicon): # else we want to extract the treesed from this priority queue
		upq = UniquePriorityQueue(N=-1)
		for l in fs.get_all():
			for e in l.dexpr.values():
				upq.push(e, 0.0)
		return upq.get_all()
	

############################################################
# Making data, sets, and objects

# quantifiers involving cardinality
all_objects = make_all_objects(shape=['man', 'woman', 'child'], job=['pirate', 'chef', 'fireman'])

def sample_context_set():
	# the i'th data point
	set_size = randint(1,8)
	
	# get the objects in the current set
	si = sample_sets_of_objects(set_size, all_objects)
	
	context = [
		set([o for o in si if o.shape=='man']), 
		set([o for o in si if o.job=='pirate']),
		set(si)
		]
		
	return context


def generate_data(data_size):
	data = []
	for i in xrange(data_size):
		
		# a context is a set of men, pirates, and everything. functions are applied to this to get truth values
		context = sample_context_set()
		
		# get all possible utterances in this context (here, this is all words in the context set)
		all_utterances = make_all_SimpleUtterances(target.all_words(), [context]) # takes a *list* of contexts and a list of words and crosses
		u = target.sample_utterance(all_utterances)
		#print "SAMPLING u=",u
		data.append( [ u, all_utterances] )
	
	return data


def show_baseline_distribution(N=10000):
	d = generate_data(N)
	
	frq = dict()
	for di in d: hashplus(frq, di[0].word)
	
	for w in frq.keys():
		print w, frq[w]

def get_single_tree_responses(t, x, utterance=True, data=False):
	"""
		Get responses to a sequence of data, as either sets, utterances, or "data"
		If utterance is true, then we extract di.context; else we pretend data is a sequence of three sets
		
		
	"""
	f = evaluate_expression(t, args=['A', 'B', 'S'])
	
	#print data[0].context
	if data:      return [f(*di[0].context) for di in x]
	if utterance: return [f(*di.context) for di in x]
	else:	return [f(*di) for di in x]
	
	
def mapto012(resp):
	"""
		Make True/False/other 1,-1,0 respectively -- handles all kinds of undef
	"""
	out = []
	for k in resp:
		if k is True:    out.append(1)
		elif k is False: out.append(-1)
		else:            out.append(0)
	return out
	
def extract_presup(resp):
	"""
		From a bunch of responses, extract the T/F presups
	"""
	out = []
	for k in resp:
		if is_undef(k): out.append(False)
		else:           out.append(True)
	return out
	
def extract_literal(resp):
	"""
		From a bunch of responses, extract the T/F literals
	"""
	out = []
	for k in resp:
		if (k is True) or (k == "undefT"): out.append(True)
		else:                              out.append(False)
	return out
def collapse_undefs(resp): 
	"""
		Collapse together our multiple kinds of undefs so that we can compare vectors
	"""
	out = []
	for k in resp:
		if is_undef(k): out.append("undef")
		else:           out.append(k)
	return out
	
############################################################
# Set up the grammar
G = PCFG()

G.add_rule('START', 'presup_', ['BOOL', 'BOOL'], 1.0)
G.add_rule('START', 'presup_', ['True', 'BOOL'], 1.0)

#G.add_rule('BOOL', 'not_', ['BOOL'], 1.0)
#G.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 1.0)
#G.add_rule('BOOL', 'or_', ['BOOL', 'BOOL'], 1.0)

G.add_rule('BOOL', 'empty_', ['SET'], 1.0)
G.add_rule('BOOL', 'nonempty_', ['SET'], 1.0)
G.add_rule('BOOL', 'subset_', ['SET', 'SET'], 1.0)
G.add_rule('BOOL', 'exhaustive_', ['SET', 'S'], 1.0)
G.add_rule('BOOL', 'cardinality1_', ['SET'], 1.0) # if cardinalities are included, don't include these!
G.add_rule('BOOL', 'cardinality2_', ['SET'], 1.0)
G.add_rule('BOOL', 'cardinality3_', ['SET'], 1.0)

G.add_rule('SET', 'union_', ['SET', 'SET'], 1.0)
G.add_rule('SET', 'intersection_', ['SET', 'SET'], 1.0)
G.add_rule('SET', 'setdifference_', ['SET', 'SET'], 1.0)

G.add_rule('SET', 'A', [], 10.0)
G.add_rule('SET', 'B', [], 10.0)
#G.add_rule('SET', 'S', [], 10.0)

# Cardinality operations
G.add_rule('BOOL', 'cardinalityeq_', ['SET', 'SET'], 1.0)
G.add_rule('BOOL', 'cardinalitygt_', ['SET', 'SET'], 1.0)
##G.add_rule('BOOL', 'cardinalityeq_', ['SET', 'CARD'], 1.0)
#G.add_rule('BOOL', 'cardinalitygt_', ['SET', 'CARD'], 1.0)
#G.add_rule('BOOL', 'cardinalitygt_', ['CARD', 'SET'], 1.0)
##G.add_rule('CARD', 'cardinality_', ['SET'], 1.0)

#G.add_rule('CARD',  '0', [], 1.0)
#G.add_rule('CARD',  '1', [], 1.0)
#G.add_rule('CARD',  '2', [], 1.0)
#G.add_rule('CARD',  '3', [], 1.0)
#G.add_rule('CARD',  '4', [], 1.0)
#G.add_rule('CARD',  '5', [], 1.0)
#G.add_rule('CARD',  '6', [], 1.0)
#G.add_rule('CARD',  '7', [], 1.0)
#G.add_rule('CARD',  '8', [], 1.0)


# create a small list of all plausible context sets. 
# NOTE: Do no use this if utterances consist of all possible words (e.g. man/pirate are allowed to vary)
all_possible_context_sets = []
APD_N = 6
for adb in xrange(APD_N):
	for bda in xrange(APD_N-adb):
		for anb in xrange(APD_N-adb-bda):
			for s in xrange(APD_N-adb-bda-anb):
				adb_ = set([Obj(shape="man", job="chef") for i in xrange(adb)])
				bda_ = set([Obj(shape="woman", job="pirate") for i in xrange(bda)])
				anb_ = set([Obj(shape="man", job="pirate") for i in xrange(anb)])
				s_   = set([Obj(shape="woman", job="chef") for i in xrange(s)])
				
				all_possible_context_sets.append( [adb_.union(anb_), bda_.union(anb_), s_])
#print all_possible_context_sets
#print len(all_possible_context_sets)



###################################################################################
## Doing Gricean lexcion things
###################################################################################

TESTING_SET_SIZE = 1000
TESTING_SET = all_possible_context_sets # [ sample_context_set() for x in xrange(TESTING_SET_SIZE)  ]
nu = 0.10 ## Note: The max weight is 1/nu, and this should not be huge compared to 1/alpha

def is_conservative(h):
	"""
		Check if a hypothesis (funciton node) is conservative or not
	"""
	f = evaluate_expression(h, ['A', 'B', 'S'])
	for x in TESTING_SET:
		a,b,s = x
		if f(a,b,s) != f(a, b.intersection(a), s.intersection(a) ): # HMM: is this right? We intersect s with a?
			return False
	return True
	
def create_hyp2target(hyps, presup=False, literal=False):
	"""
		Take a list of hypotheses and make a hash of them to the target word (or None if they aren't at target word)
		If presup or literal is True, we evaluate ONLY those parts of the lexicon
	"""
	## And figure out who is one of th target meanings
	hyp2correct = dict() # map each hypothesis to target meaning or None
	correct2response = dict() # map responses to a target word
	for w in target.all_words(): 
		r =  [target.dfunc[w](*s) for s in TESTING_SET]
		correct2response[ str(r) ] = w # store responses to 
	for h in hyps: 
		f = evaluate_expression(h, ['A', 'B', 'S'])
		r = [f(*s) for s in TESTING_SET]
		hyp2correct[h] = correct2response.get( str(r), None)
	return hyp2correct


#def correct_word(f):
	#"""
		#Return the correct word for a hypothesis
		#and None if there is none
	#"""	
	#r = [ f(*s) for s in TESTING_SET ]
	
	#for x in TESTING_SET:
		#a,b,s = x
		#if f(a,b,s) != f(a, b.intersection(a), s.intersection(a) ): # HMM: is this right? We intersect s with a?
			#return False
	#return True

@BoundedMemoize(N=1000)
def my_gricean_weight(h, f):
	"""
	Takes a hypothesis and its function and returns the weight under a gricean setup, where the production probability is proportional to 
	
	exp( 1.0 / (nu + proportionoftimeitistrue) )
	
	We boundedly memoize this
	
	"""
	#return 1.0
	pct = float(sum(map(lambda s: ifelse(f(*s), 1.0, 0.0), TESTING_SET) )) / float(TESTING_SET_SIZE) # pul out the context sets and apply f
		
	w = 1.0 / (nu + pct)
		
	#print h, w # To get an idea of how different nu values work		
	return w

# A version of simpleLexicon that uses this
# Like an idiot, we tried passing in a version of weightfunction on construction, but then we could not pickle
class GriceanSimpleLexicon(SimpleLexicon):
	def weightfunction(self, h, f):
		return my_gricean_weight(h,f)


#def random_GriceanSimpleLexicon():
	#g = GriceanSimpleLexicon(G, ['A', 'B', 'S'])
	#for w in target.all_words():
		#g.set_word(w, G.generate('START'))
	#return g
		
#####################################
## Define the target

target = GriceanSimpleLexicon(G, ['A', 'B', 'S'])
target.force_function('every', lambda A, B, S: presup_(nonempty_(A), subset_(A,B)))
target.force_function('some',  lambda A, B, S: presup_(True, nonempty_(intersection_(A,B))))
target.force_function('the',   lambda A, B, S: presup_(cardinality1_(A), nonempty_(intersection_(A,B))))
target.force_function('no',  lambda A, B, S: presup_(nonempty_(A), empty_(intersection_(A,B))))

target.force_function('both',  lambda A, B, S: presup_(cardinality2_(A), cardinality2_(intersection_(A,B))))
target.force_function('neither',  lambda A, B, S: presup_(cardinality2_(A), empty_(intersection_(A,B))))
target.force_function('either',  lambda A, B, S: presup_(cardinality2_(A), cardinality1_(intersection_(A,B))))

## some non-cardinality operations
target.force_function('one',    lambda A, B, S: presup_(nonempty_(A), cardinality1_(intersection_(A,B))))
target.force_function('two',    lambda A, B, S: presup_(nonempty_(A), cardinality2_(intersection_(A,B))))
target.force_function('three',  lambda A, B, S: presup_(nonempty_(A), cardinality3_(intersection_(A,B))))

## Cardinality operations:
target.force_function('most',  lambda A, B, S: presup_(nonempty_(A), cardinalitygt_(intersection_(A,B), setdifference_(A,B))))
#target.force_function('few',   lambda A, B, S: presup_(True, cardinalitygt_(3, intersection_(A,B))))
#target.force_function('many',   lambda A, B, S: presup_(True, cardinalitygt_(intersection_(A,B), 3)))
#target.force_function('half',   lambda A, B, S: presup_(nonempty_(A), cardinalityeq_(intersection_(A,B), setdifference_(A,B))))

# map each word to an index -- NOTE: The form in the class below MUST match this order
word2index = dict()
index2word = target.all_words()
for i, w in enumerate(index2word):
	word2index[w] = i
	index2word[i] = w

#####################################


#####################################
## FOR DEBUGGING

#distribution of context sizes
#for i in xrange(1000):
	#context = sample_context_set()
	#print len(context[0]), len(context[1]), len(context[2])
	
#comparison = GriceanSimpleLexicon(G, ['A', 'B', 'S'])
#comparison.force_function('every', lambda A, B, S:  presup_( nonempty_( A ), empty_( A ) ))
#comparison.force_function('some',  lambda A, B, S: presup_( True, subset_( A, A ) ))
##This will debug -- for a given context and all_utterances, see if our likelihood is the same as
##empirical sampling
#context = sample_context_set()
#print context
##print len(context[0]), context[0].issubset(context[1])
#all_utterances = make_all_SimpleUtterances(target.all_words(), [context])
#h = dict()
#NN = 10000
#for xi in range(NN):
	#u = target.sample_utterance(all_utterances)
	#hashplus(h, u.word)
##print h

#from scipy.stats import chisquare
#cnts = [ (h[w]/float(NN)) for w in h.keys() ]
#Z = logsumexp([target.score_utterance(SimpleUtterance(w, context), all_utterances) for w in h.keys() ])
#exps = [ exp(target.score_utterance(SimpleUtterance(w, context), all_utterances) - Z) for w in h.keys()]

#comparison_Z =  logsumexp([comparison.score_utterance(SimpleUtterance(w, context), all_utterances) for w in h.keys() ])
#comparison_exps = [ exp(comparison.score_utterance(SimpleUtterance(w, context), all_utterances) - Z) for w in h.keys()]
##print chisquare(f_obs=array(cnts), f_exp=array(exps))
#for w, c, e, ce in zip(h.keys(), cnts, exps, comparison_exps):
	#print w, "\t", target.dfunc[w](*context), "\t", c, "\t", e, "\t", ce
#print "\n\n"
	

#print data

