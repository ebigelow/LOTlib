# -*- coding: utf-8 -*-

"""

	TODO:
		- Make the lexicon be indexable like an array/dict, rather than having to say h.lex[...] say h[..]
		

"""

from random import sample
from copy import deepcopy

from LOTlib.Hypothesis import Hypothesis, StandardExpression
from LOTlib.DataAndObjects import UtteranceData
from LOTlib.Miscellaneous import *

class SimpleLexicon(Hypothesis):
	"""
		A class for representing learning single words 
		This stores a dictionary mapping words to FunctionHypotheseses
		A SimpleLexicon itself is a hypothesis, allowing joint inferences over the whole set of meanings 
	
		You may overwrite the weightfunction, which maps *[h f] to a positive number corresponding to the probability of producing a word h, f
		It defaults to lambda x: 1.0
	
		TODO: we can probably make this faster by not passing around the context sets so much.
		
	"""
	
	def __init__(self, G, args, alpha=0.90, palpha=0.90):
		Hypothesis.__init__(self)
		self.lex = dict()
		self.grammar = G
		self.args = args
		self.alpha = alpha
		self.palpha = palpha
		
	def copy(self):
		""" Copy a lexicon. We don't re-create the fucntions since that's unnecessary and slow"""
		new = SimpleLexicon(self.grammar, self.args, alpha=self.alpha, palpha=self.palpha)
		for w in self.lex.keys():
			new.lex[w] = self.lex[w].copy()
		return new
		
	def __str__(self):
		out = ''
		for w, v in self.lex.iteritems():
			out = out + str(w) + ':\t' + str(v) + '\n'
		return out
	def __hash__(self): return hash(str(self))
	def __eq__(self, other):   return (str(self)==str(other)) # simple but there are probably better ways
	
	# this sets the word and automatically compute its function
	def set_word(self, w, v, f=None):
		""" 
			This sets word w to value v. v can be either a FunctionNode or a  Hypothesis, and
			in either case it is copied here. When it is a Hypothesis, the value is exrtacted
		"""
		
		# Conver to standard expressiosn
		if isinstance(v, Hypothesis): v = v.value # extract the value (hopefully a FunctionNode)
		v = v.copy() # and copy it
		
		assert isinstance(v, FunctionNode)
			
		# set this equal to the StandardExpression
		if self.lex.get(w,None) is None:
			self.lex[w] = StandardExpression(self.grammar, v=v, f=f, args=self.args) # create this
		else:	self.lex[w].set_value(v=v, f=f)

	# set this function (to make something to sample from)
	def force_function(self, w, f):
		self.lex[w] = StandardExpression(self.grammar, v=None, f=f) # this sloppily generates v=None, but it's easy for now
		self.lex[w].value = w # Now overwrite so .value is just the word
		
	def all_words(self): return self.lex.keys()
	
	def weightfunction(self, h): return 1.0
	def weightfunction_word(self, w):
		return self.weightfunction(self.lex[w])
		
		
	###################################################################################
	## MH stuff
	###################################################################################
	
	def propose(self):
		new = self.copy()
		w = weighted_sample(self.lex.keys()) # the word to change
		p,fb = self.grammar.propose( self.lex[w].value )
		
		new.set_word(w, p)
		
		return new, fb

		
	def compute_prior(self):
		self.prior = sum([x.compute_prior() for x in self.lex.values()])
		self.lp = self.prior + self.likelihood
		return self.prior		
			
	# This combines score_utterance with likelihood so that everything is much faster
	def compute_likelihood(self, data):
		self.likelihood = 0.0
		
		# pre-compute the weights for each word
		weights = dict()
		for w in self.lex.keys():
			weights[w] = self.weightfunction(self.lex[w])
		
		# set up the stored likelihood
		N = len(data)
		self.stored_likelihood = [None]*N
		
		self.likelihood = 0.0
		for i in xrange(N):
			
			di = data[i]
			
			# evaluate the word in the context
			f = self.lex[di.word](*di.context) 
			
			# partition all the other utterances
			t,m = self.partition_words(di.all_words, di.context)
			
			## This is the slow part! You should score a bunch of utterances at once!
			all_weights  = sum(map( lambda x: weights[x], di.all_words))
			true_weights = sum(map( lambda x: weights[x], t))
			met_weights  = sum(map( lambda x: weights[x], m))
			
			p = 0.0
			w = weights[di.word] # the weight of the one we have
			if f == True:    p = self.palpha * self.alpha * w / true_weights + self.palpha * (1.0 - self.alpha) * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
			elif f == False: p = ifelse(true_weights==0, 1.0, 1.0-self.alpha) * self.palpha * w / met_weights + (1.0 - self.palpha) * w / all_weights # choose from the trues
			else:            p = ifelse(met_weights==0, 1.0, (1.0 - self.palpha)) * w / all_weights
			
			self.stored_likelihood[i] = log(p)
			
			self.likelihood += self.stored_likelihood[i]
			
		self.lp = self.prior + self.likelihood
		return self.likelihood
			
		
	###################################################################################
	## Sampling and dealing with responses
	###################################################################################
	
	# Take a set of utterances and partition them into true/met
	def partition_words(self, all_words, context):
		"""
			Partition utterances, returning those that are true, and those with the presup met	
			NOTE: The commended code is much slower
		"""
			
		trues, mets = [], []
		for w in all_words:
			fa = self.lex[w](*context)
			if fa is True:    
				trues.append(w)
				mets.append(w)
			elif fa is False: 
				mets.append(w)
			else: pass # not met
			
		return [trues, mets]
	
	# take a set of utterances and sample them according to our probability model
	def sample_word(self, all_words, context):
		
		t,m = self.partition_words(all_words, context)
		
		if flip(self.palpha) and (len(m) > 0): # if we sample from a presup is true
			if (flip(self.alpha) and (len(t)>0)):
				return weighted_sample(t,         probs=map(self.weightfunction_word, t), log=False)
			else:   return weighted_sample(m,         probs=map(self.weightfunction_word, m), log=False)
		else:           return weighted_sample(all_words, probs=map(self.weightfunction_word, all_words), log=False) # sample from all utterances
		
	# must wrap these as SimpleExpressionFunctions
	def enumerative_proposer(self):
		
		for w in self.all_words():
			for k in self.grammar.enumerate_pointwise(self.lex[w]):
				new = self.copy()
				new.set_word(w, k)
				yield new
	
	

class VectorizedLexicon(Hypothesis):
	"""
		This is a Lexicon class that stores *only* indices into my_finite_trees, and is designed for doing gibbs,
		sampling over each possible word meaning. It requires running EnumerateTrees.py to create a finite set of 
		trees, and then loading them here. Then, all inferences etc. are vectorized for super speed.
		
		This requires a bunch of variables (the global ones) from run, so it cannot portably be extracted. But it can't
		be defined inside run or else it won't pickle correctly. Ah, python. 
		
		NOTE: This takes a weirdo form for the likelihood function
	"""
	
	def __init__(self, target_words, finite_trees, priorlist, word_idx=None, ALPHA=0.75, PALPHA=0.75):
		Hypothesis.__init__(self)
		
		self.__dict__.update(locals())
		
		self.N = len(target_words)
		
		if self.word_idx is None: 
			self.word_idx = np.array( [randint(0,len(self.priorlist)-1) for i in xrange(self.N) ])
		
	def __repr__(self): return str(self)
	def __eq__(self, other): return np.all(self.word_idx == other.word_idx)
	def __hash__(self): return hash(tuple(self.word_idx))
	def __cmp__(self, other): return cmp(str(self), str(other))
	
	def __str__(self): 
		s = ''
		aw = self.target_words
		for i in xrange(len(self.word_idx)):
			s = s + aw[i] + "\t" + str(self.finite_trees[self.word_idx[i]]) + "\n"
		s = s + '\n'
		return s
	
	def copy(self):
		return VectorizedLexicon(self.target_words, self.finite_trees, self.priorlist, word_idx=np.copy(self.word_idx), ALPHA=self.ALPHA, PALPHA=self.PALPHA)
	
	def enumerative_proposer(self, wd):
		for k in xrange(len(self.finite_trees)):
			new = self.copy()
			new.word_idx[wd] = k
			yield new
			
	def compute_prior(self):
		self.prior = sum([self.priorlist[x] for x in self.word_idx])
		self.lp = self.prior + self.likelihood
		return self.prior
	
	def compute_likelihood(self, data):
		"""
			Compute the likelihood on the data, super fast
			
			These use the global variables defined by run below.
			The alternative is to define this class inside of run, but that causes pickling errors
		"""
		
		response,weights,uttered_word_index = data # unpack the data
		
		# vector 0,1,2, ... number of columsn
		zerothroughcols = np.array(range(len(uttered_word_index)))
		
		r = response[self.word_idx] # gives vectors of responses to each data point
		w = weights[self.word_idx]
		
		if r.shape[1] == 0:  # if no data
			self.likelihood = 0.0
		else:
			
			true_weights = ((r > 0).transpose() * w).transpose().sum(axis=0)
			met_weights  = ((np.abs(r) > 0).transpose() * w).transpose().sum(axis=0)
			all_weights = w.sum(axis=0)
			rutt = r[uttered_word_index, zerothroughcols] # return value of the word actually uttered
			
			## now compute the likelihood:
			lp = np.sum( np.log( self.PALPHA*self.ALPHA*weights[self.word_idx[uttered_word_index]]*(rutt>0) / (1e-5 + true_weights) + \
					self.PALPHA*( (true_weights>0)*(1.0-self.ALPHA) + (true_weights==0)*1.0) * weights[self.word_idx[uttered_word_index]] * (np.abs(rutt) > 0) / (1e-5 + met_weights) + \
					( (met_weights>0)*(1.0-self.PALPHA) + (met_weights==0)*1.0 ) * weights[self.word_idx[uttered_word_index]] / (1e-5 + all_weights)))
				
			self.likelihood = lp
		self.lp = self.likelihood+self.prior
		return self.likelihood 
		
	def propose(self):
		print "*** Cannot propose to VectorizedLexicon"
		assert False