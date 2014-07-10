# -*- coding: utf-8 -*-

"""
	For each rejection, we increase the temperature by a given amount
	TODO: ADD MEMOIZATION
"""


import LOTlib
from LOTlib import lot_iter
from LOTlib.Miscellaneous import *
from MHShared import MH_acceptance

def increase_temperature_mh_sample(current_sample, data, steps=float("inf"),
	proposer=None, skip=0, prior_temperature=1.0, likelihood_temperature=1.0, temperature=1.0, acceptance_temperature=1.0, trace=False, stats=None, increase_amount=1.1, memoizer=None):
	"""
		current_sample - the starting hypothesis
		data - the conditioning data
		steps - how many steps to run
		proposer - if not None, use this instead of inh.propose() to compute proposals
		skip - only return samples every this many steps
		temperature(s) - the sampler temperatures on variosu components
		trace - if true, we display the random number, proposal, current hypothesis, and sample proposal
		stats - if not none, then we store sampling information in it (hopefully, a MHStats object)
		noisy_memoize - if > 0, store this many hypotheses using a NoisyBoundedHash
	"""
	
	initial_acceptance_temperature = acceptance_temperature
	
	mhi = 0
	while mhi < steps:
		for skp in lot_iter(xrange(skip+1)):
			
			if proposer is None: p, fb = current_sample.propose()
			else:                p, fb = proposer(current_sample)
			
			# either compute this, or use the memoized version
			if memoizer is not None:
				raise NotImplementedError
			else:
				np, nl = p.compute_posterior(data)
						
			#print np, nl, current_sample.prior, current_sample.likelihood
			prop = (np/prior_temperature+nl/likelihood_temperature) / temperature
			cur  = (current_sample.prior/prior_temperature + current_sample.likelihood/likelihood_temperature)/temperature
			
			if MH_acceptance(cur, prop, fb, acceptance_temperature=acceptance_temperature):
				
				if p != current_sample:
					# reset this if we've actually ended up in a new state
					acceptance_temperature = initial_acceptance_temperature
				
				current_sample = p
				
				if stats is not None: stats['accept'] += 1
			else:
				acceptance_temperature = acceptance_temperature * increase_amount
			
			if stats is not None: stats['total'] += 1
			
		if trace: 
			print current_sample.posterior_score, current_sample.likelihood, current_sample.prior, qq(current_sample)
			
		yield current_sample
		
		mhi += 1
		
	#print mem.hits, mem.misses
