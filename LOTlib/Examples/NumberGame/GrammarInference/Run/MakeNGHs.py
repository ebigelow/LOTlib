
"""

Command line args
-----------------
-f --fname
    File name to save to (must be .p)
-do --domain
    Domain for NumberGameHypothesis
-a --alpha
    Alpha, the noise parameter

-g --grammar
    Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]
-d --data
    Domain for NumberGameHypothesis
-i --iters
    Number of samples to run
-c --chains
    Number of chains to run on each data input
-n
    Only keep top N samples per MPI run (if we're doing MPI), or total (if not MPI)

-mcmc
    Do we do MCMC? [1 | 0]
-mpi
    Do we use MPI? (only if MCMC) [1 | 0]
-enum
    How deep to enumerate hypotheses? (only if not MCMC)

Example
-------
$ python Demo.py -f out/ngh_lot100k.p -do 100 -a 0.9 -g lot_grammar -d josh_data -i 100000 -c 10 -n 1000 -mcmc -mpi

"""

from optparse import OptionParser

from LOTlib.Inference.Samplers.MetropolisHastings import MHSampler
from LOTlib.MPI.MPI_map import MPI_unorderedmap
from LOTlib.MCMCSummary.TopN import TopN
from LOTlib.Examples.NumberGame.GrammarInference.Model import *



# ============================================================================================================
# Parsing command-line options

parser = OptionParser()

parser.add_option("-f", "--filename",
                  dest="filename", type="string", default="ngh_default.p",
                  help="File name to save to (must be .p)")
parser.add_option("-do", "--domain",
                  dest="domain", type="int", default=100,
                  help="Domain for NumberGameHypothesis")
parser.add_option("-a", "--alpha",
                  dest="alpha", type="float", default=0.9,
                  help="Alpha, the noise parameter")

parser.add_option("-g", "--grammar",
                  dest="grammar", type="string", default="mpi-run.pkl",
                  help="Which grammar do we use? [mix_grammar | independent_grammar | lot_grammar]")
parser.add_option("-d", "--data",
                  dest="data", type="string", default="josh_data",
                  help="Which data do we use? [josh_data | filename.p]")
parser.add_option("-i", "--iters",
                  dest="iters", type="int", default=100000,
                  help="Number of samples to run per chain")
parser.add_option("-c", "--chains",
                  dest="chains", type="int", default=1000,
                  help="Number of chains to run on each data input")
parser.add_option("-n",
                  dest="N", type="int", default=1000,
                  help="Only keep top N samples per MPI run (if we're doing MPI), or total (if not MPI)")

parser.add_option("-mpi",
                  action="store_true", dest="mpi", default=True,
                  help="Do we use MPI? (only if MCMC)")
parser.add_option("-enum",
                  action="store_true", dest="enum_depth", default=False,
                  help="How deep to enumerate hypotheses? (only if not MCMC)")

(options, args) = parser.parse_args()


# ============================================================================================================
# MPI method

def mpirun(d):
    """
    Generate NumberGameHypotheses using MPI.

    """
    h0 = NumberGameHypothesis(grammar=grammar, domain=100, alpha=0.9)
    mh_sampler = MHSampler(h0, d.input, options.iters)
    hypotheses = TopN(N=options.N)

    for h in break_ctrlc(mh_sampler):
        hypotheses.add(h)
    return [h for h in hypotheses.get_all()]


# ============================================================================================================
# Sample hypotheses
# ============================================================================================================

if __name__ == "__main__":

    if options.grammar is 'mix':
        grammar = mix_grammar
    elif options.grammar is 'indep':
        grammar = independent_grammar
    elif options.grammar is 'lot':
        grammar = lot_grammar
    else:
        grammar = None

    # Add more data options . . .
    if options.data is 'josh_data':
        data = import_josh_data()
    else:
        data = import_pd_data(options.data)

    # --------------------------------------------------------------------------------------------------------
    # MCMC sampling

    # MPI
    if options.mpi:
        hypotheses = set()
        hypo_sets = MPI_unorderedmap(mpirun, [[d] for d in data * options.chains])
        for hypo_set in hypo_sets:
            hypotheses = hypotheses.union(hypo_set)

    # No MPI
    else:
        hypotheses = set()

        for d in data * options.chains:
            h0 = NumberGameHypothesis(grammar=grammar, domain=options.domain, alpha=options.alpha)
            mh_sampler = MHSampler(h0, d, options.iters)

            chain_hypos = TopN(N=options.N)
            for h in break_ctrlc(mh_sampler):
                chain_hypos.add(h)
            hypotheses = hypotheses.union(chain_hypos.get_all())


    # --------------------------------------------------------------------------------------------------------
    # Save hypotheses

    f = open(options.filename, "wb")
    pickle.dump(hypotheses, f)
