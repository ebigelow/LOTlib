"""
    Routines for evaling
"""
import sys

from LOTlib.Miscellaneous import raise_exception
from EvaluationException import EvaluationException

# All of these are defaulty in the context for eval.
from Primitives.Arithmetic import *
from Primitives.Combinators import *
from Primitives.Features import *
from Primitives.Functional import *
from Primitives.Logic import *
from Primitives.Number import *
from Primitives.Semantics import *
from Primitives.SetTheory import *
from Primitives.Trees import *
from Primitives.Stochastics import *


def register_primitive(function, name=None):
    """
        This allows us to load new functions into the evaluation environment.
        Defaultly all in LOTlib.Primitives are imported. However, we may want to add our
        own functions, and this makes that possible. As in,

        register_primitive(flatten)

        or

        register_primitive(flatten, name="myflatten")

        where flatten is a function that is defined in the calling context and name
        specifies that it takes a different name when evaled in LOTlib

        TODO: Add more convenient means for importing more methods
    """

    if name is None:
        name = function.__name__

    sys.modules['__builtin__'].__dict__[name] = function


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ The Y combinator and a bounded version
# example:
# fac = lambda f: lambda n: (1 if n<2 else n*(f(n-1)))
# Y(fac)(10)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Y = lambda f: (lambda x: x(x))(lambda y: f(lambda *args: y(y)(*args)) )
MAX_RECURSION = 25


def Y_bounded(f):
    """
    A fancy fixed point iterator that only goes MAX_RECURSION deep, else throwing a a RecusionDepthException
    """
    return (lambda x, n: x(x, n))(lambda y, n: f(lambda *args: y(y, n+1)(*args))
                                  if n < MAX_RECURSION else raise_exception(EvaluationException()), 0)


def Ystar(*l):
    """
    The Y* combinator for mutually recursive functions. Holy shit.

    (define (Y* . l)
          ((lambda (u) (u u))
            (lambda (p) (map (lambda (li) (lambda x (apply (apply li (p p)) x))) l))))

    See:
    http://okmij.org/ftp/Computation/fixed-point-combinators.html]
    http://stackoverflow.com/questions/4899113/fixed-point-combinator-for-mutually-recursive-functions

    E.g., here is even/odd:

    even,odd = Ystar( lambda e,o: lambda x: (x==0) or o(x-1), \
                          lambda e,o: lambda x: (not x==0) and e(x-1) )

        Note that we require a lambda e,o on the outside so that these can have names inside.
    """

    return (lambda u: u(u))(lambda p: map(lambda li: lambda x: apply(li, p(p))(x), l))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~ Evaluation of expressions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate_expression(e):
    """
    Evaluate the expression, wrapping in an error in case it can't be evaled
    """
    assert isinstance(e,str)
    try:
        return eval(e)
    except Exception as ex:
        print "*** Error in evaluate_expression:", ex
        raise ex
