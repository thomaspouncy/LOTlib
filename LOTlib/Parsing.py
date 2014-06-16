"""
	Parse simplified lambda expressions--enough for ATIS. 
	
	TODO: No quoting etc. supported yet. 
	
	NOTE: This is based on http://pyparsing.wikispaces.com/file/view/sexpParser.py/278417548/sexpParser.py
"""


from pyparsing import Suppress, alphanums, Group, Forward, ZeroOrMore, Word
from LOTlib.FunctionNode import FunctionNode
from LOTlib.Miscellaneous import unlist_singleton
import pprint

#####################################################################
## Here we define a super simple grammar for lambdas

LPAR, RPAR = map(Suppress, "()")
token = Word(alphanums + "-./_:*+=!<>$")

sexp = Forward()
sexpList = Group(LPAR + ZeroOrMore(sexp) + RPAR)
sexp << ( token | sexpList )

#####################################################################

def simpleLambda2List(s):
	"""
		Return a list of list of lists... for a string containing a simple lambda expression (no quoting, etc)
		NOTE: This converts to lowercase
	"""
	
	x = sexp.parseString(s, parseAll=True)
	return unlist_singleton( x.asList() ) # get back as a list rather than a pyparsing.ParseResults

def simpleLambda2FunctionNode(s, style="atis"):
	"""
		Take a string for lambda expressions and map them to a real FunctionNode tree
	"""
	return list2FunctionNode(simpleLambda2List(s), style=style)


def list2FunctionNode(l, style="atis"):
	"""
		Takes a list *l* of lambda arguments and maps it to a function node. 

		The *style* of lambda arguments could be "atis", "scheme", etc.
	"""
	
	if isinstance(l, list): 
		if len(l) == 0: return None
		elif style is 'atis':
			rec = lambda x: list2FunctionNode(x, style=style) # a wrapper to my recursive self
			if l[0] == 'lambda':
				return FunctionNode('FUNCTION', 'lambda', [rec(l[3])], generation_probability=0.0, bv_type=l[1], bv_args=None ) ## TOOD: HMM WHAT IS THE BV?
			else:
				return FunctionNode(l[0], l[0], map(rec, l[1:]), generation_probability=0.0)
		elif sytle is 'scheme':
			pass #TODO: Add this scheme functionality -- basically differnet handling of lambda bound variables
			
	else: # for non-list
		return l


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
if __name__ == '__main__':

	test1 = "(defun factorial (x) (if (= x 0) 1 (* x (factorial (- x 1)))))"
	test2 = "(lambda $0 e (and (day_arrival $0 thursday:da) (to $0 baltimore:ci) (< (arrival_time $0) 900:ti) (during_day $0 morning:pd) (exists $1 (and (airport $1) (from $0 $1)))))"
	
	y = simpleLambda2List(test1)
	#print y
	assert str(y) == str(['defun', 'factorial', ['x'], ['if', ['=', 'x', '0'], '1', ['*', 'x', ['factorial', ['-', 'x', '1']]]]])
	
	x = simpleLambda2List(test2)
	#print x
	assert str(x) == str(['lambda', '$0', 'e', ['and', ['day_arrival', '$0', 'thursday:da'], ['to', '$0', 'baltimore:ci'], ['<', ['arrival_time', '$0'], '900:ti'], ['during_day', '$0', 'morning:pd'], ['exists', '$1', ['and', ['airport', '$1'], ['from', '$0', '$1']]]]])
	
	print list2FunctionNode(x)
	