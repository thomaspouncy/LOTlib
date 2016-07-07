from FormalLanguage import FormalLanguage
from LOTlib.Grammar import Grammar
from pickle import load

class SimpleEnglish(FormalLanguage):
    """
    A simple English language with a few kinds of recursion all at once
    """
    def __init__(self, max_length=6):
        self.grammar = Grammar(start='S')
        self.grammar.add_rule('S', 'S', ['NP', 'VP'], 1.0)
        self.grammar.add_rule('NP', 'NP', ['d', 'AP', 'n'], 1.0)
        self.grammar.add_rule('AP', 'AP', ['a', 'AP'], 1.0)
        self.grammar.add_rule('AP', 'AP', None, 1.0)

        self.grammar.add_rule('VP', 'VP', ['v'], 1.0)
        self.grammar.add_rule('VP', 'VP', ['v', 'NP'], 1.0)
        self.grammar.add_rule('VP', 'VP', ['v', 't', 'S'], 1.0)
        self.grammar.add_rule('S', 'S', ['i','S','h','S'], 0.5) # add if S then S grammar

        FormalLanguage.__init__(self, max_length)

    def all_strings(self, max_length):
        """
            numerating strings is different here: we saved a sorted list of strings here, we load it and return the first
            max_length shortest strings.

            NOTE:
                max_length is not len of strings here, but the index of the sorted list
        """
        return load(open('./all_strings'))[:max_length]

#        for x in self.grammar.enumerate(d=max_length):
#            if LOTlib.SIG_INTERRUPTED: break
#            print x
#           # s = ''.join(x.all_leaves())
#           # if len(s) < max_length:
#           #     yield s


# just for testing
if __name__ == '__main__':
    language = SimpleEnglish(8)

    # for _ in xrange(100):
    #     x =  language.grammar.generate()
    #     print list(x.all_leaves())

    # for t in language.grammar.enumerate():
    #     print t

    # for e in language.all_strings():
    #     print e

    # for e in language.all_strings(max_length=8):
    #     print e

    print language.sample_data_as_FuncData(300)

    # print language.is_valid_string('PV')
    # print language.is_valid_string('DAANV')
    # print language.is_valid_string('PVTPV')
    # print language.is_valid_string('PVDN')
    # print language.is_valid_string('DNVDDN')
