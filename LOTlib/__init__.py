
#=============================================================================================================
# Store a version number
# We'll try to follow semantic versioning: http://semver.org/
#=============================================================================================================

LOTlib_VERSION = "0.7.0" # major; minor; patch

#=============================================================================================================
# This allows us to use the variable SIG_INTERRUPTED inside loops etc
# to catch breaks.
# import LOTlib
# if LOTlib.SIG_INTERRUPTED: ...
#=============================================================================================================

import signal
import sys

SIG_INTERRUPTED = False
def signal_handler(signal, frame):
    global SIG_INTERRUPTED
    SIG_INTERRUPTED = True
    print >>sys.stderr, "# Signal %s caught."%signal

# Handle interrupt CTRL-C
signal.signal(signal.SIGINT, signal_handler)


def break_ctrlc(g, reset=False, multi_break=False):
    """Easy way to ctrl-C out of a loop.

    reset -- when we get here, should we pretend like ctrl-c was never pressed?

    Lets you wrap a generater, rather than have to write "if LOTlib.SIG_INTERRUPTED..."

    """
    import LOTlib # WOW, this is weird scoping, but it doesn't work if you treat this as a local variable (you can't from LOTlib import break_ctrlc)

    if reset:
        LOTlib.SIG_INTERRUPTED = False

    for x in g:
        #global SIG_INTERRUPTED
        if LOTlib.SIG_INTERRUPTED:

            # reset if we should
            if not multi_break: SIG_INTERRUPTED = False

            # and break
            break
        else:

            yield x
