"""Your autotuner's search space and rules (Problem 12). a2.py::tune imports this file on your laptop.

space() yields (family, sizes) pairs: family "k6" takes BM, BN, BK, TM, TN; family "k7" takes BM, BN, BK, WM, WN,
WNITER, TM, TN (see kernels/). check() says whether one configuration keeps every rule: return None if it does, or a
short name for the first rule it breaks. a2.py counts the configurations each rule removed and prints the counts, so
name your rules well; your report needs them.

a2.py also refuses, on its own, sizes that cannot compile or would read outside the matrices, and tells you if your
check() let one through. It does not know about registers: that rule is yours to write.
"""
from itertools import product


def space():
    # TODO: replace this with your search space. One example configuration of each family:
    yield "k6", dict(BM=128, BN=128, BK=8, TM=8, TN=8)
    yield "k7", dict(BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4)


def check(fam, p):
    # TODO: return None if the configuration keeps every rule, else the name of the rule it breaks, for example
    # "threads", "shared memory", "registers", "patches tile the block" ...
    return None
