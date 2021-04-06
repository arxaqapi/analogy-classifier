import sys
import numpy as np

def rnd(n):
    return np.around(np.array(n), 2)


def print_err(x):
    print(x, file=sys.stderr)