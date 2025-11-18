#shortcuts https://deepnote.com/docs/keyboard-shortcuts
#qutip tutorials: https://qutip.org/qutip-tutorials/
# https://qutip.readthedocs.io/en/qutip-5.0.x/
import os
from itertools import combinations
import itertools
import numpy as np
from scipy.special import binom
from math import factorial, comb
import time
from random import sample, seed
import datetime
from concurrent.futures import ProcessPoolExecutor
import functools as ft
from qutip import *
#from mpi4py import MPI
import scipy as sp
#import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import linalg

from utility import utils

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache/')

#Define Majorana operators
def majorana(idx, L):
    parity = idx % 2 #this checks th parity of the index
    if parity:
        bndry = sigmay() #odd indices have sigmay as a boundary term
    else:
        bndry = sigmax() #even indices have sigmax as a boundary term
    b_idx = idx//2 #this gives the index on the spin chain
    if b_idx > 0:
        tensor_list = [] #this builds a tensor list of the paulis which construct the majorana
        for k in np.arange(L):
            if k < b_idx:
                tensor_list.append(sigmaz())
            elif k == b_idx:
                tensor_list.append(bndry)
            elif k > b_idx:
                tensor_list.append(qeye(2))
        tensor_list = list(tensor_list)
    elif b_idx == 0:
        tensor_list = [bndry]
        for k in np.arange(L-1):
            tensor_list.append(qeye(2))
    return tensor(tensor_list)

# SYK Hamiltonian for q-body
# @utils.cache('pkl', os.path.join(CACHE_DIR + 'fsyk'))
def Hamil(N, q, note=None):
    """
    N: number of majoranas
    """
    
    comb = combinations(np.arange(N), q)
    hyperedges = tuple([i for i in comb])
    
    # Use variance with convention J=1
    couplings = np.random.randn(len(hyperedges))
    if q == 2:
        couplings *= (1j) / np.sqrt(N)
    elif q == 4:
        couplings *= np.sqrt(6/N**3)
    else:
        raise ValueError(f'q = {q} is not supported')
    # Create a dictionary to map a hyperedge to the random coupling
    factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i,N/2) for i in range(N)]

    H = 0
    for idxs in hyperedges:
        Hk = 1
        for i in idxs:
            Hk = Hk * majs[i]
        Hk = Hk * factor[idxs]
        H = H + Hk
    
    return H.full()

