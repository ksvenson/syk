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


#Define Majorana operators
def majorana(idx, N):
    if idx % 2:
        bndry = sigmay()
    else:
        bndry = sigmax()
    return tensor([sigmaz()] * (idx//2) + [bndry] + [qeye(2)] * (N//2 - idx//2 - 1))

def Hamil(N, q, return_full=True):
    """
    N: number of majoranas
    """
    comb = combinations(np.arange(N), q)
    hyperedges = tuple([i for i in comb])
    
    # Use variance with convention J=1
    couplings = (1j)**(q/2) * np.sqrt(2**(q-1) * factorial(q-1) / (q * N**(q-1))) * np.random.randn(len(hyperedges))

    # Create a dictionary to map a hyperedge to the random coupling
    # factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i, N) for i in range(N)]

    H = 0
    for i, m_idxs in enumerate(hyperedges):
        temp = couplings[i]
        for j in m_idxs:
            temp = temp * majs[j]
        H = H + temp
    
    return H.full()

def wormhole(N, q, mu):
    assert q % 2 == 0

    comb = combinations(np.arange(N), q)
    hyperedges = tuple(i for i in comb)
    couplingsL = (1j)**(q/2) * np.sqrt(2**(q-1) * factorial(q-1) / (q * N**(q-1))) * np.random.randn(len(hyperedges))
    couplingsR = (-1)**(q/2) * couplingsL

    majs = [majorana(i, 2*N) for i in range(2*N)]

    HL = 0
    HR = 0
    for i, m_idxs in enumerate(hyperedges):
        tempL = couplingsL[i]
        tempR = couplingsR[i]
        for j in m_idxs:
            tempL = tempL * majs[j]
            tempR = tempR * majs[j + N]
        HL = HL + tempL
        HR = HR + tempR
    
    Hint = 0
    for i in range(N):
        Hint = Hint + (majs[i] * majs[i + N])
    Hint = Hint * 1j * mu

    return (HL + HR + Hint).full()
