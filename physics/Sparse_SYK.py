#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
from scipy.special import binom
from math import factorial
from dynamite.extras import majorana
from dynamite.operators import op_sum, op_product
from dynamite.states import State
from dynamite import config
from Hypergraph import *

config.initialize(['-mfn_ncv', '10'])

# Sparse SYK Hamiltonian
def sparse_H(N, hyperedges, k, q, random_seed):
    '''
    Build the q-SYK Hamiltonian for a system of N Majoranas with sparsity
    parameter k such that Hamiltonian is a sum of kN terms.
    '''
    # Fix seed to generate random couplings Jijkl
    np.random.seed(random_seed)
    
    # Use variance with convention J=1
    p = k*N/binom(N, q)
    couplings = (1j)**(q/2)*np.sqrt( factorial(q-1) / (p * N**(q-1) * 2**q) )*np.random.randn(len(hyperedges))
    
    # Create a dictionary to map a hyperedge to the random coupling
    factor = dict(zip(hyperedges, couplings))
    
    # Evaluate majoranas before building Hamiltonian
    majs = [majorana(i) for i in range(N)]

    return op_sum(op_product(majs[i] for i in idxs).scale(factor[idxs]) for idxs in hyperedges)
    