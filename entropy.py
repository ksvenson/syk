import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

import physics.SYK_Full as fsyk
from utility import utils

FIG_SAVE_OPTS = {'bbox_inches': 'tight'}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache/')
FIGS_DIR = './figs'


# @utils.cache('npy', CACHE_DIR + 'Hsample')
def H_sample(samples, N, q, note=None):
    return np.array([fsyk.Hamil(N, q, note=None) for _ in range(samples)])

# @utils.cache('npz', CACHE_DIR + 'eigensystem')
def eigensystem(H_sample, note=None):
    all_evals = []
    all_evecs = []
    for H in H_sample:
        evals, evecs = sp.linalg.eigh(H)
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}

# @utils.cache('npy', CACHE_DIR + 'entropy')
def entropy(evecs, size, note=None):
    M = np.transpose(evecs, axes=(0, 2, 1)).reshape(evecs.shape[0], evecs.shape[1], 2**size, evecs.shape[1] // 2**size)
    U, s, Vh = np.linalg.svd(M)
    probs = s**2
    return -1 * np.sum(probs * np.log(probs), axis=-1)


if __name__ == "__main__":
    np.random.seed(408139579)

    samples = 10
    N = 16
    q = 4
    size = 2

    H = H_sample(samples, N, q, note=f'N{N}_q{q}')

    eigs = eigensystem(H, note=f'N{N}_q{q}')

    ent = entropy(eigs['evecs'], size)

    fig, ax = plt.subplots()

    ax.scatter(eigs['evals'].flatten(), ent.flatten(), s=10)
    ax.set(xlabel=r'Energy ($J$)', ylabel='Entropy')

    fig.savefig(os.path.join(FIGS_DIR, 'ent.svg'), **FIG_SAVE_OPTS)
