import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

import physics.SYK_Full as fsyk
from utility import utils

FIG_SAVE_OPTS = {'bbox_inches': 'tight'}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache/')
FIGS_DIR = './figs'


@utils.cache('npy', CACHE_DIR + 'SYK4_SYK2')
def SYK4_SYK2(samples, N, g, note=None):
    return np.array([fsyk.Hamil(N, 4, note=None) + (g/N) * fsyk.Hamil(N, 2, note=None) for _ in range(samples)])

@utils.cache('npz', CACHE_DIR + 'eigensystem')
def eigensystem(H, note=None):
    all_evals = []
    all_evecs = []
    for trial_H in H:
        evals, evecs = sp.linalg.eigh(trial_H)
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}

@utils.cache('npy', CACHE_DIR + 'entropy')
def entropy(evecs, size, note=None):
    M = np.transpose(evecs, axes=(0, 2, 1)).reshape(evecs.shape[0], evecs.shape[1], 2**size, -1)
    U, s, Vh = np.linalg.svd(M)
    probs = s**2
    return -1 * np.sum(probs * np.log(probs), axis=-1)

if __name__ == "__main__":
    np.random.seed(408139579)

    # number of samples
    s = 10
    # number of majoranas
    N = 20
    # size of entanglement partition in qubits
    size = N // 4

    fig, ax = plt.subplots()
    for g in (0, 3, 10, 50, 500):
        H = SYK4_SYK2(samples, N, g, note=f's{s}_N{N}_g{g}')
        eigs = eigensystem(H, note=f'SYK4_SYK2_s{s}_N{N}_g{g}')

        ent = entropy(eigs['evecs'], size)

        eng_slope = 1 / (np.max(eigs['evals']) - np.min(eigs['evals']))
        ax.scatter((eng_slope*(eigs['evals'] - np.min(eigs['evals'])) - 0.5).flatten(), ent.flatten() / size, s=5, label=rf'$g={g}$')
        print(f'Finished g={g}')

    ax.set(xlabel=r'Energy ($\mathcal{J}$)', ylabel='Entropy')
    fig.legend()
    fig.savefig(os.path.join(FIGS_DIR, f'ent_s{s}_N{N}_g{g}.svg'), **FIG_SAVE_OPTS)
