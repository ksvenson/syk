import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

import physics.SYK_Full as fsyk
from utility import utils

FIG_SAVE_OPTS = {'bbox_inches': 'tight', 'dpi': 500}

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache/')
FIGS_DIR = './figs'
CACHE_SWITCH = True


@utils.cache('npy', CACHE_SWITCH, CACHE_DIR + 'SYK4_SYK2')
def SYK4_SYK2(samples, N, g, note=None):
    return np.array([fsyk.Hamil(N, 4) + (g/N) * fsyk.Hamil(N, 2) for _ in range(samples)])

@utils.cache('npy', CACHE_SWITCH, CACHE_DIR + 'wormhole')
def wormhole(samples, N, q, mu, note=None):
    return np.array([fsyk.wormhole(N, q, mu) for _ in range(samples)])

@utils.cache('npz', CACHE_SWITCH, CACHE_DIR + 'eigensystem')
def eigensystem(H, note=None):
    all_evals = []
    all_evecs = []
    for trial_H in H:
        evals, evecs = sp.linalg.eigh(trial_H)
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}

@utils.cache('npy', CACHE_SWITCH, CACHE_DIR + 'entropy')
def entropy(evecs, size, note=None):
    M = np.transpose(evecs, axes=(0, 2, 1)).reshape(evecs.shape[0], evecs.shape[1], 2**size, -1)
    U, s, Vh = np.linalg.svd(M)
    probs = s**2
    return -1 * np.sum(probs * np.log(probs), axis=-1)

if __name__ == "__main__":
    np.random.seed(408139579)
    binned = False

    q = 8
    # number of samples
    s = 10
    # number of majoranas
    N = 20
    # size of entanglement partition in qubits
    size = 2

    # model = 'SYK4_SYK2'
    model = 'wormhole'
    # params = (0, 3, 10, 50, 500)
    params = (0.001, 0.01)
    # params = (1, 10, 100, 1000)

    fig, ax = plt.subplots()
    for param in params:
        note = f'q{q}_s{s}_N{N}_p{param}'
        # H = SYK4_SYK2(s, N, param, note=note)
        H = wormhole(s, N//2, 8, param, note=note)

        eigs = eigensystem(H, note=f'{model}_{note}')
        ent = entropy(eigs['evecs'], size, note=f'{model}_{note}_size{size}')

        eng_slope = 1 / (np.max(eigs['evals']) - np.min(eigs['evals']))

        norm_eng = (eng_slope*(eigs['evals'] - np.min(eigs['evals'])) - 0.5).flatten()
        ent_den = ent.flatten() / size

        if binned:
            bins = 200
            mid_bins = np.linspace(-0.5, 0.5, num=bins+1)[:-1] + (1/(2*bins))
            avg_ent_den, _, _ = sp.stats.binned_statistic(norm_eng, ent_den, statistic='mean', bins=bins, range=(-0.5, 0.5))
            
            ax.plot(mid_bins, avg_ent_den, label=rf'$\mu={param}$')
        else:
            ax.scatter(norm_eng, ent_den, s=2, label=rf'$\mu={param}$')
        print(f'Finished p={param}')

    ax.set(xlabel=r'Energy (Normalized to [-0.5, 0.5])', ylabel=f'{size}-qubit EE', yscale='log')
    ax.legend()
    ax.set(title=rf'{model}: $q={q}$, $N={N}$')
    fname = f'ent_{model}_q{q}_s{s}_N{N}_size{size}.png'
    if binned:
        fname = 'binned_' + fname
    fig.savefig(os.path.join(FIGS_DIR, fname), **FIG_SAVE_OPTS)
