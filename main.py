import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import SYK_Full as fsyk
import Sparse_SYK as ssyk

if __name__ == "__main__":
    seed = 408139579

    N = 20
    q = 4

    full_H = fsyk.Hamil(N, q, seed)
    sparse_H = ssyk.sparse_H(N, )
    print(H.shape)

    evals = H.eigenenergies()
    np.save(f'full_syk_evals_N{N}_q{q}.npy', evals)

    fig, ax = plt.subplots()

    ax.hist(evals, bins=100)
    ax.set(xlabel='Energy', ylabel='Counts', title=rf'$N={N}$, $q={q}$, Full SYK Spectrum')

    fig.savefig('spectrum.svg', bbox_inches='tight')
