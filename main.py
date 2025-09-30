import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import SYK_Full as syk

if __name__ == "__main__":
    seed = 408139579

    N = 20
    q = 4

    H = syk.Hamil(N, q, seed)
    print(H.shape)

    evals = H.eigenenergies()
    np.save(f'full_syk_evals_N{N}_q{q}.npy', evals)

    fig, ax = plt.subplots()

    ax.hist(evals, bins=100)
    ax.set(xlabel='Energy', ylabel='Counts', title=rf'$N={N}$, $q={q}$, Full SYK Spectrum')

    fig.savefig('spectrum.svg', bbox_inches='tight')
