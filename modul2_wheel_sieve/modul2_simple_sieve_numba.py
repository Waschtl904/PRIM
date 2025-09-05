# modul2_simple_sieve_numba.py

import numpy as np
import numba as nb
from math import sqrt


@nb.njit(cache=True)
def simple_sieve(limit: int) -> np.ndarray:
    """
    Eratosthenes-Sieb bis 'limit' (Numba-optimiert).
    Gibt ein np.int32-Array aller Primzahlen < limit zurück.
    """
    if limit < 2:
        return np.empty(0, np.int32)
    is_prime = np.ones(limit, np.bool_)
    is_prime[0] = is_prime[1] = False
    rt = int(sqrt(limit)) + 1
    for i in range(2, rt):
        if is_prime[i]:
            for j in range(i * i, limit, i):
                is_prime[j] = False
    return np.nonzero(is_prime)[0].astype(np.int32)
