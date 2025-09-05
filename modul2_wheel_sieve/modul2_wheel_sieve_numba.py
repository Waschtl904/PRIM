import numpy as np
import numba as nb
from math import sqrt

# Wheel-30 Konstanten
WHEEL30_SIZE = 30
WHEEL30_COPRIME = np.array([1, 7, 11, 13, 17, 19, 23, 29], dtype=np.int32)


@nb.njit(cache=True)
def simple_sieve(limit: int):
    if limit < 2:
        return np.empty(0, np.int32)
    is_prime = np.ones(limit, np.bool_)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit, i):
                is_prime[j] = False
    return np.nonzero(is_prime)[0].astype(np.int32)


@nb.njit(cache=True)
def wheel30_segmented_sieve(limit: int, segment_size: int = 32768):
    if limit < 2:
        return np.empty(0, np.int32)
    # Basis-Primzahlen
    root = int(sqrt(limit)) + 1
    base_primes = simple_sieve(root)

    # Schätzen Sie Gesamtzahl der Primzahlen (~limit / log limit)
    est = int(limit / np.log(limit) * 1.2)
    out = np.empty(est, np.int32)
    count = 0

    # Füge 2,3,5 hinzu
    if 2 < limit:
        out[count] = 2
        count += 1
    if 3 < limit:
        out[count] = 3
        count += 1
    if 5 < limit:
        out[count] = 5
        count += 1

    # Segmentierte Wheel-30-Siebung
    for low in range(7, limit, segment_size):
        high = min(low + segment_size, limit)
        # Wheel-Kandidaten im Segment
        size = (high - low) * len(WHEEL30_COPRIME) // WHEEL30_SIZE + 8
        cand = np.empty(size, np.int32)
        m = 0
        for k in range(low // WHEEL30_SIZE, high // WHEEL30_SIZE + 1):
            base = k * WHEEL30_SIZE
            for r in WHEEL30_COPRIME:
                v = base + r
                if v < low or v >= high:
                    continue
                cand[m] = v
                m += 1
        cand = cand[:m]

        # Markierung für Kandidaten
        mask = np.ones(m, np.bool_)
        for p in base_primes:
            if p * p > high:
                break
            if p <= 5:
                continue
            for i in range(m):
                if mask[i] and cand[i] % p == 0:
                    mask[i] = False

        # Speichern
        for i in range(m):
            if mask[i]:
                out[count] = cand[i]
                count += 1

    # Rückgabe korrekter Größe
    return out[:count]
