def wheel30_segmented_sieve(int limit, int segment_size=32768):
    """
    Kombiniertes Wheel-30-Segment-Sieb.
    Gibt ein numpy-Array aller Primzahlen < limit zurück.
    """
    # Deklarationen am Anfang
    cdef int sqrt_lim, current, end, k, r, base, candidate, nseg, idx, p
    cdef INT32[:] base_primes
    cdef list result
    cdef cnp.ndarray[INT32, ndim=1] seg

    sqrt_lim = <int>sqrt(limit) + 1
    base_primes = simple_sieve(sqrt_lim)
    result = []

    current = 2
    while current < limit:
        end = current + segment_size
        if end > limit:
            end = limit

        # Wheel-30-Kandidaten sammeln
        k = current // WHEEL30_SIZE
        while True:
            base = k * WHEEL30_SIZE
            if base >= end:
                break
            for r in WHEEL30_COPRIME:
                candidate = base + r
                if candidate >= end:
                    break
                if candidate >= current:
                    result.append(candidate)
            k += 1

        # In-place Filter via Basis-Primzahlen
        seg = np.array(result, dtype=np.int32)
        nseg = seg.shape[0]
        for p in base_primes:
            if p <= 5:
                continue
            if p * p > seg[nseg-1]:
                break
            for idx in range(nseg):
                if seg[idx] != 0 and seg[idx] % p == 0:
                    seg[idx] = 0

        # Gefilterte Werte behalten
        result = [x for x in seg.tolist() if x != 0]
        current = end

    # 2,3,5 voranstellen
    cdef int prepend_count = 3
    cdef np.ndarray[INT32, ndim=1] all_primes = np.empty(len(result) + prepend_count, dtype=np.int32)
    all_primes[0] = 2
    all_primes[1] = 3
    all_primes[2] = 5
    for idx in range(len(result)):
        all_primes[idx + prepend_count] = result[idx]

    return all_primes
