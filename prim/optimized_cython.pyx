# prim/optimized_cython.pyx
import cython
from libc.stdlib cimport rand, srand
from libc.time cimport time

@cython.boundscheck(False)
@cython.wraparound(False)
def optimized_primality_test(long long n):
    """
    Cython-optimierte deterministische Miller-Rabin für 64-Bit-Zahlen.
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Deterministische Basen für 64-Bit-Zahlen
    cdef long long bases[7]
    bases[0] = 2
    bases[1] = 325
    bases[2] = 9375
    bases[3] = 28178
    bases[4] = 450775
    bases[5] = 9780504
    bases[6] = 1795265022

    # Schreibe n-1 = d * 2^s
    cdef long long d = n - 1
    cdef int s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1

    cdef int i, r
    cdef long long a, x

    for i in range(7):
        a = bases[i]
        if a % n == 0:
            continue

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for r in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False

    return True
