# cython: boundscheck=False, wraparound=False, cdivision=True
# prim/baillie_psw_cython.pyx

import cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint64_t, int64_t
from libc.stdbool cimport bool

# Externe C-Funktionen deklarieren
cdef extern from "baillie_psw_complete.h":
    bool baillie_psw_complete(uint64_t n) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
def baillie_psw_fast(uint64_t n) -> bool:
    """Cython-optimierte Version des Baillie-PSW Tests."""
    cdef bool result
    with nogil:
        result = baillie_psw_complete(n)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def baillie_psw_array(cnp.uint64_t[:] numbers):
    """
    Teste Array von Zahlen mit maximaler Performance.
    Kein Python-Overhead während der Berechnung.
    """
    cdef int n_count = len(numbers)
    cdef cnp.uint8_t[:] results = np.zeros(n_count, dtype=np.uint8)
    cdef int i
    
    with nogil:
        for i in range(n_count):
            results[i] = 1 if baillie_psw_complete(numbers[i]) else 0
    
    return np.asarray(results, dtype=bool)

@cython.boundscheck(False)
@cython.wraparound(False)
def find_primes_in_range(uint64_t start, uint64_t end):
    """
    Finde alle Primzahlen in einem Bereich mit maximaler Geschwindigkeit.
    Gibt numpy array der gefundenen Primzahlen zurück.
    """
    cdef list primes = []
    cdef uint64_t n
    
    # Optimierung: Beginne bei ungerader Zahl
    if start == 2:
        primes.append(2)
        start = 3
    elif start % 2 == 0:
        start += 1
    
    with nogil:
        for n in range(start, end + 1, 2):
            if baillie_psw_complete(n):
                with gil:
                    primes.append(n)
    
    return np.array(primes, dtype=np.uint64)

def count_primes_up_to(uint64_t limit):
    """
    Zähle Primzahlen bis zu einem Limit.
    Speicher-effizient, da nur Counter zurückgegeben wird.
    """
    cdef uint64_t count = 0
    cdef uint64_t n
    
    if limit >= 2:
        count = 1  # Zähle 2
        
        with nogil:
            for n in range(3, limit + 1, 2):
                if baillie_psw_complete(n):
                    count += 1
    
    return count
