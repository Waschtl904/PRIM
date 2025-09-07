# prim/optimized_cython_wrapper.py
from prim.optimized_cython import optimized_primality_test


def optimized_primality_test_wrapper(n: int) -> bool:
    """
    Wrapper für die Cython-optimierte Miller-Rabin-Funktion.
    Gibt True zurück, wenn n als Primzahl erkannt wird.
    """
    return bool(optimized_primality_test(n))
