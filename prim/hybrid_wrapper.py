from typing import Union, List

from prim.baillie_psw_wrapper import baillie_psw as _baillie_psw_ctypes
from prim.fj_wrapper import forisek_jancina_test


def hybrid_test(n: Union[int, str]) -> bool:
    """
    Hybrid-Primzahltest:
    - Für n <= 2^32-1: deterministischer Forisek-Jancina Test
    - Für größere n: probabilistischer Baillie-PSW Test
    """
    if isinstance(n, str):
        n = int(n)
    if n < 0:
        return False

    if n <= 0xFFFFFFFF:
        return forisek_jancina_test(n)
    return _baillie_psw_ctypes(n)


def is_prime(n: Union[int, str]) -> bool:
    """Hauptfunktion für Hybrid-Primzahltest."""
    return hybrid_test(n)


def batch_primality_test(numbers: List[Union[int, str]]) -> List[bool]:
    """Batch-Processing für Hybrid-Test."""
    return [hybrid_test(n) for n in numbers]


# Alias für Kompatibilität
primality_test = is_prime
