# -*- coding: utf-8 -*-
"""
Optimized Cython Wrapper (Fallback zu Python wenn Cython nicht verfügbar)
"""

try:
    # Versuche Cython-Version zu importieren (falls vorhanden)
    from .core.optimized_cython import optimized_primality_test

    HAVE_CYTHON = True
except ImportError:
    # Fallback zu Miller-Rabin Python Implementation
    from .core.forisek_jancina import forisek_jancina_test as optimized_primality_test

    HAVE_CYTHON = False


def optimized_primality_test_wrapper(n: int) -> bool:
    """
    Optimized primality test wrapper
    Uses Cython if available, otherwise falls back to Python
    """
    return optimized_primality_test(n)


__all__ = ["optimized_primality_test_wrapper", "optimized_primality_test"]
