# -*- coding: utf-8 -*-
"""
PRIM - Modulares Framework für Primzahltests und -analysen
=========================================================

Dieses Package stellt verschiedene Algorithmen für Primzahltests zur Verfügung.
"""

from ._version import __version__

# Core algorithms
from .core.forisek_jancina import forisek_jancina_test, is_prime_forisek_jancina

try:
    from .core.baillie_psw import baillie_psw
except ImportError:
    # Fallback wenn baillie_psw nicht verfügbar
    baillie_psw = None

try:
    from .core.lucas_lehmer import is_mersenne_prime
except ImportError:
    is_mersenne_prime = None

# Wrapper functions
from .fj_wrapper import forisek_jancina_test

try:
    from .baillie_psw_wrapper import baillie_psw
except ImportError:
    pass

from .hybrid_wrapper import is_prime, batch_primality_test

__all__ = [
    "__version__",
    # Core functions
    "forisek_jancina_test",
    "is_prime_forisek_jancina",
    "baillie_psw",
    "is_mersenne_prime",
    # Wrapper functions
    "is_prime",
    "batch_primality_test",
]
