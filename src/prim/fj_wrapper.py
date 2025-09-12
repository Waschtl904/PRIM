# -*- coding: utf-8 -*-
"""
Forisek-Jancina Test Wrapper
"""

from .core.forisek_jancina import forisek_jancina_test as _forisek_jancina_test


def forisek_jancina_test(n: int) -> bool:
    """Forisek-Jancina Primzahltest wrapper"""
    return _forisek_jancina_test(n)


# Alias für Kompatibilität
forisek_jancina_test_wrapper = forisek_jancina_test


__all__ = ["forisek_jancina_test", "forisek_jancina_test_wrapper"]
