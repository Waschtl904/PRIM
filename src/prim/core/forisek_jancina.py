# -*- coding: utf-8 -*-
"""
Forisek-Jancina Primzahltest
"""


def is_prime_forisek_jancina(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    for p in range(2, int(n**0.5) + 1):
        if n % p == 0:
            return False
    return True
