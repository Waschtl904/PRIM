# src/prim/fj_wrapper.py

from prim.core.forisek_jancina import forisek_jancina_test

def forisek_jancina_test_wrapper(n: int) -> bool:
    return forisek_jancina_test(n)
