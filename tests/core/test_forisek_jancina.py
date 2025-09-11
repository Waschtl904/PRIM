# tests/test_forisek_jancina.py

from prim.fj_wrapper import forisek_jancina_test


def test_forisek_jancina_small():
    assert forisek_jancina_test(29)
