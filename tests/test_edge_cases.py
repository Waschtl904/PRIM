import pytest
from prim.core.forisek_jancina import is_prime_forisek_jancina as is_prime
from prim.core.lucas_lehmer import is_mersenne_prime


@pytest.mark.parametrize(
    "n,expected",
    [
        ((2**31 - 1) * (2**19 - 1), False),
        (561, False),
        (0, False),
        (1, False),
        (2, True),
        (3, True),
    ],
)
def test_edge_cases(n, expected):
    assert is_prime(n) is expected


@pytest.mark.slow
def test_large_mersenne():
    # Exponent 61 → prüft 2**61 - 1
    assert is_mersenne_prime(61) is True
