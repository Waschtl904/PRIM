import pytest

from prim.fj_wrapper import forisek_jancina_test as is_prime


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, True),
        (3, True),
        (4, False),
        (561, False),
        (1105, False),  # Carmichael-Zahl
    ],
)
def test_edge_cases(n, expected):
    assert is_prime(n) is expected
