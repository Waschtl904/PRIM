import pytest

from prim.hybrid_wrapper import is_prime, batch_primality_test


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, True),
        (3, True),
        (561, False),
        (4294967295, False),  # 2^32 − 1 (Mersenne-Kompositum)
        (4294967297, False),  # 2^32 + 1 (Fermat-Kompositum)
    ],
)
def test_individual(n, expected):
    assert is_prime(n) is expected


def test_batch():
    nums = [2, 561, 4294967297]
    expected = [True, False, False]
    assert batch_primality_test(nums) == expected
