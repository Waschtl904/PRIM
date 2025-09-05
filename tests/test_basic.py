import pytest
from modul4_benchmarks import AdvancedBenchmarkAnalyzer


def test_fj32_small_primes():
    analyzer = AdvancedBenchmarkAnalyzer(
        use_numba=False, use_parallel=False, use_cache=False
    )
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        assert analyzer.fj32_fallback(p) is True


def test_sieve_basic():
    analyzer = AdvancedBenchmarkAnalyzer(
        use_numba=True, use_parallel=False, use_cache=True
    )
    primes = analyzer.sieve_numba(30)
    assert primes.tolist() == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
