"""
Analysis and benchmarking tools.
"""

from .benchmarks import run_benchmark
from .prime_gap import analyze_prime_gaps

__all__ = [
    "run_benchmark",
    "analyze_prime_gaps",
]
