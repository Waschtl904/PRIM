import sys
import os
import time
from typing import List, Callable
import subprocess
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class PrimalityBenchmark:
    def __init__(self):
        self.algorithms = {}

    def register_algorithm(self, name: str, func: Callable):
        self.algorithms[name] = func

    def benchmark_algorithm(
        self, algorithm: str, numbers: List[int], iterations: int = 5
    ) -> float:
        func = self.algorithms[algorithm]
        start_time = time.perf_counter()
        for _ in range(iterations):
            for n in numbers:
                func(n)
        end_time = time.perf_counter()
        return (end_time - start_time) / iterations

    def run_comprehensive_benchmark(self):
        test_ranges = [
            ("Small (10^5)", list(range(100_000, 101_000, 20))),
            ("Medium (10^7)", list(range(10_000_000, 10_010_000, 200))),
            ("Large (10^9)", list(range(1_000_000_000, 1_000_001_000, 2_000))),
        ]

        results = {}
        for range_name, test_numbers in test_ranges:
            print(f"\nBenchmarking {range_name}...")
            results[range_name] = {}
            subset = test_numbers[:1000]  # größeres Subset für stabilere Messung
            for algo_name in self.algorithms:
                try:
                    benchmark_time = self.benchmark_algorithm(
                        algo_name, subset, iterations=10
                    )
                    numbers_per_second = len(subset) / benchmark_time
                    results[range_name][algo_name] = {
                        "time_seconds": benchmark_time,
                        "numbers_per_second": numbers_per_second,
                    }
                    print(f"  {algo_name}: {numbers_per_second:.2f} numbers/sec")
                except Exception as e:
                    print(f"  {algo_name}: ERROR - {e}")
                    results[range_name][algo_name] = {"error": str(e)}
        return results

    def save_results(self, results: dict, filename: str = "benchmark_results.json"):
        filepath = os.path.join("benchmarks", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")


def naive_trial_division(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def hybrid_test_wrapper(n: int) -> bool:
    exe_path = os.path.normpath(
        os.path.join(os.getcwd(), "modul10_hybrid_router", "hybrid_test.exe")
    )
    proc = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
    return "PRIME" in proc.stdout.upper()


def main():
    print("Starting Comprehensive PRIM Benchmark Suite")
    print("=" * 60)

    benchmark = PrimalityBenchmark()

    # 1. Naive Trial Division
    benchmark.register_algorithm("Naive Trial Division", naive_trial_division)

    # 2. Miller-Rabin (Cython/Python)
    try:
        from prim.optimized_cython_wrapper import optimized_primality_test_wrapper

        benchmark.register_algorithm(
            "Miller-Rabin (Cython)", optimized_primality_test_wrapper
        )
        print("✓ Cython Miller-Rabin geladen")
    except ImportError:
        try:
            from prim.optimized_cython import optimized_primality_test

            benchmark.register_algorithm(
                "Miller-Rabin (Python)", optimized_primality_test
            )
            print("✓ Python Miller-Rabin Fallback geladen")
        except ImportError:
            print("✗ Miller-Rabin nicht verfügbar")

    # 3. Forisek-Jancina (C)
    try:
        from prim.fj_wrapper import forisek_jancina_test

        # Validierung
        assert forisek_jancina_test(97), "Forisek-Jancina-Check fehlgeschlagen"
        benchmark.register_algorithm("Forisek-Jancina (C)", forisek_jancina_test)
        print("✓ Forisek-Jancina C-Implementation geladen")
    except Exception as e:
        print(f"✗ Forisek-Jancina Fehler: {e}")

    # 4. Baillie-PSW (C)
    try:
        from prim.baillie_psw_wrapper import baillie_psw

        # Validierung
        assert baillie_psw(97), "Baillie-PSW-Check fehlgeschlagen"
        benchmark.register_algorithm("Baillie-PSW (C)", baillie_psw)
        print("✓ Baillie-PSW C-Implementation geladen")
    except Exception as e:
        print(f"✗ Baillie-PSW Fehler: {e}")

    # 5. Hybrid Test
    benchmark.register_algorithm("Hybrid Test", hybrid_test_wrapper)
    print("✓ Hybrid Test geladen")

    # Validierung aller Implementierungen
    sample_primes = [2, 3, 5, 97, 7919, 3303820981600721647]
    for name, func in benchmark.algorithms.items():
        for p in sample_primes:
            # Skip naive und MR für sehr große Zahlen
            if name in ("Naive Trial Division", "Miller-Rabin (Cython)") and p > 2**32:
                continue
            if not func(p):
                raise RuntimeError(f"{name} gibt für {p} False zurück!")
    print("✓ Alle Implementierungen validiert")

    print(f"\nRegistrierte Algorithmen: {len(benchmark.algorithms)}")
    for name in benchmark.algorithms:
        print(f"  - {name}")

    # Durchführen
    results = benchmark.run_comprehensive_benchmark()
    benchmark.save_results(results)

    print("\n" + "=" * 60)
    print("Benchmark Summary:")
    for range_name, range_results in results.items():
        print(f"\n{range_name}:")
        for algo, metrics in range_results.items():
            if "error" not in metrics:
                print(f"  {algo}: {metrics['numbers_per_second']:.2f} numbers/sec")
            else:
                print(f"  {algo}: ERROR - {metrics['error']}")


if __name__ == "__main__":
    main()
