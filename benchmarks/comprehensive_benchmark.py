import time
from typing import List, Callable
import os
import subprocess
import json


class PrimalityBenchmark:
    def __init__(self):
        self.algorithms = {}
        self.results = {}

    def register_algorithm(self, name: str, func: Callable):
        self.algorithms[name] = func

    def benchmark_algorithm(
        self, algorithm: str, numbers: List[int], iterations: int = 1
    ) -> float:
        func = self.algorithms[algorithm]
        start_time = time.perf_counter()
        for _ in range(iterations):
            for n in numbers:
                try:
                    func(n)
                except Exception as e:
                    print(f"Error in {algorithm} with {n}: {e}")
        end_time = time.perf_counter()
        return (end_time - start_time) / iterations

    def run_comprehensive_benchmark(self):
        test_ranges = [
            ("Small (10^3)", list(range(1000, 2000, 10))),
            ("Medium (10^6)", list(range(1000000, 1001000, 100))),
            ("Large (10^7)", list(range(10000000, 10001000, 1000))),
        ]
        results = {}
        for range_name, test_numbers in test_ranges:
            print(f"\nBenchmarking {range_name}...")
            results[range_name] = {}
            for algo_name in self.algorithms:
                try:
                    subset = test_numbers[:50]
                    benchmark_time = self.benchmark_algorithm(algo_name, subset)
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
    """
    Wrapper für den Hybrid-Router.
    Da das Skript aus dem Projekt-Root aufgerufen wird, liegt die EXE hier:
      modul10_hybrid_router/hybrid_test.exe
    """
    exe_path = os.path.normpath(
        os.path.join(os.getcwd(), "modul10_hybrid_router", "hybrid_test.exe")
    )
    proc = subprocess.run([exe_path, str(n)], capture_output=True, text=True)
    return "PRIME" in proc.stdout


def main():
    print("Starting Comprehensive PRIM Benchmark Suite")
    print("=" * 60)

    benchmark = PrimalityBenchmark()

    # Naive Trial Division
    benchmark.register_algorithm("Naive Trial Division", naive_trial_division)

    # Miller-Rabin placeholder (Python)
    try:
        from prim.optimized_cython import optimized_primality_test

        benchmark.register_algorithm("Miller-Rabin (Python)", optimized_primality_test)
    except ImportError:
        print("optimized_cython nicht verfügbar")

    # Forisek-Jancina via subprocess wrapper
    try:
        from prim.fj_wrapper import forisek_jancina_test as fj_py

        benchmark.register_algorithm("Forisek-Jancina (Python)", fj_py)
    except ImportError:
        print("Forisek-Jancina Python-Wrapper nicht verfügbar")

    # Baillie-PSW via subprocess wrapper
    try:
        from prim.baillie_psw_wrapper import baillie_psw

        benchmark.register_algorithm("Baillie-PSW (MR det.)", baillie_psw)
    except ImportError:
        print("Baillie-PSW Modul nicht verfügbar")

    # Hybrid Test
    benchmark.register_algorithm("Hybrid Test", hybrid_test_wrapper)

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
                print(f"  {algo}: {metrics['error']}")


if __name__ == "__main__":
    main()
