"""
Modul 3: Hybrid-Workflow "Sieb + FJ32"
=====================================

Kombiniert das schnelle Numba-Sieb aus Modul 2 mit dem deterministischen
FJ32-Test aus Modul 1 für verifizierte Primzahllisten und Benchmark-Vergleiche.

Autor: Sebastian
Datum: September 2025
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import importlib.util

# Projekt-Root definieren und in sys.path einfügen
project_root = r"C:\Users\sebas\Desktop\coding\PRIM"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Dynamisches Laden von primetest.pyd
pyd_path = os.path.join(project_root, "primetest.pyd")
if os.path.exists(pyd_path):
    spec = importlib.util.spec_from_file_location("primetest", pyd_path)  # type: ignore
    primetest = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(primetest)  # type: ignore
    print("✓ primetest.pyd dynamisch geladen")
else:
    print("⚠️ primetest.pyd nicht gefunden, normaler Import folgt")

# Damit Numba das Modul im Cache wiederfindet, Pfad zur Sieb-Implementierung
sieve_folder = os.path.join(project_root, "modul2_wheel_sieve")
if sieve_folder not in sys.path:
    sys.path.insert(0, sieve_folder)

# Import des Numba-Siebs
try:
    import modul2_simple_sieve_numba as sieve_mod

    simple_sieve = sieve_mod.simple_sieve
    print("✓ Numba-Sieb erfolgreich importiert")
    HAS_NUMBA_SIEVE = True
except ImportError as e:
    print(f"⚠️ Warnung: Konnte Numba-Sieb nicht importieren: {e}")
    HAS_NUMBA_SIEVE = False

# Sicherstellen, dass primetest existiert; falls dynamisch nicht geladen, regulär importieren
try:
    primetest  # type: ignore
except NameError:
    import primetest  # type: ignore

# Auswahl der echten FJ32-Funktion aus dem Modul
if hasattr(primetest, "fj32_c"):  # type: ignore
    _test_func = primetest.fj32_c  # type: ignore[attr-defined]
elif hasattr(primetest, "fj_hash_c"):  # type: ignore
    _test_func = primetest.fj_hash_c  # type: ignore[attr-defined]
else:
    raise ImportError("Keine passende Primtest-Funktion in primetest.pyd")

HAS_FJ32 = True
print("✓ FJ32-Test-Funktion ermittelt:", _test_func.__name__)


def fallback_simple_sieve(limit: int) -> np.ndarray:
    """
    Fallback-Implementierung des Eratosthenes-Siebs falls Numba-Version nicht verfügbar
    """
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = False
    return np.where(is_prime)[0].astype(np.int64)


def fallback_is_prime_fj32(n: int) -> bool:
    """
    Fallback-Implementierung eines einfachen Primzahltests falls FJ32 nicht verfügbar
    """
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


class HybridPrimeAnalyzer:
    """
    Hauptklasse für Hybrid-Workflow: Sieb + FJ32-Verifikation
    """

    def __init__(self):
        self.results = []
        self.has_numba_sieve = HAS_NUMBA_SIEVE
        self.has_fj32 = HAS_FJ32

    def get_sieve_function(self):
        return simple_sieve if self.has_numba_sieve else fallback_simple_sieve

    def get_primality_test(self):
        return _test_func if self.has_fj32 else fallback_is_prime_fj32

    def run_sieve_only(self, limit: int) -> Tuple[np.ndarray, float]:
        print(f"🔍 Führe Sieb-Test bis {limit:,} aus...")
        sieve_func = self.get_sieve_function()
        start = time.perf_counter()
        primes = sieve_func(limit)
        elapsed = time.perf_counter() - start
        print(f"✓ Sieb fand {len(primes):,} Primzahlen in {elapsed:.6f}s")
        return primes, elapsed

    def run_fj32_verification(self, primes: np.ndarray) -> Tuple[int, int, float]:
        print(f"🔍 Verifiziere {len(primes):,} Primzahlen mit FJ32...")
        test_func = self.get_primality_test()
        verified = 0
        failed = 0
        start = time.perf_counter()
        for p in primes:
            if test_func(int(p)):
                verified += 1
            else:
                failed += 1
                if failed <= 10:
                    print(f"⚠️ FJ32 widersprach Sieb bei {p}")
        elapsed = time.perf_counter() - start
        print(
            f"✓ Verifikation: {verified:,} bestätigt, {failed:,} Widersprüche in {elapsed:.6f}s"
        )
        return verified, failed, elapsed

    def run_fj32_only(self, limit: int) -> Tuple[List[int], float]:
        print(f"🔍 Führe reinen FJ32-Test bis {limit:,} aus...")
        test_func = self.get_primality_test()
        primes = []
        start = time.perf_counter()
        for n in range(2, limit):
            if test_func(n):
                primes.append(n)
        elapsed = time.perf_counter() - start
        print(f"✓ FJ32 fand {len(primes):,} Primzahlen in {elapsed:.6f}s")
        return primes, elapsed

    def run_hybrid_workflow(self, limit: int) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"🚀 HYBRID-WORKFLOW für N = {limit:,}")
        print(f"{'='*60}")
        sieve_primes, t_sieve = self.run_sieve_only(limit)
        v, f, t_verify = self.run_fj32_verification(sieve_primes)
        if limit <= 100_000:
            _, t_fj32 = self.run_fj32_only(limit)
        else:
            print(f"⏭️ Überspringe reinen FJ32-Test (N={limit:,} zu groß)")
            t_fj32 = float("inf")
        total = t_sieve + t_verify
        result = {
            "limit": limit,
            "sieve_count": len(sieve_primes),
            "sieve_time": t_sieve,
            "verified": v,
            "failed": f,
            "verify_time": t_verify,
            "hybrid_time": total,
            "fj32_time": t_fj32,
            "has_fj32": self.has_fj32,
        }
        if t_fj32 != float("inf"):
            result["speedup"] = t_fj32 / total
            print(
                "📊 PERFORMANCE-ANALYSE:\n"
                f"   Sieb: {t_sieve:.6f}s, Verif.: {t_verify:.6f}s, Hybrid: {total:.6f}s, "
                f"FJ32: {t_fj32:.6f}s, Speedup: {result['speedup']:.2f}x"
            )
        self.results.append(result)
        return result

    def benchmark_multiple_limits(self, limits: List[int]) -> pd.DataFrame:
        print(f"{'='*60}\n📊 MULTI-LIMIT BENCHMARK\nTeste Limits: {limits}\n{'='*60}")
        for L in limits:
            self.run_hybrid_workflow(L)
        return pd.DataFrame(self.results)

    def create_performance_plot(self, df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.plot(df["limit"], df["sieve_time"], "b-o", label="Sieb allein")
        ax1.plot(df["limit"], df["hybrid_time"], "g-s", label="Hybrid")
        valid = df[df["fj32_time"] != float("inf")]
        if not valid.empty:
            ax1.plot(valid["limit"], valid["fj32_time"], "r-^", label="FJ32 allein")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_title("Laufzeit-Vergleich")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df["limit"], df["sieve_count"], "m-D")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_title("Anzahl Primzahlen")
        ax2.grid(True)

        if "speedup" in df:
            ax3.plot(df["limit"], df["speedup"], "c-o")
            ax3.set_xscale("log")
            ax3.set_title("Speedup Hybrid vs. FJ32")
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "Keine Speedup-Daten", ha="center", va="center")
            ax3.set_title("Speedup")

        success = (df["verified"] / df["sieve_count"]) * 100
        ax4.plot(df["limit"], success, "g-o")
        ax4.set_xscale("log")
        ax4.set_ylim(95, 101)
        ax4.set_title("Verifikations-Erfolg (%)")
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig("modul3_hybrid_performance.png", dpi=300)
        plt.show()


def main():
    print("🚀 MODUL 3: HYBRID-WORKFLOW 'SIEB + FJ32'")
    analyzer = HybridPrimeAnalyzer()
    test_limits = [1_000, 10_000, 100_000, 1_000_000]
    df = analyzer.benchmark_multiple_limits(test_limits)
    print("\n📋 ZUSAMMENFASSUNG:")
    print(
        df[["limit", "sieve_count", "sieve_time", "hybrid_time", "has_fj32"]].to_string(
            index=False
        )
    )
    analyzer.create_performance_plot(df)
    df.to_csv("modul3_hybrid_results.csv", index=False)
    print("💾 Ergebnisse gespeichert in modul3_hybrid_results.csv")
    return analyzer, df


if __name__ == "__main__":
    main()
