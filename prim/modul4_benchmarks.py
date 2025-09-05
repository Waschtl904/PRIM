#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 4 - Erweiterte Benchmarks & Optimierung
=============================================

Systematische Performance- und Speicheranalyse über Parameter-Grid:
- Obergrenzen N = 10^6, 10^7, 10^8
- Segment-Größen für segmentiertes Sieb
- Numba-Flags (parallel=True, cache=True)
- Metriken: Laufzeit und Speicherverbrauch

Autor: Sebastian
Datum: September 2025
"""

import os
import sys
import time
import tracemalloc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Numba Imports
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warnung: Numba nicht verfügbar. Fallback auf reine Python-Implementierung.")


class AdvancedBenchmarkAnalyzer:
    """
    Erweiterte Benchmark-Klasse für Modul 4
    Erweitert die HybridPrimeAnalyzer um Speicher-Tracking und Parameter-Grid-Analyse
    """

    def __init__(
        self, use_numba: bool = True, use_parallel: bool = False, use_cache: bool = True
    ):
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_parallel = use_parallel
        self.use_cache = use_cache
        self.results = []

        # Verzeichnisse sicherstellen
        Path("data").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)

        # C-Extension laden (falls verfügbar)
        self.fj32_available = False
        try:
            sys.path.insert(0, os.getcwd())
            import primetest  # type: ignore

            self.fj32_c = primetest.fj32_c
            self.fj32_available = True
            print("✓ primetest.pyd (FJ32-C) erfolgreich geladen")
        except ImportError as e:
            print(f"⚠ primetest.pyd nicht verfügbar: {e}")
            print("  Verwende Fallback-Implementierung")

        # Numba-JIT-Funktionen kompilieren
        if self.use_numba:
            self._compile_numba_functions()

    def _compile_numba_functions(self):
        """Kompiliert Numba-JIT-Funktionen mit konfigurierten Flags"""

        # Basis Sieb-Funktion
        if self.use_parallel:

            @jit(nopython=True, parallel=True, cache=self.use_cache)
            def sieve_numba_parallel(limit):
                is_prime = np.ones(limit + 1, dtype=np.bool_)
                is_prime[0] = is_prime[1] = False

                for i in prange(2, int(limit**0.5) + 1):
                    if is_prime[i]:
                        for j in range(i * i, limit + 1, i):
                            is_prime[j] = False

                return np.where(is_prime)[0]

            self.sieve_numba = sieve_numba_parallel
        else:

            @jit(nopython=True, cache=self.use_cache)
            def sieve_numba_serial(limit):
                is_prime = np.ones(limit + 1, dtype=np.bool_)
                is_prime[0] = is_prime[1] = False

                for i in range(2, int(limit**0.5) + 1):
                    if is_prime[i]:
                        for j in range(i * i, limit + 1, i):
                            is_prime[j] = False

                return np.where(is_prime)[0]

            self.sieve_numba = sieve_numba_serial

        # Segmentiertes Sieb
        @jit(nopython=True, cache=self.use_cache)
        def segmented_sieve_numba(limit, segment_size):
            # Erst kleine Primzahlen bis sqrt(limit) finden
            sqrt_limit = int(limit**0.5) + 1
            small_primes = np.ones(sqrt_limit + 1, dtype=np.bool_)
            small_primes[0] = small_primes[1] = False

            for i in range(2, int(sqrt_limit**0.5) + 1):
                if small_primes[i]:
                    for j in range(i * i, sqrt_limit + 1, i):
                        small_primes[j] = False

            base_primes = np.where(small_primes)[0]
            all_primes = base_primes.copy()

            # Segmentweise sieben
            for start in range(sqrt_limit + 1, limit + 1, segment_size):
                end = min(start + segment_size - 1, limit)
                segment = np.ones(end - start + 1, dtype=np.bool_)

                for p in base_primes:
                    if p * p > end:
                        break

                    # Erstes Vielfaches von p im Segment finden
                    start_multiple = max(p * p, (start + p - 1) // p * p)

                    for j in range(start_multiple, end + 1, p):
                        segment[j - start] = False

                segment_primes = np.where(segment)[0] + start
                all_primes = np.concatenate((all_primes, segment_primes))

            return all_primes

        self.segmented_sieve_numba = segmented_sieve_numba

    def fj32_fallback(self, n: int) -> bool:
        """Fallback-Implementierung von Forisek-Jancina-32"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False

        # Miller-Rabin mit festen Basen für n < 2^32
        witnesses = [2, 7, 61] if n < 4759123141 else [2, 3, 5, 7, 11, 13, 17]

        # n-1 = d * 2^r schreiben
        r = 0
        d = n - 1
        while d % 2 == 0:
            r += 1
            d //= 2

        for a in witnesses:
            if a >= n:
                continue

            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue

            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False

        return True

    def measure_with_memory(self, func, *args, **kwargs) -> Tuple[Any, float, float]:
        """Führt Funktion aus und misst Zeit + Speicherverbrauch"""
        tracemalloc.start()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtime = end_time - start_time
        memory_mb = peak / 1024 / 1024  # Umrechnung in MB

        return result, runtime, memory_mb

    def benchmark_sieve_methods(
        self, limit: int, segment_sizes: List[int]
    ) -> Dict[str, Any]:
        """Benchmarkt verschiedene Sieb-Methoden"""
        results = {}

        # Standard Numba Sieb
        if self.use_numba:
            primes, runtime, memory = self.measure_with_memory(self.sieve_numba, limit)
            results["numba_standard"] = {
                "count": len(primes),
                "runtime": runtime,
                "memory_mb": memory,
                "segment_size": None,
            }

        # Segmentierte Siebe
        for segment_size in segment_sizes:
            if self.use_numba:
                primes, runtime, memory = self.measure_with_memory(
                    self.segmented_sieve_numba, limit, segment_size
                )
                results[f"numba_segmented_{segment_size}"] = {
                    "count": len(primes),
                    "runtime": runtime,
                    "memory_mb": memory,
                    "segment_size": segment_size,
                }

        return results

    def benchmark_primality_tests(self, test_numbers: List[int]) -> Dict[str, Any]:
        """Benchmarkt Primzahltests"""
        results = {}

        # FJ32 C-Extension
        if self.fj32_available:
            start_time = time.perf_counter()
            fj32_results = [self.fj32_c(n) for n in test_numbers]
            fj32_time = time.perf_counter() - start_time

            results["fj32_c"] = {"runtime": fj32_time, "results": fj32_results}

        # Fallback-Implementierung
        start_time = time.perf_counter()
        fallback_results = [self.fj32_fallback(n) for n in test_numbers]
        fallback_time = time.perf_counter() - start_time

        results["fj32_fallback"] = {
            "runtime": fallback_time,
            "results": fallback_results,
        }

        return results

    def run_parameter_grid_benchmark(
        self,
        limits: List[int],
        segment_sizes: List[int],
        numba_configs: List[Dict[str, bool]],
    ) -> pd.DataFrame:
        """Führt vollständigen Parameter-Grid-Benchmark durch"""

        all_results = []
        total_combinations = len(limits) * len(segment_sizes) * len(numba_configs)
        current_combination = 0

        print(
            f"Starte Parameter-Grid-Benchmark mit {total_combinations} Konfigurationen..."
        )

        for limit in limits:
            for segment_size in segment_sizes:
                for config in numba_configs:
                    current_combination += 1
                    print(
                        f"[{current_combination}/{total_combinations}] "
                        f"N={limit:,}, Segment={segment_size:,}, Config={config}"
                    )

                    # Neue Analyzer-Instanz für diese Konfiguration
                    analyzer = AdvancedBenchmarkAnalyzer(
                        use_numba=config.get("use_numba", True),
                        use_parallel=config.get("parallel", False),
                        use_cache=config.get("cache", True),
                    )

                    try:
                        # Sieb-Benchmark
                        sieve_results = analyzer.benchmark_sieve_methods(
                            limit, [segment_size]
                        )

                        # Test-Zahlen für Primzahltests generieren (letzte 100
                        # aus dem Sieb)
                        if "numba_standard" in sieve_results:
                            # Verwende Standard-Sieb für Test-Zahlen
                            test_primes, _, _ = analyzer.measure_with_memory(
                                analyzer.sieve_numba, min(10000, limit)
                            )
                            test_numbers = (
                                test_primes[-100:].tolist()
                                if len(test_primes) >= 100
                                else test_primes.tolist()
                            )
                        else:
                            # Fallback: einfache Test-Zahlen
                            test_numbers = list(range(max(2, limit - 200), limit, 2))[
                                :100
                            ]

                        # Primzahltests
                        primality_results = analyzer.benchmark_primality_tests(
                            test_numbers
                        )

                        # Ergebnisse sammeln
                        for method, data in sieve_results.items():
                            result_row = {
                                "limit": limit,
                                "segment_size": segment_size,
                                "use_numba": config.get("use_numba", True),
                                "parallel": config.get("parallel", False),
                                "cache": config.get("cache", True),
                                "method": method,
                                "prime_count": data["count"],
                                "sieve_runtime": data["runtime"],
                                "sieve_memory_mb": data["memory_mb"],
                                "fj32_runtime": primality_results.get("fj32_c", {}).get(
                                    "runtime",
                                    primality_results.get("fj32_fallback", {}).get(
                                        "runtime", 0
                                    ),
                                ),
                                "fj32_method": (
                                    "fj32_c"
                                    if "fj32_c" in primality_results
                                    else "fj32_fallback"
                                ),
                            }
                            all_results.append(result_row)

                    except Exception as e:
                        print(f"  ⚠ Fehler bei Konfiguration: {e}")
                        continue

        return pd.DataFrame(all_results)

    def create_visualizations(self, df: pd.DataFrame) -> None:
        """Erstellt verschiedene Visualisierungen der Benchmark-Ergebnisse"""

        # Style setzen
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Runtime vs. Limit (Log-Log-Plot)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Sieve Runtime
        for method in df["method"].unique():
            method_data = df[df["method"] == method]
            ax1.loglog(
                method_data["limit"],
                method_data["sieve_runtime"],
                marker="o",
                label=method,
                alpha=0.7,
            )
        ax1.set_xlabel("Limit N")
        ax1.set_ylabel("Sieb-Laufzeit (s)")
        ax1.set_title("Sieb-Performance: Laufzeit vs. Limit")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Memory Usage
        for method in df["method"].unique():
            method_data = df[df["method"] == method]
            ax2.loglog(
                method_data["limit"],
                method_data["sieve_memory_mb"],
                marker="s",
                label=method,
                alpha=0.7,
            )
        ax2.set_xlabel("Limit N")
        ax2.set_ylabel("Speicherverbrauch (MB)")
        ax2.set_title("Speicherverbrauch vs. Limit")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # FJ32 Runtime
        for fj32_method in df["fj32_method"].unique():
            method_data = df[df["fj32_method"] == fj32_method]
            ax3.loglog(
                method_data["limit"],
                method_data["fj32_runtime"],
                marker="^",
                label=fj32_method,
                alpha=0.7,
            )
        ax3.set_xlabel("Limit N")
        ax3.set_ylabel("FJ32-Laufzeit (s)")
        ax3.set_title("FJ32-Performance vs. Limit")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Efficiency: Primes per second
        df["primes_per_second"] = df["prime_count"] / df["sieve_runtime"]
        for method in df["method"].unique():
            method_data = df[df["method"] == method]
            ax4.loglog(
                method_data["limit"],
                method_data["primes_per_second"],
                marker="d",
                label=method,
                alpha=0.7,
            )
        ax4.set_xlabel("Limit N")
        ax4.set_ylabel("Primzahlen pro Sekunde")
        ax4.set_title("Effizienz: Primzahlen/Sekunde")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "plots/modul4_performance_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        # 2. Heatmap für Segment-Size vs. Limit
        if len(df["segment_size"].dropna().unique()) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Runtime Heatmap
            pivot_runtime = df[
                df["method"].str.contains("segmented", na=False)
            ].pivot_table(
                values="sieve_runtime",
                index="segment_size",
                columns="limit",
                aggfunc="mean",
            )
            sns.heatmap(pivot_runtime, annot=True, fmt=".4f", cmap="viridis", ax=ax1)
            ax1.set_title("Laufzeit-Heatmap: Segment-Größe vs. Limit")
            ax1.set_ylabel("Segment-Größe")
            ax1.set_xlabel("Limit N")

            # Memory Heatmap
            pivot_memory = df[
                df["method"].str.contains("segmented", na=False)
            ].pivot_table(
                values="sieve_memory_mb",
                index="segment_size",
                columns="limit",
                aggfunc="mean",
            )
            sns.heatmap(pivot_memory, annot=True, fmt=".2f", cmap="plasma", ax=ax2)
            ax2.set_title("Speicher-Heatmap: Segment-Größe vs. Limit")
            ax2.set_ylabel("Segment-Größe")
            ax2.set_xlabel("Limit N")

            plt.tight_layout()
            plt.savefig("plots/modul4_heatmaps.png", dpi=300, bbox_inches="tight")
            plt.show()

    def export_results(
        self, df: pd.DataFrame, filename_base: str = "modul4_results"
    ) -> None:
        """Exportiert Ergebnisse in verschiedene Formate"""

        # CSV Export
        csv_path = f"data/{filename_base}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✓ CSV exportiert: {csv_path}")

        # Excel Export mit mehreren Sheets
        excel_path = f"data/{filename_base}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Vollständige Ergebnisse
            df.to_excel(writer, sheet_name="Vollständige_Ergebnisse", index=False)

            # Zusammenfassung nach Methode
            summary_by_method = (
                df.groupby(["method", "limit"])
                .agg(
                    {
                        "sieve_runtime": ["mean", "std"],
                        "sieve_memory_mb": ["mean", "std"],
                        "prime_count": "first",
                    }
                )
                .round(6)
            )
            summary_by_method.to_excel(writer, sheet_name="Zusammenfassung_Methoden")

            # Best Configurations
            best_configs = df.loc[df.groupby("limit")["sieve_runtime"].idxmin()]
            best_configs.to_excel(
                writer, sheet_name="Beste_Konfigurationen", index=False
            )

        print(f"✓ Excel exportiert: {excel_path}")

        # Empfehlungen generieren
        self._generate_recommendations(df, filename_base)

    def _generate_recommendations(self, df: pd.DataFrame, filename_base: str) -> None:
        """Generiert Empfehlungen basierend auf den Benchmark-Ergebnissen"""

        recommendations = []
        recommendations.append("# Modul 4 - Benchmark-Empfehlungen\n")
        recommendations.append(
            f"Generiert am: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Beste Konfiguration pro Limit
        recommendations.append("## Optimale Konfigurationen nach Limit\n")
        for limit in sorted(df["limit"].unique()):
            limit_data = df[df["limit"] == limit]
            best_runtime = limit_data.loc[limit_data["sieve_runtime"].idxmin()]
            best_memory = limit_data.loc[limit_data["sieve_memory_mb"].idxmin()]

            recommendations.append(f"### N = {limit:,}\n")
            recommendations.append(
                f"- **Beste Laufzeit**: {best_runtime['method']} "
                f"({best_runtime['sieve_runtime']:.6f}s, "
                f"{best_runtime['sieve_memory_mb']:.2f}MB)\n"
            )
            recommendations.append(
                f"- **Geringster Speicher**: {best_memory['method']} "
                f"({best_memory['sieve_runtime']:.6f}s, "
                f"{best_memory['sieve_memory_mb']:.2f}MB)\n\n"
            )

        # Allgemeine Empfehlungen
        recommendations.append("## Allgemeine Empfehlungen\n\n")

        # Numba Parallel Analyse
        if df["parallel"].any():
            parallel_avg = df[df["parallel"]]["sieve_runtime"].mean()
            serial_avg = df[df["parallel"] == False]["sieve_runtime"].mean()
            if parallel_avg < serial_avg:
                recommendations.append(
                    "✓ **Numba Parallelisierung** verbessert die Performance signifikant\n"
                )
            else:
                recommendations.append(
                    "⚠ **Numba Parallelisierung** zeigt keinen deutlichen Vorteil\n"
                )

        # Segmentierung Analyse
        segmented_data = df[df["method"].str.contains("segmented", na=False)]
        if not segmented_data.empty:
            best_segments = (
                segmented_data.groupby("segment_size")["sieve_runtime"]
                .mean()
                .sort_values()
            )
            recommendations.append(
                f"✓ **Optimale Segment-Größe**: {best_segments.index[0]:,} "
                f"(Ø {best_segments.iloc[0]:.6f}s)\n"
            )

        # In Datei schreiben
        rec_path = f"data/{filename_base}_empfehlungen.md"
        with open(rec_path, "w", encoding="utf-8") as f:
            f.writelines(recommendations)

        print(f"✓ Empfehlungen exportiert: {rec_path}")


def main():
    """Hauptfunktion für Modul 4 Benchmarks"""
    print("=" * 60)
    print("Modul 4 - Erweiterte Benchmarks & Optimierung")
    print("=" * 60)

    # Parameter-Grid definieren
    limits = [10**6, 10**7]  # Beginne mit kleineren Werten für Tests
    segment_sizes = [10000, 50000, 100000, 500000]
    numba_configs = [
        {"use_numba": True, "parallel": False, "cache": True},
        {"use_numba": True, "parallel": True, "cache": True},
        {"use_numba": True, "parallel": False, "cache": False},
    ]

    # Falls Numba nicht verfügbar, reduziere Konfigurationen
    if not NUMBA_AVAILABLE:
        numba_configs = [{"use_numba": False, "parallel": False, "cache": False}]

    print(f"Parameter-Grid:")
    print(f"  Limits: {[f'{l:,}' for l in limits]}")
    print(f"  Segment-Größen: {[f'{s:,}' for s in segment_sizes]}")
    print(f"  Numba-Konfigurationen: {len(numba_configs)}")
    print()

    # Benchmark-Analyzer initialisieren
    analyzer = AdvancedBenchmarkAnalyzer()

    # Parameter-Grid-Benchmark ausführen
    print("Starte vollständigen Benchmark...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Numba-Warnungen unterdrücken
        results_df = analyzer.run_parameter_grid_benchmark(
            limits, segment_sizes, numba_configs
        )

    if results_df.empty:
        print("❌ Keine Ergebnisse erhalten. Benchmark fehlgeschlagen.")
        return

    print(f"\n✓ Benchmark abgeschlossen. {len(results_df)} Ergebnisse erhalten.")

    # Grundlegende Statistiken anzeigen
    print("\n" + "=" * 60)
    print("ERGEBNISÜBERSICHT")
    print("=" * 60)

    summary = (
        results_df.groupby("limit")
        .agg(
            {
                "sieve_runtime": ["min", "max", "mean"],
                "sieve_memory_mb": ["min", "max", "mean"],
                "prime_count": "first",
            }
        )
        .round(6)
    )

    print(summary)

    # Visualisierungen erstellen
    print("\nErstelle Visualisierungen...")
    analyzer.create_visualizations(results_df)

    # Ergebnisse exportieren
    print("\nExportiere Ergebnisse...")
    analyzer.export_results(
        results_df, f"modul4_benchmark_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print("\n✅ Modul 4 Benchmark erfolgreich abgeschlossen!")
    print("\nDateien erstellt:")
    print("  📊 plots/modul4_performance_analysis.png")
    print("  📊 plots/modul4_heatmaps.png")
    print("  📈 data/modul4_benchmark_YYYYMMDD_HHMMSS.csv")
    print("  📈 data/modul4_benchmark_YYYYMMDD_HHMMSS.xlsx")
    print("  📝 data/modul4_benchmark_YYYYMMDD_HHMMSS_empfehlungen.md")


if __name__ == "__main__":
    main()
