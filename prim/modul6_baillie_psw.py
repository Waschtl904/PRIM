#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 6 – Baillie-PSW-Test mit echtem Lucas-Selfridge-Verfahren
===============================================================
Dieses Skript:
- Führt Miller-Rabin mit festen Basen durch
- Führt Selfridge’s Lucas-PRP durch
- Kombiniert beides zum Baillie-PSW-Test
- Vergleicht Performance und Zuverlässigkeit mit Forisek-Jancina
- Exportiert Ergebnisse in modul6_data/ und modul6_plots/
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prim.modul4_benchmarks import AdvancedBenchmarkAnalyzer


def miller_rabin(n: int) -> bool:
    n = int(n)
    bases = [2, 7, 61]
    if n < 2 or n % 2 == 0:
        return n == 2
    # schreibe n-1 = d * 2^s
    d, s = n - 1, 0
    while d & 1 == 0:
        d >>= 1
        s += 1
    for a in bases:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def lucas_prp(n: int) -> bool:
    n = int(n)
    # Selfridge’s Parameterwahl: suche D mit Jacobi(D|n) = -1
    D = 5
    sign = 1
    while pow(D, (n - 1) // 2, n) != n - 1:
        D = sign * (abs(D) + 2)
        sign = -sign
    P, Q = 1, (1 - D) // 4

    # Lucas-Iteration
    def lucas_step(n, P, Q, k):
        # returns (U_k, V_k, Q^k)
        if k == 0:
            return (0, 2, 1)
        if k == 1:
            return (1, P, Q)
        # rekursives Verdoppeln
        U2j, V2j, Q2j = lucas_step(n, P, Q, k >> 1)
        U = (U2j * V2j) % n
        V = (V2j * V2j - 2 * Q2j) % n
        Qk = pow(Q2j, 2, n)
        if k & 1:
            U1, V1, Q1 = lucas_step(n, P, Q, 1)
            U = (U * V1 + V * U1) % n
            V = (V * V1 + U * U1 * D) % n
            Qk = (Qk * Q1) % n
        return (U, V, Qk)

    # Prüfe U_{n+1} mod n
    U, V, _ = lucas_step(n, P, Q, n + 1)
    return U == 0


def baillie_psw(n: int) -> bool:
    return miller_rabin(n) and lucas_prp(n)


def main():
    # Konfiguration laden
    try:
        with open("config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except BaseException:
        cfg = {"use_numba": True, "use_parallel": True, "use_cache": True}

    analyzer = AdvancedBenchmarkAnalyzer(
        use_numba=cfg.get("use_numba", True),
        use_parallel=cfg.get("use_parallel", True),
        use_cache=cfg.get("use_cache", True),
    )

    # Testzahlen
    limit = 10**6
    test_numbers = np.concatenate(
        [
            analyzer.sieve_numba(limit),
            # Carmichael-Zahlen
            np.array([2, 3, 4, 561, 1105, 1729, 2465, 2821]),
        ]
    )

    results = []
    for n in test_numbers:
        n = int(n)
        # Forisek-Jancina
        t0 = time.time()
        fj = analyzer.fj32_fallback(n)
        t_fj = time.time() - t0
        # Baillie-PSW
        t0 = time.time()
        bp = baillie_psw(n)
        t_bp = time.time() - t0
        results.append(
            {
                "n": n,
                "is_prime_fj": fj,
                "time_fj": t_fj,
                "is_prime_bp": bp,
                "time_bp": t_bp,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv("modul6_data/baillie_psw_results.csv", index=False)

    # Performance-Vergleich
    plt.figure(figsize=(6, 4))
    plt.scatter(df["time_fj"], df["time_bp"], alpha=0.3)
    plt.xlabel("FJ32-C Zeit (s)")
    plt.ylabel("Baillie-PSW Zeit (s)")
    plt.title("Performance-Vergleich")
    plt.savefig("modul6_plots/performance_compare.png", dpi=300)
    plt.close()

    # Fehlerraten-Analyse
    errors = df[df["is_prime_fj"] != df["is_prime_bp"]]
    plt.figure(figsize=(6, 4))
    plt.hist(errors["n"], bins=20, color="salmon")
    plt.title("Zahlen mit unterschiedlichen Ergebnissen")
    plt.xlabel("n")
    plt.ylabel("Anzahl")
    plt.savefig("modul6_plots/discrepancies.png", dpi=300)
    plt.close()

    print("✅ Baillie-PSW-Analyse abgeschlossen.")
    print("  Ergebnisse: modul6_data/baillie_psw_results.csv")
    print(
        "  Plots: modul6_plots/performance_compare.png, modul6_plots/discrepancies.png"
    )


if __name__ == "__main__":
    main()
