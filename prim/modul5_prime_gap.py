#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modul 5 – Prime-Gap-Analyse
===========================

Dieses Skript:
- Erzeugt Primzahlliste bis N
- Berechnet Abstaende benachbarter Primzahlen (Prime Gaps)
- Visualisiert Histogramm und Verteilung
- Exportiert Ergebnisse in modul5_data/ und modul5_plots/
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # 1. Konfiguration laden
    import json

    from modul4_benchmarks import AdvancedBenchmarkAnalyzer

    # 1. Konfiguration sicher laden mit Fehlerfallback
    try:
        with open("config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"? Fehler beim Laden von config.json: {e}")
    print(
        "  Verwende Standard-Konfiguration: use_numba=True, parallel=True, cache=True"
    )
    cfg = {"use_numba": True, "parallel": True, "cache": True}

    analyzer = AdvancedBenchmarkAnalyzer(
        use_numba=cfg.get("use_numba", True),
        use_parallel=cfg.get("parallel", True),
        use_cache=cfg.get("cache", True),
    )

    # 2. Parameter
    N = 10**6  # Obergrenze fuer Prime-Gap-Analyse

    # 3. Primzahlliste erzeugen
    primes = analyzer.sieve_numba(N)

    # 4. Prime Gaps berechnen
    gaps = np.diff(primes)

    # 5. Statistik
    stats = {
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "max_gap": int(np.max(gaps)),
    }
    print("Statistik:", stats)

    # 6. Ergebnisse exportieren
    df_primes = pd.DataFrame({"prime": primes})
    df_gaps = pd.DataFrame({"gap": gaps})
    df_primes.to_csv("modul5_data/primes.csv", index=False)
    df_gaps.to_csv("modul5_data/gaps.csv", index=False)

    # 7. Visualisierungen
    plt.figure(figsize=(8, 4))
    plt.hist(gaps, bins=50, color="skyblue")
    plt.title("Histogramm der Prime Gaps bis N={:,}".format(N))
    plt.xlabel("Gap")
    plt.ylabel("Haeufigkeit")
    plt.savefig("modul5_plots/gaps_histogram.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(primes[1:], gaps, ".", markersize=2)
    plt.title("Prime Gap vs. Prime bis N={:,}".format(N))
    plt.xlabel("Prime")
    plt.ylabel("Gap")
    plt.savefig("modul5_plots/gaps_scatter.png", dpi=300)
    plt.close()

    print("? Prime-Gap-Analyse abgeschlossen.")
    print("  - Daten: modul5_data/primes.csv, modul5_data/gaps.csv")
    print("  - Plots: modul5_plots/gaps_histogram.png, modul5_plots/gaps_scatter.png")


if __name__ == "__main__":
    main()
