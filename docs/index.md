# PRIM Framework Dokumentation

## Übersicht

Willkommen zur offiziellen Dokumentation des **PRIM** (Primality Research and Implementation Module) Frameworks. Diese Dokumentation bietet einen umfassenden Leitfaden für Installation, Nutzung und Entwicklung mit PRIM.

## Inhaltsverzeichnis

- [Installation](installation.md)
- [Erste Schritte](getting-started.md)
- [API-Referenz](api-reference.md)
- [Algorithmus-Details](algorithms.md)
- [Performance Guide](performance.md)
- [Beispiele](examples.md)
- [Entwicklungsrichtlinien](development.md)

## Was ist PRIM?

PRIM ist ein modulares Framework für die Implementierung und Analyse von Primzahltests. Es kombiniert:

- **Hochperformante C/C++ Implementierungen** für geschwindigkeitskritische Operationen
- **Flexible Python APIs** für einfache Integration und Prototyping
- **Umfassende Benchmark-Suites** für Performance-Analysen
- **Wissenschaftliche Visualisierungen** für Primzahlforschung

## Kernalgorithmen

### Miller-Rabin Test
Probabilistischer Primzahltest mit konfigurierbarer Genauigkeit.

### Baillie-PSW Test
Praktisch deterministischer Test ohne bekannte Gegenbeispiele.

### Forisek-Jancina Algorithmus
Optimierter deterministischer Test für 32/64-bit Integer.

## Schneller Einstieg

```python
import prim

# Einfacher Primzahltest
result = prim.is_prime(1009)
print(f"1009 ist prim: {result}")

# Performance-Benchmark
prim.benchmark_algorithms([1009, 10007, 100003])
```

## Projektstruktur

```
PRIM/
├── src/prim/              # Python-Kernpaket
│   ├── algorithms/        # Algorithmus-Implementierungen
│   ├── analysis/          # Analysewerkzeuge
│   ├── core/             # Grundfunktionalitäten
│   └── utils/            # Hilfsfunktionen
├── native/               # C/C++ Module
├── tests/               # Testsuite
├── benchmarks/          # Performance-Tests
├── examples/            # Jupyter Notebooks
└── docs/               # Diese Dokumentation
```
