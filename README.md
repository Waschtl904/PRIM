# PRIM – Modulares Framework für Primzahltests

[![CI](https://github.com/Waschtl904/PRIM/workflows/CI/badge.svg)](https://github.com/Waschtl904/PRIM/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PRIM** ist ein hochmodernes Framework für die Implementierung, Analyse und Benchmarking verschiedener Primzahltests. Es kombiniert Python-Flexibilität mit C/C++-Performance für optimale Ergebnisse in der algorithmischen Primzahlforschung.

## 🎯 Features

- **Fortgeschrittene Algorithmen**: Miller-Rabin, Baillie-PSW, Forisek-Jancina
- **Multi-Language Performance**: Python + C/C++ Hybridimplementierungen
- **Umfassende Benchmarks**: Detaillierte Performance-Analysen
- **Modulare Architektur**: Saubere Trennung verschiedener Funktionalitäten
- **CI/CD Integration**: Automatisierte Tests und Qualitätsprüfungen

## 📁 Projektstruktur

```
PRIM/
├── src/prim/              # Kern-Python-Paket
│   ├── algorithms/        # Algorithmus-Implementierungen
│   ├── analysis/          # Analysewerkzeuge
│   └── core/             # Kernfunktionalitäten
├── native/               # C/C++ Performance-Module
├── benchmarks/           # Benchmark-Suites
├── tests/               # Testsuite
├── docs/                # Dokumentation
└── examples/            # Jupyter Notebooks mit Beispielen
```

## 🚀 Schnellstart

### Installation

```bash
git clone https://github.com/Waschtl904/PRIM.git
cd PRIM
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Grundlegende Nutzung

```python
from prim import miller_rabin, baillie_psw, forisek_jancina

# Miller-Rabin Test
is_prime = miller_rabin(1009, rounds=10)
print(f"1009 ist prim: {is_prime}")

# Baillie-PSW Test (deterministisch für kleine Zahlen)
is_prime = baillie_psw(1009)
print(f"1009 ist prim: {is_prime}")

# Forisek-Jancina (optimal für 32-bit)
is_prime = forisek_jancina(1009)
print(f"1009 ist prim: {is_prime}")
```

### Command Line Interface

```bash
# Primzahltest für einzelne Zahl
python -m prim test 1009

# Benchmark verschiedener Algorithmen
python -m prim benchmark --range 1000000

# Hilfe anzeigen
python -m prim --help
```

### C/C++ Module kompilieren

```bash
# Forisek-Jancina C-Implementation
gcc -O3 native/forisek_jancina/_fj32_c.c -o fj32_c
./fj32_c 1000000

# Vollständiges Build-Script
./build_complete_baillie_psw.sh
```

## 📊 Module im Detail

### Core Algorithmen

| Algorithmus | Typ | Komplexität | Anwendung |
|-------------|-----|-------------|-----------|
| **Miller-Rabin** | Probabilistisch | O(k log³ n) | Allgemein, konfigurierbare Genauigkeit |
| **Baillie-PSW** | Deterministisch* | O(log³ n) | Höchste Zuverlässigkeit, keine bekannten Gegenbeispiele |
| **Forisek-Jancina** | Deterministisch | O(log² n) | Optimal für 32/64-bit Integer |

*praktisch deterministisch für Zahlen < 2⁶⁴

### Spezialisierte Module

- **`wheel_sieve/`**: Optimierte Sieb-Algorithmen mit Wheel-Factorization
- **`hybrid_performance/`**: Performance-Vergleiche und Optimierungsstrategien
- **`prime_gap_analysis/`**: Statistische Analyse von Primzahlabständen
- **`quality_metrics/`**: Code-Qualität und Linting-Integration

## 🧪 Tests und Qualitätssicherung

```bash
# Alle Tests ausführen
pytest

# Mit Coverage-Report
pytest --cov=prim --cov-report=html

# Linting
flake8 src/
black src/
mypy src/prim/

# Pre-commit hooks
pre-commit run --all-files
```

## 📈 Benchmarks

Das Framework enthält umfassende Benchmark-Suites:

```bash
# Grundlegende Benchmarks
python benchmarks/comprehensive_benchmark.py

# Spezifische Algorithmus-Vergleiche
python -m prim benchmark --algorithms miller_rabin,baillie_psw --range 1000000
```

Beispiel-Ergebnisse für Zahlen bis 10⁶:
- Forisek-Jancina: ~0.8ms durchschnittlich
- Baillie-PSW: ~1.2ms durchschnittlich
- Miller-Rabin (10 Runden): ~2.1ms durchschnittlich

## 📚 Beispiele und Tutorials

Das `examples/` Verzeichnis enthält interaktive Jupyter Notebooks:

- `01_forisek_jancina_algorithm.ipynb`: Einführung in den FJ-Algorithmus
- `02_hybrid_algorithms.ipynb`: Vergleich verschiedener Ansätze
- `03_performance_benchmarks.ipynb`: Detaillierte Performance-Analysen
- `04_prime_gap_analysis.ipynb`: Primzahlabstand-Forschung
- `05_baillie_psw_test.ipynb`: Baillie-PSW Deep-Dive

## 🛠️ Entwicklung

### Setup der Entwicklungsumgebung

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Neue Algorithmen hinzufügen

1. Implementierung in `src/prim/algorithms/`
2. Tests in `tests/`
3. Benchmark-Integration in `benchmarks/`
4. Dokumentation aktualisieren

## 📖 Dokumentation

Vollständige Dokumentation verfügbar in `docs/`:
- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Algorithm Details](docs/algorithms.md)
- [Performance Guide](docs/performance.md)

## 🤝 Beitragen

Wir freuen uns über Beiträge! Bitte lesen Sie [CONTRIBUTING.md](CONTRIBUTING.md) für Details zu:
- Code-Style Guidelines
- Pull Request Prozess
- Issue-Tracking
- Entwicklungsworkflow

## 📄 Lizenz

Dieses Projekt steht unter der MIT Lizenz - siehe [LICENSE](LICENSE) für Details.

## 🔗 Weiterführende Links

- [Forisek-Jancina Paper](https://ceur-ws.org/Vol-1326/020-Forisek.pdf)
- [Baillie-PSW Strengthening](https://arxiv.org/abs/2006.14425)
- [Miller-Rabin Analysis](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test)

## 📧 Kontakt

**Sebastian Waschtl** - [GitHub](https://github.com/Waschtl904)

---

**PRIM** - *Precision in Primality Research* 🔢
