# PRIM Projekt Restructuring Plan

## Phase 1: Verzeichnisstruktur modernisieren

### Zielstruktur:
```
PRIM/
├── src/
│   └── prim/
│       ├── __init__.py
│       ├── core/                    # Kern-Algorithmen
│       │   ├── __init__.py
│       │   ├── forisek_jancina.py   # modul1 → core
│       │   ├── wheel_sieve.py       # modul2 → core
│       │   └── baillie_psw.py       # modul6 → core
│       ├── analysis/                # Analyse & Benchmarks
│       │   ├── __init__.py
│       │   ├── benchmarks.py        # modul4 → analysis
│       │   ├── prime_gap.py         # modul5 → analysis
│       │   └── hybrid.py            # modul3 → analysis
│       ├── utils/                   # Hilfsfunktionen
│       │   ├── __init__.py
│       │   └── common.py
│       └── cli/                     # Command Line Interface
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── core/
│   │   ├── test_forisek_jancina.py
│   │   ├── test_wheel_sieve.py
│   │   └── test_baillie_psw.py
│   └── analysis/
│       ├── test_benchmarks.py
│       ├── test_prime_gap.py
│       └── test_hybrid.py
├── examples/                        # Jupyter Notebooks als Tutorials
│   ├── 01_getting_started.ipynb
│   ├── 02_benchmark_comparison.ipynb
│   ├── 03_prime_gap_analysis.ipynb
│   └── 04_custom_algorithms.ipynb
├── docs/
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── api/
│   │   └── tutorials/
│   └── build/
├── data/                            # Daten & Plots
│   ├── benchmarks/
│   ├── prime_gaps/
│   └── quality_metrics/
├── scripts/                         # Build & Utility Scripts
│   ├── build_extensions.sh
│   ├── run_benchmarks.sh
│   └── prepare_dev.py
├── native/                          # C/C++ Extensions
│   ├── forisek_jancina/
│   │   ├── fj32_c.c
│   │   └── setup.py
│   ├── wheel_sieve/
│   │   ├── wheel_sieve.c
│   │   ├── wheel_sieve.pyx
│   │   └── setup.py
│   └── primetest/
│       ├── primetest.cpp
│       └── setup.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── docs.yml
│       └── release.yml
├── pyproject.toml                   # Moderne Konfiguration
├── setup.cfg                        # Legacy Support
├── README.md
├── README.de.md                     # Deutsche Version
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
└── .gitignore
```

## Migration Steps:

### 1. Backup & Branch erstellen
```bash
git checkout -b restructure-v1.0
git add .
git commit -m "Backup before restructuring"
```

### 2. Neue Verzeichnisse erstellen
```bash
mkdir -p src/prim/{core,analysis,utils,cli}
mkdir -p tests/{core,analysis}
mkdir -p examples data/{benchmarks,prime_gaps,quality_metrics}
mkdir -p scripts native/{forisek_jancina,wheel_sieve,primetest}
mkdir -p docs/{source,build}
```

### 3. Module migrieren
- `prim/modul1_forisek_jancina.py` → `src/prim/core/forisek_jancina.py`
- `prim/modul2_simple_sieve_numba.py` → `src/prim/core/wheel_sieve.py`
- `prim/modul3_hybrid.py` → `src/prim/analysis/hybrid.py`
- `prim/modul4_benchmarks.py` → `src/prim/analysis/benchmarks.py`
- `prim/modul5_prime_gap.py` → `src/prim/analysis/prime_gap.py`
- `prim/modul6_baillie_psw.py` → `src/prim/core/baillie_psw.py`

### 4. C/C++ Extensions aufräumen
- `_fj32_c.c` → `native/forisek_jancina/fj32_c.c`
- `primetest.cpp` → `native/primetest/primetest.cpp`
- `modul2_wheel_sieve/wheel_sieve.c` → `native/wheel_sieve/wheel_sieve.c`

### 5. Notebooks zu Examples
- Alle `.ipynb` Dateien nach `examples/` mit besseren Namen

### 6. Daten organisieren
- `modul5_data/`, `modul5_plots/` → `data/prime_gaps/`
- `modul6_data/`, `modul6_plots/` → `data/quality_metrics/`

## Phase 2: Konfiguration modernisieren

### pyproject.toml erstellen (PEP 621)
### Import-Pfade anpassen
### Tests reorganisieren
### CI/CD erweitern

## Phase 3: Dokumentation ausbauen

### Sphinx-Konfiguration
### API-Dokumentation
### Tutorial-Notebooks
### README überarbeiten