PRIM
Modulares Repository für Primzahltests

Übersicht
Dieses Projekt enthält verschiedene Module zur Analyse und Implementierung von Primzahltests:

modul2_wheel_sieve/ – Rad-Sieb-Implementierung

modul3_hybrid_performance/ – Hybrid-Algorithmen und Benchmarks

modul5_data/ & modul5_plots/ – Prime-Gap-Analyse

modul6_data/ & modul6_plots/ – Code-Qualität und Linting

prim/ – Python-Paket mit Kernfunktionen

_fj32_c.c, primetest.cpp – C/C++ Performance-Module

Installation
Repository klonen:

bash
git clone https://github.com/Waschtl904/PRIM.git
cd PRIM
Python-Umgebung erstellen und aktivieren:

bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\\Scripts\\activate   # Windows
Abhängigkeiten installieren:

bash
pip install -r requirements.txt
Nutzung
Python-Paket
bash
python -m prim --help
C/C++-Module
bash
gcc -O3 _fj32_c.c -o fj32_c
./fj32_c 1000000
Testen
Python-Tests:

bash
pytest tests/
Benchmarks:

bash
cd modul3_hybrid_performance
bash run_benchmarks.sh
Beitragende
Lesen Sie die Anleitung in CONTRIBUTING.md, um beizutragen.

Lizenz
MIT License