# PRIM Projekt Migration - Schritt-für-Schritt Anleitung

## Übersicht
Diese Anleitung führt Sie durch die Modernisierung und Reorganisation Ihres PRIM-Projekts. Der Prozess dauert etwa 30-60 Minuten.

## Voraussetzungen
- Git ist installiert und konfiguriert
- Python 3.8+ ist verfügbar
- Sie befinden sich im PRIM-Projektverzeichnis
- Alle aktuellen Änderungen sind committed

## Phase 1: Vorbereitung (5 Minuten)

### 1.1 Status prüfen
```bash
# Aktueller Status
git status
git log --oneline -5

# Sicherstellen, dass alles committed ist
git add .
git commit -m "Pre-migration checkpoint"
```

### 1.2 Backup erstellen
```bash
# Backup-Branch erstellen
git checkout -b backup-before-restructure
git checkout main  # oder Ihr Hauptbranch
```

### 1.3 Migration-Tools laden
```bash
# Migration-Script herunterladen (falls nicht vorhanden)
# Datei: migrate_structure.py
# Ausführbar machen (Linux/macOS)
chmod +x migrate_structure.py
```

## Phase 2: Struktur-Migration (10 Minuten)

### 2.1 Dry-Run durchführen
```bash
# Testen Sie die Migration zuerst
python migrate_structure.py --dry-run

# Bei Problemen: Anpassungen am Script vornehmen
```

### 2.2 Migration ausführen
```bash
# Vollständige Migration mit Backup
python migrate_structure.py --backup

# Status prüfen
git status
```

### 2.3 Manuelle Nachbearbeitung
**Import-Statements aktualisieren:** Öffnen Sie diese Dateien und aktualisieren Sie die Imports:

```python
# Alte Imports (ersetzen Sie diese):
from prim.modul1_forisek_jancina import forisek_jancina_test
from prim.modul4_benchmarks import run_benchmark

# Neue Imports:
from prim.core.forisek_jancina import forisek_jancina_test  
from prim.analysis.benchmarks import run_benchmark
```

**Wichtige Dateien zum Überprüfen:**
- `src/prim/core/*.py`
- `src/prim/analysis/*.py` 
- `tests/*/test_*.py`
- `examples/*.ipynb`

## Phase 3: Konfiguration modernisieren (10 Minuten)

### 3.1 Neue pyproject.toml installieren
```bash
# Alte setup.py sichern
mv setup.py setup.py.old

# Neue pyproject.toml von den bereitgestellten Dateien kopieren
# Datei: pyproject.toml (bereitgestellt)
```

### 3.2 GitHub Actions aktualisieren
```bash
# Neue CI/CD-Pipeline
cp github-ci-cd.yml .github/workflows/ci.yml

# Alte Workflow-Datei entfernen (falls vorhanden)
# git rm .github/workflows/alte-datei.yml
```

### 3.3 Requirements aktualisieren
```bash
# Development-Requirements hinzufügen
# Datei: requirements-dev.txt (bereitgestellt)

# Basis-Requirements prüfen und aktualisieren
# Vergleichen Sie requirements.txt mit den neuen Abhängigkeiten
```

## Phase 4: Installation und Tests (10 Minuten)

### 4.1 Virtual Environment neu erstellen
```bash
# Alte venv löschen und neu erstellen
rm -rf venv/
python -m venv venv

# Aktivieren
source venv/bin/activate  # Linux/macOS
# oder: venv\Scripts\activate  # Windows
```

### 4.2 Paket in Entwicklungsmodus installieren
```bash
# Upgrade pip zuerst
pip install --upgrade pip setuptools wheel

# Paket installieren
pip install -e ".[dev,docs,benchmarks]"

# Alternativ, falls obiges nicht funktioniert:
pip install -r requirements-dev.txt
pip install -e .
```

### 4.3 Tests durchführen
```bash
# Basis-Tests
python -c "import prim; print(f'PRIM v{prim.__version__} loaded successfully')"

# Unit-Tests (wenn vorhanden)
pytest tests/ -v

# Code-Quality prüfen
flake8 src/ --count --statistics
black --check src/
```

## Phase 5: Dokumentation erstellen (10 Minuten)

### 5.1 Sphinx-Dokumentation initialisieren
```bash
# Dokumentations-Verzeichnis aufbauen
cd docs/
sphinx-quickstart --no-sep --project="PRIM" --author="Sebastian Waschtl" \
                  --release="1.0.0" --language="de" --extensions=autodoc,viewcode,napoleon

# Zurück zum Projekt-Root
cd ..
```

### 5.2 Beispiel-Notebooks erstellen
```bash
# Beispiel: Erstes Getting-Started-Notebook
jupyter notebook examples/01_getting_started.ipynb

# Inhalt:
# - Import des Pakets
# - Einfacher Primzahltest
# - Benchmark-Beispiel
```

### 5.3 README aktualisieren
Aktualisieren Sie `README.md` mit:
- Neuer Installation-Anleitung
- Aktualisierten Beispielen
- Verweis auf Dokumentation
- Badge für CI/CD-Status

## Phase 6: Git und Release (10 Minuten)

### 6.1 Änderungen committen
```bash
# Alles hinzufügen und committen
git add .
git status
git commit -m "feat: Modernize project structure

- Migrate to src/ layout with proper package structure
- Add modern pyproject.toml configuration  
- Update CI/CD pipeline with comprehensive testing
- Reorganize modules into core/ and analysis/
- Move notebooks to examples/
- Add development requirements and tools

BREAKING CHANGE: Import paths have changed
- prim.modulX -> prim.core.* or prim.analysis.*
"
```

### 6.2 Tag für Release erstellen
```bash
# Ersten Release-Tag erstellen
git tag -a v1.0.0 -m "Release v1.0.0: Modern project structure"
git push origin main --tags
```

### 6.3 GitHub Actions testen
```bash
# Push um CI/CD zu triggern
git push origin main

# GitHub-Seite öffnen und Actions-Tab prüfen
# https://github.com/Waschtl904/PRIM/actions
```

## Phase 7: Verifikation und Cleanup (5 Minuten)

### 7.1 Funktionalität testen
```bash
# CLI-Kommandos testen
prim --help
prim-benchmark --help

# Python-Import testen
python -c "
from prim.core import forisek_jancina
from prim.analysis import benchmarks
print('All imports successful!')
"
```

### 7.2 Dokumentation generieren
```bash
# Sphinx-Docs erstellen
cd docs/
make html
cd ..

# Lokal öffnen: docs/build/html/index.html
```

### 7.3 Alte Dateien entfernen
```bash
# Nach erfolgreicher Migration
rm setup.py.old
rm migrate_structure.py  # Falls nicht mehr gebraucht

# Commit der finalen Bereinigung
git add .
git commit -m "cleanup: Remove migration artifacts"
```

## Nächste Schritte

### Sofort umsetzbar:
1. **Tests erweitern**: Mehr Unit-Tests für alle Module schreiben
2. **Dokumentation**: API-Docs und Tutorials vervollständigen  
3. **Pre-commit hooks**: `pre-commit install` für automatische Code-Checks

### Mittelfristig (nächste Wochen):
1. **PyPI-Release**: Paket auf PyPI veröffentlichen
2. **Coverage erhöhen**: Ziel 80%+ Test-Coverage
3. **Performance-Benchmarks**: Automatisierte Performance-Regressions-Tests
4. **Docker**: Container für konsistente Entwicklungsumgebung

### Langfristig (nächste Monate):
1. **Neue Module**: Miller-Rabin, AKS-Algorithmus hinzufügen
2. **GPU-Acceleration**: CUDA/OpenCL für große Primzahlen
3. **Web-Interface**: Streamlit/Dash-Dashboard
4. **Community**: Contributing-Guidelines, Issue-Templates

## Troubleshooting

### Problem: Import-Fehler nach Migration
**Lösung:**
```bash
# Paket neu installieren
pip uninstall prim
pip install -e .
```

### Problem: Tests schlagen fehl
**Lösung:**
```bash
# Prüfen Sie Import-Paths in Test-Dateien
# Aktualisieren Sie relative Imports
# pytest -v --tb=short für bessere Fehler-Anzeige
```

### Problem: CI/CD-Fehler
**Lösung:**
```bash
# Lokale Reproduktion:
flake8 src/ tests/
black --check src/ tests/
pytest tests/
```

### Problem: C-Extensions bauen nicht
**Lösung:**
```bash
# Windows: Visual Studio Build Tools installieren
# Linux: build-essential installieren
# macOS: xcode-select --install
```

## Erfolgskriterien

Migration ist erfolgreich abgeschlossen wenn:
- ✅ Alle Tests laufen durch
- ✅ Paket kann importiert werden
- ✅ CI/CD-Pipeline ist grün
- ✅ Dokumentation generiert ohne Fehler
- ✅ Mindestens ein Beispiel-Notebook läuft
- ✅ CLI-Kommandos funktionieren

**Herzlichen Glückwunsch!** Ihr PRIM-Projekt ist nun modern strukturiert und bereit für professionelle Entwicklung und Zusammenarbeit.