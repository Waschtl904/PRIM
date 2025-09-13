# Installation

## Systemanforderungen

- **Python**: 3.8 oder höher
- **Compiler**: GCC oder MSVC (für C/C++ Module)
- **Betriebssystem**: Linux, macOS, Windows
- **RAM**: Mindestens 4GB empfohlen

## Grundinstallation

### 1. Repository klonen

```bash
git clone https://github.com/Waschtl904/PRIM.git
cd PRIM
```

### 2. Virtuelle Umgebung erstellen

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Abhängigkeiten installieren

```bash
# Basis-Installation
pip install -r requirements.txt

# Entwicklungsumgebung
pip install -r requirements-dev.txt
```

## C/C++ Module kompilieren

### Automatisches Build

```bash
# Linux/macOS
./build_complete_baillie_psw.sh

# Windows PowerShell
.\build_complete_baillie_psw.ps1
```

### Manueller Build

```bash
# Forisek-Jancina C-Modul
gcc -O3 -fPIC -shared native/forisek_jancina/_fj32_c.c -o prim_fj.so

# Baillie-PSW Modul
gcc -O3 -fPIC -shared native/baillie_psw/baillie_psw.c -o prim_baillie.so
```

## Entwicklungsumgebung

### Pre-commit Hooks installieren

```bash
pre-commit install
```

### IDE-Konfiguration

#### Visual Studio Code
Empfohlene Extensions:
- Python Extension Pack
- C/C++ Extension Pack
- Jupyter
- GitLens

#### PyCharm
- Projekt als Python-Projekt öffnen
- Interpreter auf venv setzen
- Code Style auf Black konfigurieren

## Verifikation der Installation

```python
# Test der Basisinstallation
python -c "import prim; print('PRIM erfolgreich installiert!')"

# Funktionstest
python -c "
import prim
result = prim.miller_rabin(1009, rounds=10)
print(f'Miller-Rabin Test: {result}')
"
```

## Problembehandlung

### Häufige Probleme

**Problem**: `ModuleNotFoundError: No module named 'prim'`
**Lösung**:
```bash
pip install -e .
```

**Problem**: C-Module können nicht kompiliert werden
**Lösung**:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows
# Visual Studio Build Tools installieren
```

**Problem**: GitHub Actions schlagen fehl
**Lösung**: Stelle sicher, dass `mypy.ini` im Projekt-Root existiert:
```ini
[mypy]
python_version = 3.8
strict = false
ignore_missing_imports = true
files = src/prim
```

### Performance-Optimierung

```bash
# Optimierte C-Compilation
export CFLAGS="-O3 -march=native -mtune=native"
./build_complete_baillie_psw.sh
```

## Docker-Installation (Optional)

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y build-essential git
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN ./build_complete_baillie_psw.sh

CMD ["python", "-m", "prim", "--help"]
```

## Nächste Schritte

Nach erfolgreicher Installation:
1. [Erste Schritte](getting-started.md)
2. [API-Referenz durchgehen](api-reference.md)
3. [Beispiele ausprobieren](examples.md)
