#!/usr/bin/env python3
"""
Benchmark Runner für PRIM Repository
Führt alle Benchmarks mit korrekter Pfad-Konfiguration aus
"""

import sys
import os
from pathlib import Path

# Bestimme Projektverzeichnis 
current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
project_root = current_file.parent
src_path = project_root / "src"

# Füge src zum Python-Path hinzu
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print("PRIM Benchmark Runner")
print("=" * 30)
print(f"Project Root: {project_root}")
print(f"Python Path: {src_path}")

try:
    import prim
    print("✅ PRIM Module importiert")

    benchmark_dir = project_root / "benchmarks"

    if benchmark_dir.exists():
        print(f"\nBenchmark-Verzeichnis: {benchmark_dir}")

        for benchmark_file in benchmark_dir.glob("*.py"):
            if benchmark_file.name != "__init__.py":
                print(f"\n🚀 Führe aus: {benchmark_file.name}")
                try:
                    import subprocess

                    # Setze PYTHONPATH für Subprocess
                    env = os.environ.copy()
                    env['PYTHONPATH'] = str(src_path) + os.pathsep + env.get('PYTHONPATH', '')

                    result = subprocess.run([
                        sys.executable, str(benchmark_file)
                    ], cwd=str(project_root), 
                       capture_output=True, 
                       text=True, 
                       env=env,
                       timeout=300)  # 5 Minuten timeout

                    if result.returncode == 0:
                        print(f"✅ {benchmark_file.name} erfolgreich")
                        if result.stdout:
                            # Erste 300 Zeichen des Outputs
                            output = result.stdout[:300]
                            if len(result.stdout) > 300:
                                output += "..."
                            print(f"Output: {output}")
                    else:
                        print(f"❌ {benchmark_file.name} fehlgeschlagen")
                        if result.stderr:
                            error = result.stderr[:300]
                            if len(result.stderr) > 300:
                                error += "..."
                            print(f"Error: {error}")

                except subprocess.TimeoutExpired:
                    print(f"⏰ {benchmark_file.name} timeout (> 5min)")
                except Exception as e:
                    print(f"❌ Fehler bei {benchmark_file.name}: {e}")
    else:
        print("❌ Benchmark-Verzeichnis nicht gefunden!")

except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("\nTipp:")
    print("1. pip install -e .[dev]")
    print("2. Prüfe: python -c 'import prim; print(prim)'")

except Exception as e:
    print(f"❌ Unerwarteter Fehler: {e}")

print("\n" + "=" * 30)
print("Benchmark Runner beendet")
