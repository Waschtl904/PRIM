#!/usr/bin/env python3
"""
Benchmark Fix-Skript für PRIM Repository
Repariert häufige Probleme mit Benchmarks nach Strukturänderungen
"""

import sys
import importlib.util
from pathlib import Path


def fix_benchmark_imports():
    """Repariert Import-Probleme in Benchmark-Skripten"""

    print("🔧 BENCHMARK IMPORTS REPARIEREN")
    print("-" * 40)

    # Finde alle Python-Dateien in benchmarks/
    benchmark_dir = Path("benchmarks")

    if not benchmark_dir.exists():
        print("❌ benchmarks/ Verzeichnis nicht gefunden!")
        return False

    # Häufige Import-Fixes
    import_fixes = {
        "from prim import": "from src.prim import",
        "import prim": "import sys; sys.path.insert(0, 'src'); import prim",
        "from prim.": "from src.prim.",
    }

    fixed_files = []

    for py_file in benchmark_dir.glob("*.py"):
        print(f"Prüfe: {py_file}")

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Wende Fixes an
            for old_import, new_import in import_fixes.items():
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    print(f"  ✅ Fix: {old_import} → {new_import}")

            # Schreibe nur wenn geändert
            if content != original_content:
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(content)
                fixed_files.append(py_file)
                print(f"  💾 {py_file} gespeichert")
            else:
                print(f"  ✅ {py_file} ok")

        except Exception as e:
            print(f"  ❌ Fehler bei {py_file}: {e}")

    print(f"\n📊 {len(fixed_files)} Dateien repariert")
    return True


def test_benchmark_imports():
    """Testet ob Benchmark-Imports funktionieren"""

    print("\n🧪 BENCHMARK IMPORTS TESTEN")
    print("-" * 40)

    # Füge src zum Python-Path hinzu
    src_path = Path("src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Python-Path erweitert: {src_path}")

    # Teste kritische Imports
    critical_imports = ["prim", "prim.core", "prim.algorithms", "numpy", "matplotlib"]

    success_count = 0

    for module_name in critical_imports:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")

    print(f"\n📊 {success_count}/{len(critical_imports)} Imports erfolgreich")
    return success_count == len(critical_imports)


def create_benchmark_runner():
    """Erstellt ein Benchmark-Runner-Skript"""

    runner_script = """#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Bestimme Projektverzeichnis
current_file = Path(__file__).resolve() if '__file__' in globals() else Path.cwd()
project_root = current_file.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

print("PRIM Benchmark Runner")
print("=" * 30)
print(f"Project Root: {project_root}")
print(f"Python Path: {src_path}")

try:
    import prim
    print(f"✅ PRIM Module importiert")

    benchmark_dir = project_root / "benchmarks"

    if benchmark_dir.exists():
        print(f"\nBenchmark-Verzeichnis: {benchmark_dir}")

        for benchmark_file in benchmark_dir.glob("*.py"):
            if benchmark_file.name != "__init__.py":
                print(f"\n🚀 Führe aus: {benchmark_file.name}")
                try:
                    import subprocess
                    result = subprocess.run([
                        sys.executable, str(benchmark_file)
                    ], cwd=str(project_root), capture_output=True, text=True)

                    if result.returncode == 0:
                        print(f"✅ {benchmark_file.name} erfolgreich")
                    else:
                        print(f"❌ {benchmark_file.name} fehlgeschlagen")
                        if result.stderr:
                            print("Error:", result.stderr[:200])

                except Exception as e:
                    print(f"❌ Fehler bei {benchmark_file.name}: {e}")
    else:
        print("❌ Benchmark-Verzeichnis nicht gefunden!")

except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("\nTipp: pip install -e .[dev]")

except Exception as e:
    print(f"❌ Unerwarteter Fehler: {e}")
"""

    with open("run_benchmarks.py", "w", encoding="utf-8") as f:
        f.write(runner_script)

    print(f"\n✅ Benchmark Runner erstellt: run_benchmarks.py")


if __name__ == "__main__":
    print("PRIM BENCHMARK REPARATUR")
    print("=" * 40)

    fix_benchmark_imports()
    test_benchmark_imports()
    create_benchmark_runner()

    print("\n=" * 40)
    print("BENCHMARK-REPARATUR ABGESCHLOSSEN")
    print("\nNächste Schritte:")
    print("1. python run_benchmarks.py")
    print("2. python benchmarks/comprehensive_benchmark.py")
