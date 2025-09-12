#!/usr/bin/env python3
"""
Erweiterte Benchmark-Import-Reparatur für PRIM Repository
"""

import os
import sys
from pathlib import Path


def fix_comprehensive_benchmark():
    """Repariert spezifisch comprehensive_benchmark.py"""

    benchmark_file = Path("benchmarks/comprehensive_benchmark.py")

    if not benchmark_file.exists():
        print("❌ comprehensive_benchmark.py nicht gefunden!")
        return False

    print(f"🔧 Repariere: {benchmark_file}")

    # Lese aktuelle Datei
    with open(benchmark_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Füge sys.path Insert am Anfang hinzu
    path_setup = """#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Füge src zum Python-Path hinzu
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

"""

    # Entferne alte Path-Setups und Import-Zeilen
    lines = content.split("\n")
    new_lines = []
    skip_until_import = True

    for line in lines:
        # Überspringe alte Path-Setups und erste Import-Zeilen
        if skip_until_import:
            if line.startswith("import") or line.startswith("from"):
                skip_until_import = False
                # Ersetze src.prim mit prim
                if "from src.prim" in line:
                    line = line.replace("from src.prim", "from prim")
                new_lines.append(line)
            elif "#!/usr/bin/env python3" in line or line.strip() == "":
                continue  # Überspringe alte Shebangs und leere Zeilen
            elif "sys.path" in line or "Path(__file__)" in line:
                continue  # Überspringe alte Path-Setups
        else:
            # Ersetze weitere src.prim Imports
            if "from src.prim" in line:
                line = line.replace("from src.prim", "from prim")
            if "import src.prim" in line:
                line = line.replace("import src.prim", "import prim")
            new_lines.append(line)

    # Kombiniere neuen Content
    new_content = path_setup + "\n".join(new_lines)

    # Schreibe zurück
    with open(benchmark_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ {benchmark_file} repariert")
    return True


def test_imports():
    """Teste ob alle kritischen Imports funktionieren"""

    # Füge src zum Path hinzu
    src_path = Path("src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print("\n🧪 TESTE BENCHMARK IMPORTS:")
    print("-" * 40)

    test_imports = [
        "prim",
        "prim.core",
        "prim.algorithms",
        "prim.hybrid_wrapper",
        "numpy",
        "time",
    ]

    success = 0
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
            success += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
        except Exception as e:
            print(f"⚠️  {module}: {e}")

    print(f"\n📊 {success}/{len(test_imports)} Imports erfolgreich")
    return success == len(test_imports)


if __name__ == "__main__":
    print("ERWEITERTE BENCHMARK REPARATUR")
    print("=" * 40)

    fix_comprehensive_benchmark()
    test_imports()

    print("\n" + "=" * 40)
    print("REPARATUR ABGESCHLOSSEN")
    print("\nNächste Schritte:")
    print("1. python run_benchmarks.py")
    print("2. python benchmarks/comprehensive_benchmark.py")
