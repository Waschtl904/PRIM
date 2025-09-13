#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatische Behebung von Flake8 E231 Fehlern (fehlende Leerzeichen nach Kommas/Doppelpunkten)
"""

import os
import re


def fix_spacing_issues(file_path):
    """
    Automatisch Leerzeichen nach Kommas und Doppelpunkten in Python-Dateien einfügen.
    """
    if not os.path.exists(file_path):
        print(f"Datei {file_path} existiert nicht")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Regex-Patterns für häufige Flake8 E231 Fehler
    patterns = [
        # Komma ohne nachfolgendes Leerzeichen (aber nicht am Zeilenende)
        (r",([^\s\n])", r", \1"),
        # Doppelpunkt ohne nachfolgendes Leerzeichen (Dictionary/Function Args)
        (r":([^\s\n=:])", r": \1"),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Nur schreiben wenn sich etwas geändert hat
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ {file_path} - Leerzeichen-Probleme behoben")
        return True
    else:
        print(f"- {file_path} - Keine Änderungen nötig")
        return False


def fix_variable_name_l(file_path):
    """
    Ersetzt mehrdeutige Variable 'l' durch 'lst' oder ähnliches.
    """
    if not os.path.exists(file_path):
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Ersetze standalone 'l' Variable durch 'lst'
    patterns = [
        (r"\bl\s*=", r"lst ="),
        (r"for\s+l\s+in", r"for lst in"),
        (r"\bl\s*\)", r"lst)"),
        (r"\bl\s*,", r"lst,"),
        (r"\bl\s*\]", r"lst]"),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ {file_path} - Variable 'l' in 'lst' umbenannt")
        return True
    return False


def remove_unused_imports(file_path):
    """
    Entfernt spezifische ungenutzte Imports.
    """
    if not os.path.exists(file_path):
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Entferne spezifische ungenutzte Import-Zeilen
    imports_to_remove = [
        "from sympy.ntheory.primetest import isprime as sympy_isprime",
        "import os",  # falls ungenutzt
    ]

    new_lines = []
    removed_count = 0

    for line in lines:
        should_remove = False
        for unused_import in imports_to_remove:
            if unused_import in line.strip():
                should_remove = True
                removed_count += 1
                print(f"  Entferne: {line.strip()}")
                break

        if not should_remove:
            new_lines.append(line)

    if removed_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"✓ {file_path} - {removed_count} ungenutzte Imports entfernt")
        return True
    return False


def main():
    # Dateien, die bearbeitet werden sollen
    files_to_fix = [
        "src/prim/analysis/benchmarks.py",
        "src/prim/analysis/hybrid.py",
        "src/prim/core/baillie_psw.py",
        "benchmarks/comprehensive_benchmark.py",
    ]

    print("Automatische Behebung von Flake8-Fehlern:")
    print("=" * 50)

    total_fixed = 0

    for file_path in files_to_fix:
        print(f"\nBearbeite: {file_path}")

        # Leerzeichen-Probleme beheben
        if fix_spacing_issues(file_path):
            total_fixed += 1

        # Variable 'l' umbenennen
        if fix_variable_name_l(file_path):
            total_fixed += 1

        # Ungenutzte Imports entfernen
        if remove_unused_imports(file_path):
            total_fixed += 1

    print(f"\n✅ Fertig! {total_fixed} Dateien wurden bearbeitet.")
    print("\nFühre jetzt aus:")
    print("  flake8 src/prim benchmarks")
    print("um zu prüfen, ob die Fehler behoben sind.")


if __name__ == "__main__":
    main()
