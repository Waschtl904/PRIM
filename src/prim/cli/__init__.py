# src/prim/cli/__init__.py

from typing import List
import os


def foo() -> List[int]:
    """Beispiel-Funktion für CLI-Module."""
    return [1, 2, 3]


def run() -> None:
    """Startet die CLI-Anwendung."""
    print("Running PRIM CLI…")
    print(f"Current directory: {os.getcwd()}")
