# src/prim/__init__.py

import os

# Füge src/prim_src als Quelle für das prim-Paket hinzu
__path__.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prim_src"))
)
