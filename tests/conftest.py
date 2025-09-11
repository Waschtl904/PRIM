# tests/conftest.py

import sys
import os

# base_dir ist der Projekt-Root (das Verzeichnis, das 'prim/' und 'tests/' enthält)
base_dir = os.path.dirname(os.path.abspath(__file__))  # tests/
project_root = os.path.dirname(base_dir)  # Projekt-Root

if project_root not in sys.path:
    sys.path.insert(0, project_root)
