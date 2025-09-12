# src/prim/primetest.py

"""
Alias-Modul für Legacy-Import in prim.analysis.hybrid.
"""
# Hier importieren wir lediglich die aktuellen Funktionen,
# die prim.analysis.hybrid bei fehlendem C-Extension-Import braucht.

from prim.core.forisek_jancina import forisek_jancina_test
from prim.analysis.hybrid import is_prime, batch_primality_test

# prim.analysis.hybrid greift nur auf `primetest` zu, um
# C-Extensions zu laden. Da wir rein Python-Fallbacks nutzen,
# reicht diese Weiterleitung.
