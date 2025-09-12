# test_lucas.py
from prim.core.baillie_psw import lucas_prp


def test_values():
    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97]
    composites = [4, 6, 8, 9, 15, 21, 25, 27, 91]
    for p in primes:
        assert lucas_prp(p), f"lucas_prp sollte True für Primzahl {p} liefern"
    for c in composites:
        assert not lucas_prp(c), f"lucas_prp sollte False für Kompositum {c} liefern"
    print("✅ lucas_prp Funktion funktioniert korrekt für Testwerte.")


if __name__ == "__main__":
    test_values()
