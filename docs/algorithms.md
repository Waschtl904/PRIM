# Algorithmus-Details

## Übersicht der Primzahltests

PRIM implementiert drei Hauptkategorien von Primzahltests, jeweils mit spezifischen Stärken und Anwendungsbereichen.

## Miller-Rabin Test

### Theorie
Der Miller-Rabin Test ist ein probabilistischer Primzahltest basierend auf dem kleinen Satz von Fermat und Eigenschaften quadratischer Reste.

### Algorithmus
```
Eingabe: n (ungerade Zahl > 3), k (Anzahl Runden)
1. Schreibe n-1 = 2^r * d mit d ungerade
2. Für i = 1 bis k:
   a. Wähle zufälliges a ∈ [2, n-2]
   b. x := a^d mod n
   c. Falls x = 1 oder x = n-1: continue
   d. Für j = 1 bis r-1:
      - x := x^2 mod n
      - Falls x = n-1: continue mit nächstem a
   e. Falls hier angekommen: return "zusammengesetzt"
3. Return "wahrscheinlich prim"
```

### Implementation
```python
def miller_rabin(n: int, rounds: int = 10) -> bool:
    """Miller-Rabin Primzahltest

    Args:
        n: Zu testende Zahl
        rounds: Anzahl Testrunden (höher = genauer)

    Returns:
        bool: True wenn wahrscheinlich prim
    """
```

### Performance-Eigenschaften
- **Komplexität**: O(k log³ n)
- **Fehlerrate**: ≤ 4^(-k) für k Runden
- **Optimale Rundenzahl**: 10-20 für praktische Anwendungen

## Baillie-PSW Test

### Theorie
Kombination aus Miller-Rabin Test (Basis 2) und Lucas-Pseudoprimzahltest. Praktisch deterministisch - kein zusammengesetztes Gegenbeispiel bekannt.

### Algorithmus
```
Eingabe: n (ungerade Zahl > 1)
1. Trivialtests: 2, 3, 5, 7, ...
2. Miller-Rabin mit Basis 2
3. Falls n quadratfreier Rest: return "zusammengesetzt"
4. Lucas-Test mit Parametern (P, Q)
5. Return "prim" falls alle Tests bestanden
```

### Parameterauswahl
```python
def find_lucas_parameters(n: int) -> tuple[int, int]:
    """Findet optimale (P, Q) Parameter für Lucas-Test"""
    D = 5
    while jacobi_symbol(D, n) != -1:
        D = -D + 2 if D > 0 else -D - 2
    return 1, (1 - D) // 4
```

### Stärken
- **Kein bekanntes Gegenbeispiel** für n < 2^64
- **Deterministisches Verhalten** in der Praxis
- **Moderate Laufzeit** trotz hoher Sicherheit

## Forisek-Jancina Algorithmus

### Innovation
Hochoptimierter deterministischer Test speziell für 32-bit und 64-bit Integer. Reduziert Aufwand von 3 Miller-Rabin Runden auf 1 Hash + 1 Miller-Rabin.

### Kernidee
```
Für n < 2^32:
1. Berechne h = hash(n) mit spezieller Hash-Funktion
2. Falls h in vordefinierter Menge S: Miller-Rabin(n, basis=2)
3. Sonst: return "zusammengesetzt"
```

### Hash-Funktion
```c
uint32_t fj_hash(uint32_t n) {
    return ((uint64_t)n * 0x9e3779b9) >> (64 - LOG_TABLE_SIZE);
}
```

### Lookup-Tabelle
- **32-bit Version**: 512KB Tabelle
- **64-bit Version**: 8MB Tabelle
- **Trefferquote**: >99.9% für echte Primzahlen

### Performance-Vorteil
```
Traditionell: 3 × Miller-Rabin = ~3ms
Forisek-Jancina: 1 Hash + 1 Miller-Rabin = ~0.8ms
Speedup: ~3.7x
```

## Vergleichstabelle

| Algorithmus | Typ | Laufzeit (10⁶) | Genauigkeit | Speicher |
|-------------|-----|-----------------|-------------|----------|
| Miller-Rabin (k=10) | Probabilistisch | 2.1ms | 4^(-10) | O(1) |
| Baillie-PSW | Quasi-deterministisch | 1.2ms | Praktisch 100% | O(1) |
| Forisek-Jancina | Deterministisch | 0.8ms | 100% (≤2³²) | 512KB |

## Implementierungsdetails

### Modulpotenzen
```python
def mod_pow(base: int, exp: int, mod: int) -> int:
    """Schnelle modulare Potenzierung"""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result
```

### Jacobi-Symbol
```python
def jacobi_symbol(a: int, n: int) -> int:
    """Berechnet Jacobi-Symbol (a/n)"""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n muss ungerade und positiv sein")

    result = 1
    a %= n

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result = -result

        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n

    return result if n == 1 else 0
```

## Erweiterte Optimierungen

### Wheel Factorization
Vorfilterung durch kleine Primzahlen:
```python
WHEEL_2357 = [i for i in range(210) if gcd(i, 2*3*5*7) == 1]

def is_wheel_coprime(n: int) -> bool:
    return (n % 210) in WHEEL_2357
```

### Batch-Processing
```python
def batch_primality_test(candidates: list[int],
                        algorithm: str = "baillie_psw") -> list[bool]:
    """Effiziente Verarbeitung vieler Kandidaten"""
```

## Anwendungsempfehlungen

### Kleine Zahlen (n < 10⁶)
**Trial Division** mit Wheel Factorization

### Mittlere Zahlen (10⁶ < n < 2³²)
**Forisek-Jancina** für maximale Geschwindigkeit

### Große Zahlen (n > 2³²)
**Baillie-PSW** für höchste Zuverlässigkeit

### Kryptographische Anwendungen
**Miller-Rabin** mit k≥20 Runden für provable Sicherheit
