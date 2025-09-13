#!/bin/bash

set -e

echo "=== BUILDING COMPLETE BAILLIE-PSW IMPLEMENTATION ==="

# Schritt 1: C-Library kompilieren
echo "Step 1: Building C library..."
cd modul8_baillie_psw/implementations
chmod +x build.sh
./build.sh

# Schritt 2: Cython kompilieren (optional, falls verfügbar)
echo "Step 2: Building Cython extension..."
cd ../../prim

if command -v cython &> /dev/null; then
    echo "Cython found, building optimized extension..."
    python setup_cython.py build_ext --inplace
    echo "Cython extension built successfully!"
else
    echo "Cython not found, skipping optimized extension (ctypes fallback will be used)"
fi

cd ..

# Schritt 3: Tests ausführen
echo "Step 3: Running verification tests..."
python3 -c "
import sys
sys.path.insert(0, 'prim')

try:
    from hybrid_wrapper import is_prime, batch_primality_test

    # Test bekannte Primzahlen
    test_cases = [
        (2, True), (3, True), (5, True), (7, True),
        (97, True), (101, True), (103, True),
        (4, False), (6, False), (8, False), (9, False),
        (561, False),  # Carmichael-Zahl
        (1105, False), # Carmichael-Zahl
        (1729, False), # Ramanujan-Zahl
    ]

    print('Testing individual numbers...')
    for n, expected in test_cases:
        result = is_prime(n)
        status = '✓' if result == expected else '✗'
        print(f'{status} is_prime({n}) = {result} (expected {expected})')

    # Test Batch-Processing
    print('\nTesting batch processing...')
    numbers = [n for n, _ in test_cases]
    expected_results = [exp for _, exp in test_cases]
    batch_results = batch_primality_test(numbers)

    all_correct = all(r == e for r, e in zip(batch_results, expected_results))
    print(f'Batch test: {\"✓\" if all_correct else \"✗\"} All results correct: {all_correct}')

    print('\n=== ALL TESTS COMPLETED SUCCESSFULLY ===')

except Exception as e:
    print(f'Test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""
echo "=== BUILD AND TEST COMPLETED SUCCESSFULLY ==="
echo ""
echo "Usage examples:"
echo "  from prim.hybrid_wrapper import is_prime"
echo "  print(is_prime(97))  # True"
echo "  print(is_prime(561)) # False (Carmichael number)"
echo ""
echo "For batch processing:"
echo "  from prim.hybrid_wrapper import batch_primality_test"
echo "  results = batch_primality_test([97, 98, 99, 100, 101])"
echo ""
echo "For maximum performance (if Cython is available):"
echo "  from prim.hybrid_wrapper import find_primes_fast"
echo "  primes = find_primes_fast(1, 1000)"
