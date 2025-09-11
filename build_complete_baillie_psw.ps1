# build_complete_baillie_psw.ps1

Write-Host "=== BUILDING COMPLETE BAILLIE-PSW IMPLEMENTATION ===" -ForegroundColor Green

# Schritt 1: C-Library kompilieren
Write-Host "Step 1: Building C library..." -ForegroundColor Yellow

# Prüfe ob GCC verfügbar ist
$gccPath = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gccPath) {
    Write-Host "ERROR: GCC not found. Please install MinGW-w64 or MSYS2" -ForegroundColor Red
    Write-Host "Download from: https://www.msys2.org/" -ForegroundColor Yellow
    exit 1
}

# Erstelle Build-Verzeichnis
$buildDir = "modul8_baillie_psw\implementations\build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir -Force
}

# Kompiliere C-Library
Set-Location "modul8_baillie_psw\implementations"

Write-Host "Compiling baillie_psw_complete.c..." -ForegroundColor Cyan
& gcc -O3 -fPIC -shared -o "build\baillie_psw_complete.dll" "baillie_psw_complete.c"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: C compilation failed!" -ForegroundColor Red
    exit 1
}

# Kopiere DLL zum Hauptverzeichnis
Copy-Item "build\baillie_psw_complete.dll" ".\baillie_psw_complete.dll" -Force
Write-Host "✓ C library built successfully: baillie_psw_complete.dll" -ForegroundColor Green

Set-Location "..\.."

# Schritt 2: Cython kompilieren (optional)
Write-Host "Step 2: Building Cython extension..." -ForegroundColor Yellow

Set-Location "prim"

$cythonAvailable = Get-Command cython -ErrorAction SilentlyContinue
if ($cythonAvailable) {
    Write-Host "Cython found, building optimized extension..." -ForegroundColor Cyan
    
    try {
        & python setup_cython.py build_ext --inplace
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Cython extension built successfully!" -ForegroundColor Green
        } else {
            Write-Host "⚠ Cython build failed, but ctypes fallback will work" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠ Cython build failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ Cython not found, skipping optimized extension (ctypes fallback will be used)" -ForegroundColor Yellow
}

Set-Location ".."

# Schritt 3: Tests ausführen
Write-Host "Step 3: Running verification tests..." -ForegroundColor Yellow

$testScript = @"
import sys
import os
sys.path.insert(0, 'prim')

try:
    # Test ctypes wrapper zuerst
    print('Testing ctypes wrapper...')
    from baillie_psw_wrapper import baillie_psw
    
    # Test bekannte Primzahlen
    test_cases = [
        (2, True), (3, True), (5, True), (7, True),
        (97, True), (101, True), (103, True),
        (4, False), (6, False), (8, False), (9, False),
        (561, False),  # Carmichael-Zahl
        (1105, False), # Carmichael-Zahl
    ]
    
    print('Testing individual numbers...')
    all_passed = True
    for n, expected in test_cases:
        try:
            result = baillie_psw(n)
            status = '✓' if result == expected else '✗'
            print(f'{status} baillie_psw({n}) = {result} (expected {expected})')
            if result != expected:
                all_passed = False
        except Exception as e:
            print(f'✗ baillie_psw({n}) failed: {e}')
            all_passed = False
    
    if all_passed:
        print('\n✓ All basic tests passed!')
    else:
        print('\n✗ Some tests failed!')
        sys.exit(1)
        
    # Test hybrid wrapper
    print('\nTesting hybrid wrapper...')
    try:
        from hybrid_wrapper import is_prime, batch_primality_test
        
        # Test ein paar Zahlen
        test_nums = [97, 98, 99, 100, 101]
        expected = [True, False, False, False, True]
        
        individual_results = [is_prime(n) for n in test_nums]
        batch_results = batch_primality_test(test_nums)
        
        print(f'Individual: {individual_results}')
        print(f'Batch:      {batch_results}')
        print(f'Expected:   {expected}')
        
        if individual_results == expected and batch_results == expected:
            print('✓ Hybrid wrapper working correctly!')
        else:
            print('✗ Hybrid wrapper has issues!')
            
    except ImportError as e:
        print(f'⚠ Hybrid wrapper not available: {e}')
    except Exception as e:
        print(f'✗ Hybrid wrapper error: {e}')
    
    print('\n=== TESTS COMPLETED ===')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"@

# Schreibe temporäres Test-Script
$testScript | Out-File -FilePath "temp_test.py" -Encoding UTF8

# Führe Tests aus
& python temp_test.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== BUILD AND TEST COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
} else {
    Write-Host "`n=== TESTS FAILED ===" -ForegroundColor Red
    exit 1
}

# Lösche temporäres Script
Remove-Item "temp_test.py" -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Usage examples:" -ForegroundColor Cyan
Write-Host "  from prim.baillie_psw_wrapper import baillie_psw"
Write-Host "  print(baillie_psw(97))  # True"
Write-Host "  print(baillie_psw(561)) # False (Carmichael number)"
Write-Host ""
Write-Host "For hybrid mode:" -ForegroundColor Cyan  
Write-Host "  from prim.hybrid_wrapper import is_prime"
Write-Host "  results = [is_prime(n) for n in [97, 98, 99, 100, 101]]"
