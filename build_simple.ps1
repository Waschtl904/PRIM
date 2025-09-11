# Vereinfachte Version - nur das Nötigste

Write-Host "Building Baillie-PSW for Windows..." -ForegroundColor Green

# Prüfe GCC
if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: GCC not found. Install MSYS2 from https://www.msys2.org/" -ForegroundColor Red
    exit 1
}

# Kompiliere C-Code
Write-Host "Compiling C library..." -ForegroundColor Yellow
Set-Location "modul8_baillie_psw\implementations"

gcc -O3 -shared -o baillie_psw_complete.dll baillie_psw_complete.c

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Successfully built baillie_psw_complete.dll" -ForegroundColor Green
} else {
    Write-Host "✗ Compilation failed!" -ForegroundColor Red
    exit 1
}

Set-Location "..\.."

# Quick Test
Write-Host "Testing..." -ForegroundColor Yellow
python -c "
import sys
sys.path.insert(0, 'prim')
from baillie_psw_wrapper import baillie_psw
print('Testing baillie_psw(97):', baillie_psw(97))
print('Testing baillie_psw(561):', baillie_psw(561))
print('Success!')
"

Write-Host "✓ Build completed!" -ForegroundColor Green
