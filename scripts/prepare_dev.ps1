# prepare.ps1
<#
.SYNOPSIS
  Bereitet die Arbeitsumgebung für PRIM-Modul 4/5 vor.
#>

# 1. In Projektverzeichnis wechseln
Set-Location $PSScriptRoot

# 2. Virtuelle Umgebung aktivieren
Write-Host "Aktiviere virtuelle Umgebung..."
& .\.venv\Scripts\Activate.ps1

# 3. Abhängigkeiten installieren
if (Test-Path requirements.txt) {
    Write-Host "Installiere Python-Abhaengigkeiten..."
    pip install -r requirements.txt
}
else {
    Write-Warning "requirements.txt nicht gefunden - Installation uebersprungen."
}

# 4. Pruefen auf primetest.pyd
$primitiveDll = Join-Path $PSScriptRoot "primetest.pyd"
if (Test-Path $primitiveDll) {
    Write-Host "primetest.pyd gefunden."
}
else {
    Write-Warning "primetest.pyd fehlt - Fallback-Primtest wird genutzt."
}

# 5. Konfigurationsdatei schreiben
$config = @{
    use_numba       = $true
    parallel        = $true
    cache           = $true
    use_full_params = $false
}
$configJson = $config | ConvertTo-Json -Depth 3
Out-File -FilePath config.json -Encoding UTF8 -InputObject $configJson
Write-Host "Konfigurationsdatei config.json angelegt."

# 6. Umgebungsvariable fuer Notebook setzen
[System.Environment]::SetEnvironmentVariable("PRIM_USE_FULL", ($config.use_full_params), "Process")
Write-Host "Umgebungsvariable PRIM_USE_FULL gesetzt auf $($config.use_full_params)"

# 7. Abschlussmeldung
Write-Host ""
Write-Host "Vorbereitung abgeschlossen. Führen Sie jetzt Modul 4 aus:"
Write-Host "  python modul4_benchmarks.py  oder  jupyter notebook modul4_benchmarks.ipynb"
