#!/bin/bash

set -e

echo "Building complete Baillie-PSW implementation..."

# Erstelle Build-Verzeichnis
mkdir -p build
cd build

# Kompiliere die C-Bibliothek
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    gcc -O3 -fPIC -shared -o baillie_psw_complete.dll ../baillie_psw_complete.c
    echo "Built baillie_psw_complete.dll"
else
    # Linux/macOS
    gcc -O3 -fPIC -shared -o baillie_psw_complete.so ../baillie_psw_complete.c
    echo "Built baillie_psw_complete.so"
fi

echo "Build completed successfully!"

# Kopiere zur Integration ins Repository
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    cp baillie_psw_complete.dll ../ 2>/dev/null || true
else
    cp baillie_psw_complete.so ../ 2>/dev/null || true
fi

cd ..
echo "Library copied to modul8_baillie_psw/implementations/"
