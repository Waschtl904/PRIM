CONTRIBUTING
Vielen Dank für Dein Interesse an einem Beitrag zu PRIM! Um den Prozess so reibungslos wie möglich zu gestalten, beachte bitte folgende Schritte:

1. Fork & Clone
Forke das Repository auf GitHub.

Klone Deine Fork lokal:

bash
git clone https://github.com/DeinNutzername/PRIM.git
cd PRIM
2. Einrichtung
Erstelle und aktiviere ein virtuelles Environment:

bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\\Scripts\\activate   # Windows
Installiere Abhängigkeiten:

bash
pip install -r requirements.txt
3. Arbeitszweig (Branch)
Erstelle für jedes Feature oder jeden Bugfix einen separaten Branch:

bash
git checkout -b feature/neues-feature
4. Code-Qualität
Verwende Flake8 zur Überprüfung des Python-Codes:

bash
flake8
Formatiere C/C++-Code nach Google C++ Style Guide oder clang-format.

5. Tests
Für Python: Füge Unit-Tests im Verzeichnis tests/ hinzu und führe pytest aus.

Für C/C++: Stelle sicher, dass Dein Code kompiliert und etwaige Tests/Benchmarks in modul3_hybrid_performance erfolgreich durchlaufen.

6. Pull Request
Push Deinen Branch zu GitHub:

bash
git push origin feature/neues-feature
Erstelle einen Pull Request gegen den main-Branch dieses Repositories.

Beschreibe Dein Feature oder Deinen Bugfix detailliert im PR-Text.

7. Review & Merge
Das Maintainer-Team prüft Deinen PR und gibt Feedback.

Bitte antworte zeitnah auf Review-Kommentare und passe Deinen Code gegebenenfalls an.

Vielen Dank für Deinen Beitrag!