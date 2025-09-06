#!/usr/bin/env python3
"""
PRIM Project Migration Script

Migrates the current PRIM project structure to the new modern layout.
This script should be run from the project root directory.

Usage:
    python migrate_structure.py [--dry-run] [--backup]
    
Options:
    --dry-run    Show what would be done without making changes
    --backup     Create a backup branch before migration
"""

import os
import shutil
import argparse
from pathlib import Path
import subprocess
import sys


class ProjectMigrator:
    """Handles the migration of PRIM project structure."""
    
    def __init__(self, dry_run=False, backup=False):
        self.dry_run = dry_run
        self.backup = backup
        self.root = Path.cwd()
        
    def log(self, message, level="INFO"):
        """Log messages with level."""
        prefix = "DRY-RUN: " if self.dry_run else ""
        print(f"[{level}] {prefix}{message}")
        
    def run_command(self, cmd, check=True):
        """Execute shell command."""
        self.log(f"Running: {' '.join(cmd)}")
        if not self.dry_run:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result
        return None
        
    def create_directory(self, path):
        """Create directory if it doesn't exist."""
        path = Path(path)
        self.log(f"Creating directory: {path}")
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)
            
    def move_file(self, src, dst):
        """Move file from src to dst."""
        src, dst = Path(src), Path(dst)
        self.log(f"Moving: {src} -> {dst}")
        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dst))
            else:
                self.log(f"Source file not found: {src}", "WARNING")
                
    def copy_file(self, src, dst):
        """Copy file from src to dst."""
        src, dst = Path(src), Path(dst)
        self.log(f"Copying: {src} -> {dst}")
        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(str(src), str(dst))
            else:
                self.log(f"Source file not found: {src}", "WARNING")
                
    def write_file(self, path, content):
        """Write content to file."""
        path = Path(path)
        self.log(f"Writing file: {path}")
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            
    def create_backup(self):
        """Create a backup branch."""
        if self.backup:
            self.log("Creating backup branch: backup-before-restructure")
            self.run_command(["git", "checkout", "-b", "backup-before-restructure"])
            self.run_command(["git", "add", "."])
            self.run_command(["git", "commit", "-m", "Backup before restructuring"])
            self.run_command(["git", "checkout", "main"])
            
    def create_new_structure(self):
        """Create the new directory structure."""
        self.log("Creating new directory structure...")
        
        directories = [
            "src/prim/core",
            "src/prim/analysis", 
            "src/prim/utils",
            "src/prim/cli",
            "tests/core",
            "tests/analysis",
            "examples",
            "docs/source/api",
            "docs/source/tutorials",
            "docs/build",
            "data/benchmarks",
            "data/prime_gaps", 
            "data/quality_metrics",
            "scripts",
            "native/forisek_jancina",
            "native/wheel_sieve",
            "native/primetest",
        ]
        
        for directory in directories:
            self.create_directory(directory)
            
    def migrate_python_modules(self):
        """Migrate Python modules to new structure."""
        self.log("Migrating Python modules...")
        
        migrations = [
            # Core algorithms
            ("prim/modul1_forisek_jancina.py", "src/prim/core/forisek_jancina.py"),
            ("prim/modul2_simple_sieve_numba.py", "src/prim/core/wheel_sieve.py"),
            ("prim/modul6_baillie_psw.py", "src/prim/core/baillie_psw.py"),
            
            # Analysis modules
            ("prim/modul3_hybrid.py", "src/prim/analysis/hybrid.py"),
            ("prim/modul4_benchmarks.py", "src/prim/analysis/benchmarks.py"),
            ("prim/modul5_prime_gap.py", "src/prim/analysis/prime_gap.py"),
        ]
        
        for src, dst in migrations:
            self.move_file(src, dst)
            
    def migrate_notebooks(self):
        """Migrate Jupyter notebooks to examples."""
        self.log("Migrating Jupyter notebooks...")
        
        notebook_migrations = [
            ("prim/modul1_forisek_jancina.ipynb", "examples/01_forisek_jancina_algorithm.ipynb"),
            ("prim/modul3_hybrid.ipynb", "examples/02_hybrid_algorithms.ipynb"),
            ("prim/modul4_benchmarks.ipynb", "examples/03_performance_benchmarks.ipynb"),
            ("prim/modul5_prime_gap.ipynb", "examples/04_prime_gap_analysis.ipynb"),
            ("prim/modul6_baillie_psw.ipynb", "examples/05_baillie_psw_test.ipynb"),
            ("modul2_wheel_sieve/modul2_wheel_sieve.ipynb", "examples/02_wheel_sieve.ipynb"),
        ]
        
        for src, dst in notebook_migrations:
            self.move_file(src, dst)
            
    def migrate_c_extensions(self):
        """Migrate C/C++ extensions."""
        self.log("Migrating C/C++ extensions...")
        
        c_migrations = [
            ("_fj32_c.c", "native/forisek_jancina/fj32_c.c"),
            ("primetest.cpp", "native/primetest/primetest.cpp"),
            ("primetest.pyd", "native/primetest/primetest.pyd"),
            ("modul2_wheel_sieve/wheel_sieve.c", "native/wheel_sieve/wheel_sieve.c"),
            ("modul2_wheel_sieve/wheel_sieve.pyx", "native/wheel_sieve/wheel_sieve.pyx"),
            ("modul2_wheel_sieve/wheel_sieve.cp310-win_amd64.pyd", "native/wheel_sieve/wheel_sieve.cp310-win_amd64.pyd"),
        ]
        
        for src, dst in c_migrations:
            self.move_file(src, dst)
            
    def migrate_data_and_plots(self):
        """Migrate data and plot directories."""
        self.log("Migrating data and plots...")
        
        data_migrations = [
            ("modul5_data", "data/prime_gaps"),
            ("modul5_plots", "data/prime_gaps/plots"),
            ("modul6_data", "data/quality_metrics"),
            ("modul6_plots", "data/quality_metrics/plots"),
        ]
        
        for src, dst in data_migrations:
            if Path(src).exists():
                if not self.dry_run:
                    if Path(dst).exists():
                        shutil.rmtree(dst)
                    shutil.move(str(src), str(dst))
                self.log(f"Moving directory: {src} -> {dst}")
                
    def migrate_scripts_and_builds(self):
        """Migrate build scripts and utilities."""
        self.log("Migrating scripts...")
        
        script_migrations = [
            ("prepare.ps1", "scripts/prepare_dev.ps1"),
            ("modul2_wheel_sieve/setup.py", "native/wheel_sieve/setup.py"),
        ]
        
        for src, dst in script_migrations:
            self.copy_file(src, dst)
            
        # Move build results
        build_dirs = ["build", "modul2_wheel_sieve/build"]
        for build_dir in build_dirs:
            if Path(build_dir).exists() and not self.dry_run:
                shutil.rmtree(build_dir)
                
    def create_init_files(self):
        """Create __init__.py files for new package structure."""
        self.log("Creating __init__.py files...")
        
        init_files = {
            "src/prim/__init__.py": '''"""
PRIM - Modulares Framework für Primzahltests und -analysen
========================================================

Ein umfassendes Python-Framework für die Implementierung, Analyse und 
den Vergleich verschiedener Primzahltests.

Modules:
    core: Kern-Algorithmen für Primzahltests
    analysis: Tools für Benchmark und Analyse
    utils: Hilfsfunktionen
    cli: Command-Line Interface
"""

from ._version import __version__

__all__ = ["__version__"]
''',
            
            "src/prim/core/__init__.py": '''"""
Core primality testing algorithms.
"""

from .forisek_jancina import forisek_jancina_test
from .baillie_psw import baillie_psw_test

__all__ = [
    "forisek_jancina_test",
    "baillie_psw_test",
]
''',
            
            "src/prim/analysis/__init__.py": '''"""
Analysis and benchmarking tools.
"""

from .benchmarks import run_benchmark
from .prime_gap import analyze_prime_gaps

__all__ = [
    "run_benchmark", 
    "analyze_prime_gaps",
]
''',
            
            "src/prim/utils/__init__.py": '''"""
Utility functions and helpers.
"""

__all__ = []
''',
            
            "src/prim/cli/__init__.py": '''"""
Command-line interface.
"""

__all__ = []
''',
        }
        
        for path, content in init_files.items():
            self.write_file(path, content)
            
    def update_imports(self):
        """Update import statements in migrated files."""
        self.log("Import updates will need to be done manually after migration")
        self.log("Common changes needed:")
        self.log("  - Update relative imports to new package structure")
        self.log("  - Change 'from prim.modulX import' to 'from prim.core/analysis import'")
        self.log("  - Update test imports to match new structure")
        
    def create_version_file(self):
        """Create a version file."""
        version_content = '''"""Version information for PRIM."""

__version__ = "1.0.0"
'''
        self.write_file("src/prim/_version.py", version_content)
        
    def cleanup_old_structure(self):
        """Clean up old files and directories."""
        self.log("Cleaning up old structure...")
        
        cleanup_items = [
            "prim/tests",  # Tests moved to root level
            "primtests.egg-info",  # Old package info
        ]
        
        for item in cleanup_items:
            path = Path(item)
            if path.exists() and not self.dry_run:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                self.log(f"Removed: {item}")
                
    def migrate(self):
        """Run the complete migration."""
        self.log("Starting PRIM project migration...")
        
        # Create backup if requested
        self.create_backup()
        
        # Migration steps
        self.create_new_structure()
        self.migrate_python_modules()
        self.migrate_notebooks()
        self.migrate_c_extensions()
        self.migrate_data_and_plots()
        self.migrate_scripts_and_builds()
        self.create_init_files()
        self.create_version_file()
        self.cleanup_old_structure()
        self.update_imports()
        
        self.log("Migration completed!")
        self.log("Next steps:")
        self.log("1. Update import statements in migrated files")
        self.log("2. Replace setup.py with pyproject.toml")
        self.log("3. Update CI/CD configuration")
        self.log("4. Test the new structure with: python -m pytest tests/")
        self.log("5. Build and test package: pip install -e .")


def main():
    parser = argparse.ArgumentParser(description="Migrate PRIM project structure")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup branch before migration")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("prim").exists() or not Path("README.md").exists():
        print("ERROR: This script must be run from the PRIM project root directory")
        sys.exit(1)
        
    migrator = ProjectMigrator(dry_run=args.dry_run, backup=args.backup)
    migrator.migrate()


if __name__ == "__main__":
    main()