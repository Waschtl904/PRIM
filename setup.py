from setuptools import setup, find_packages

setup(
    name="primtests",
    version="0.1.0",
    description="Hybride Primzahltests und Analyse-Tools",
    author="Sebastian Waschtl",
    packages=find_packages(),
    py_modules=["modul4_benchmarks", "modul5_prime_gap"],
    install_requires=["numpy", "pandas", "numba", "matplotlib", "seaborn", "openpyxl"],
    entry_points={"console_scripts": ["primetest-cli=modul5_prime_gap:main"]},
)
