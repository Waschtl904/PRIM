# setup.py
from setuptools import setup, find_packages

setup(
    name="PRIM",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # hier deine Abhängigkeiten, z.B. "numpy", "pandas", ...
    ],
)
