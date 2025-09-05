from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        name="primetest",
        sources=["primetest.cpp"],
        include_dirs=[pybind11.get_include()],  # holt das pybind11-Header-Verzeichnis
        language="c++",
    ),
]

setup(
    name="primetest",
    version="0.0.1",
    ext_modules=ext_modules,
    zip_safe=False,
)
