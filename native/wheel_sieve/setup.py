# modul2_wheel_sieve/setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="wheel_sieve",
        sources=["wheel_sieve.pyx"],
        include_dirs=[np.get_include()],
        language="c",
    )
]

setup(
    name="wheel_sieve",
    version="0.1.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
