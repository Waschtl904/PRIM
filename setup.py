from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
import os

# Python-Entwicklungsheaders-Ordner ermitteln
python_include = os.path.join(sys.exec_prefix, "include")

extensions = [
    Extension(
        "prim.optimized_cython",
        ["prim/optimized_cython.pyx"],
        include_dirs=[python_include],
    )
]

setup(
    name="prim-cython-extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
