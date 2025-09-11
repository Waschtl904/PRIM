from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import platform

# Finde die C-Library
if platform.system() == "Windows":
    lib_name = "baillie_psw_complete.dll"
    extra_link_args = []
else:
    lib_name = "baillie_psw_complete.so"
    extra_link_args = ["-Wl,-rpath,$ORIGIN/../modul8_baillie_psw/implementations"]

lib_path = os.path.join("..", "modul8_baillie_psw", "implementations")

extensions = [
    Extension(
        "baillie_psw_cython",
        ["baillie_psw_cython.pyx"],
        include_dirs=[numpy.get_include(), lib_path],
        library_dirs=[lib_path],
        libraries=["baillie_psw_complete"] if platform.system() != "Windows" else [],
        extra_objects=(
            [os.path.join(lib_path, lib_name)] if platform.system() == "Windows" else []
        ),
        extra_link_args=extra_link_args,
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
    zip_safe=False,
)
