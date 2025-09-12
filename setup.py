from setuptools import setup, find_packages

setup(
    name="prim",
    use_scm_version=False,
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
