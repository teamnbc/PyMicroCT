from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('cyutils.pyx')
)

# Compile using python setup.py build_ext --inplace
