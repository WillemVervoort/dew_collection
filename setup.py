from distutils.core import setup
# from distutils.extension import Extension
from Cython.Distutils import Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("*", ["*.pyx"], include_dirs = [numpy.get_include()],
    cython_directives = {"boundscheck": False, "wraparound": False,
    "cdivision": True})
]

setup(
  name = "DF-TRAP",
  ext_modules = cythonize(extensions),
  )
