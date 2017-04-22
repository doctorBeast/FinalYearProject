from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

ext_modules = [
	Extension("Disparity",
		["Disparity.pyx"],
		extra_compile_args = ['-fopenmp'],
		extra_link_args=['-fopenmp'],
		)
]
setup(
	ext_modules = cythonize(ext_modules),
	include_dirs=[numpy.get_include()],
)
