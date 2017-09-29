from distutils.core import setup
from Cython.Build import cythonize

setup(
        ext_modules = cythonize("doc2vec_inner_ct.pyx")
)
