"""Lazy numerical imports; CLI help and parser coverage need no GPU imports."""
import importlib
import os

# Keep each evaluation process from creating its own large BLAS thread pool.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'


class LazyModule:
    def __init__(self, name):
        self.name = name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = importlib.import_module(self.name)
        return getattr(self.module, name)
