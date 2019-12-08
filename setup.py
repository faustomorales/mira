# -*- coding: utf-8 -*-
from setuptools.extension import Extension
from setuptools import setup

import numpy
import versioneer

extensions = [
    Extension(
        'mira.utils.compute_overlap',
        ['mira/utils/compute_overlap.pyx'],
        include_dirs=[numpy.get_include()],
    ),
]

setup(version=versioneer.get_version(),
      ext_modules    = extensions,
      cmdclass=versioneer.get_cmdclass())