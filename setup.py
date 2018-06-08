# -*- coding: utf-8 -*-
"""
reference : https://github.com/pypa/sampleproject/blob/master/setup.py
"""

from setuptools import (setup, find_packages)
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='snipy',
      version='0.0.5.1',
      description='python utility @ your own risk',
      long_description='snippet-like code to make snippet-like code',

      # The project's main homepage.
      url='https://github.com/dade-ai/snipy',
      author='dade',
      author_email='aiplore@gmail.com',

      license='MIT',

      # The project's main homepage.

      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.6',
      ],

      packages=find_packages(),
      install_requires=[
            'numpy',
            'setuptools',
      ],
      extras_require={
            'all': ['Jinja2'],
            'plot': ['scikit-image', 'matplotlib'],
            'image': ['scikit-image', 'Pillow', 'scikit-image'],
            'file': ['joblib', 'scandir'],
            'perlin': ['noise'],
            'h5': ['h5py'],
            'json': ['ujson'],
            'sizeof': ['pympler'],
            'cuda': ['pycuda'],  # todo 다른 모듈로 나중에 바꾸자
      }
      )
