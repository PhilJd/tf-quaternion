# Author: Philipp Jund (jundp@informatik.uni-freiburg.de)
from setuptools import setup, find_packages
import os

dirname = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dirname, 'README.md')) as f:
    long_description = f.read()

# To update pip package run:
# python setup.py sdist && python setup.py bdist_wheel && twine upload dist/*


# check if tensorflow is installed. If it's not, add it to dependencies. This
# is to prevent replacement of existing tf versions (e.g. tf-nightly)

requirements = []
try:
    import tensorflow as tf
except ImportError:
    # unfortunately pip suppresses this warning by default
    print("WARNING: Installing CPU-only version of tensorflow. If you have a"
          "GPU and CUDA available consider installing tensorflow-gpu.")
    requirements += ['tensorflow']


setup(
    name='tfquaternion',
    version='0.1.4',
    description="A differentiable quaternion implementation in tensorflow.",
    long_description=long_description,
    url='https://github.com/PhilJd/tf-quaternion',

    author='Philipp Jund',
    author_email='jundp@cs.uni-freiburg.de',

    keywords='quaternion tensorflow differentiable',
    packages=find_packages(),

    install_requires=['numpy'] + requirements,

    license='Apache 2.0',

    classifiers=[
        # Change later on:
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',

        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # Todo(phil): update this
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
