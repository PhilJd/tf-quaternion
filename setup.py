# Author: Philipp Jund (jundp@informatik.uni-freiburg.de)
from setuptools import setup, find_packages

setup(
    name='tfquaternion',
    version='0.1',
    description="A differentiable quaternion implementation in tensorflow.",
    url='',

    author='Philipp Jund',
    author_email='jundp@cs.uni-freiburg.de',

    keywords='quaternion, tensorflow, differentiable',

    license='Apache 2.0',

    classifiers=[
        # Change later on:
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Machine Learning',

        'License :: OSI Approved :: Apache License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # Todo(phil): update this
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
