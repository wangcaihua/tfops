
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from distutils.command.install_data import install_data
from setuptools.command.install_scripts import install_scripts

from cmake_setup import *

PACKAGE_NAME = 'tensorflow-custom-ops'


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


setup(
    name=PACKAGE_NAME,
    version='0.0.1',
    description=('tensorflow-custom-ops is an examples for custom ops for TensorFlow'),
    author='fitzwang',
    author_email='fitzwang@qq.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=[
        'tensorflow >= 1.13.1',
        'cmake_setup >= 0.1.1'
    ],
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension('tfop')],
    cmdclass={
        'build_ext': CMakeBuildExt,
        'install_lib': install_lib,
        'install': InstallPlatlib},
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
    keywords='tensorflow custom op machine learning'
)

# python setup.py sdist bdist_wheel
