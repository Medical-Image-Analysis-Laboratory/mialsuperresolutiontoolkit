#!/usr/bin/env python

"""Connectome Mapper and CMTKlib
"""
import os
import sys
from glob import glob
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

packages=["pysrtk","pysrtk.interfaces",
          "pysrtk.workflows"]

################################################################################
# For some commands, use setuptools

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

def main(**extra_args):
    from distutils.core import setup
    from pysrtk.info import __version__
    setup(name='pysrtk',
          version=__version__,
          description='Py(thon) Super Resolution Toolkit',
          long_description="""Py(thon) Super Resolution Toolkit consists of a set of python nipype interfaces to the C++ image processing tools of the MIAL Super Resolution Toolkit necessary to perform motion-robust super-resolution fetal MRI reconstruction. """,
          author= 'Connectomics Lab, CHUV',
          author_email='sebastien.tourbier@alumni.epfl.ch',
          url='http://www.connectomics.org/',
          scripts = glob('scripts/run.py'),
          license='3-Clause BSD License',
          packages = packages,
        classifiers = [c.strip() for c in """\
            Development Status :: 2 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Scientific/Engineering
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],
          maintainer = 'Connectomics Lab, CHUV',
          maintainer_email = 'sebastien.tourbier@alumni.epfl.ch',
          #package_data = package_data,
          requires=["numpy (>=1.2)", "nibabel (>=1.1.0)", "pybids (>=0.6.4)"],
          **extra_args
         )

if __name__ == "__main__":
    main(**extra_setuptools_args)
