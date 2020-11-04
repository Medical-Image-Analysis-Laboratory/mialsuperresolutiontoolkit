#!/usr/bin/env python

"""PyMIALSRTK
"""
import os
import sys
# from glob import glob
if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

packages = ["pymialsrtk",
            "pymialsrtk.interfaces",
            "pymialsrtk.pipelines",
            "pymialsrtk.pipelines.anatomical"]

package_data = {"pymialsrtk":
                ['data/Network_checkpoints/Network_checkpoints_localization/*',
                 'data/Network_checkpoints/Network_checkpoints_segmentation/*']
                }

# package_data = {'cmtklib':
#                 ['data/parcellation/lausanne2008/*/*.*']
#                 }

################################################################################
# For some commands, use setuptools

if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist', 'bdist_dumb',
            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setup_egg import extra_setuptools_args

# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if 'extra_setuptools_args' not in globals():
    extra_setuptools_args = dict()


def main(**extra_args):
    """Main function of the ``setup.py``"""

    from distutils.core import setup
    from pymialsrtk.info import __version__
    setup(name='pymialsrtk',
          version=__version__,
          description='PyMIALSRTK: Nipype pipelines for the MIAL Super Resolution Toolkit ',
          long_description="""PyMIALSRTK interfaces with the MIALSRTK C++ tools and implements
                              a full processing pipeline using the NiPype dataflow library,
                              from motion-corrupted anisotropic multi-slice MRI scans
                              to a motion-free isotropic high-resolution image. """,
          author='Sebastien Tourbier',
          author_email='sebastien.tourbier@alumni.epfl.ch',
          url='http://www.connectomics.org/',
          scripts=['scripts/superresolution'],
          license='Modified BSD License',
          packages=packages,
          classifiers=[c.strip() for c in """\
            Development Status :: 1 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Scientific/Engineering
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],
          maintainer='Medical Image Analysis Laboratory, University Hospital of Lausanne and the MIALSRTK developers',
          maintainer_email='sebastien.tourbier@alumni.epfl.ch',
          package_data=package_data,
          requires=["numpy (>=1.2)", "nibabel (>=2.0.0)", "pybids (>=0.9.1)"],
          **extra_args)


if __name__ == "__main__":
    main(**extra_setuptools_args)
