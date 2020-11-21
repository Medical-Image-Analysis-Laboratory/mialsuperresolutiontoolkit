#!/usr/bin/env python

"""``Setup.py`` for PyMIALSRTK."""

import os
import setuptools


directory = os.path.dirname(os.path.abspath(__file__))

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

packages = ["pymialsrtk",
            "pymialsrtk.cli",
            "pymialsrtk.interfaces",
            "pymialsrtk.pipelines",
            "pymialsrtk.pipelines.anatomical"]

package_data = {"pymialsrtk":
                ['data/Network_checkpoints/Network_checkpoints_localization/*',
                 'data/Network_checkpoints/Network_checkpoints_segmentation/*'],
                }

# Extract package requirements from Conda environment.yml
include_conda_pip_dependencies = False

install_requires = []
dependency_links = []
if include_conda_pip_dependencies:
    path = os.path.join(directory, 'docker', 'bidsapp', 'environment.yml')
    with open(path) as read_file:
        state = "PREAMBLE"
        for line in read_file:
            line = line.rstrip().lstrip(" -")
            if line == "dependencies:":
                state = "CONDA_DEPS"
            elif line == "pip:":
                state = "PIP_DEPS"
            elif state == "CONDA_DEPS":
                line = '=='.join(line.split('='))
                line = line.split('==')[0]
                # Python is a valid dependency for Conda but not setuptools, so skip it
                if "python" in line:
                    pass
                else:
                    # Appends to dependencies
                    install_requires.append(line)
            elif state == "PIP_DEPS":
                line = line.split('==')[0]
                # Appends to dependency links
                dependency_links.append(line)
                # Adds package name to dependencies
                install_requires.append(line)

print(f'Install requires: {install_requires}')
print(f'Dependency links: {dependency_links}')


################################################################################
# For some commands, use setuptools

#if len(set(('develop', 'bdist_egg', 'bdist_rpm', 'bdist',  'bdist_wheel', 'bdist_dumb',
#            'bdist_wininst', 'install_egg_info', 'egg_info', 'easy_install',
#            )).intersection(sys.argv)) > 0:


# extra_setuptools_args can be defined from the line above, but it can
# also be defined here because setup.py has been exec'ed from
# setup_egg.py.
if 'extra_setuptools_args' not in globals():
    extra_setuptools_args = dict()


def main(**extra_args):
    """Main function of the ``setup.py``"""

    from distutils.core import setup
    from pymialsrtk.info import __version__
    setuptools.setup(name='pymialsrtk',
          version=__version__,
          description='PyMIALSRTK: Nipype pipelines for the MIAL Super Resolution Toolkit ',
          long_description="""PyMIALSRTK interfaces with the MIALSRTK C++ tools and implements
                              a full processing pipeline using the NiPype dataflow library,
                              from motion-corrupted anisotropic multi-slice MRI scans
                              to a motion-free isotropic high-resolution image. """,
          author='Sebastien Tourbier',
          author_email='sebastien.tourbier@alumni.epfl.ch',
          url='http://www.connectomics.org/',
          # scripts=['pymialsrtk/cli/mialsuperresolutiontoolkit-bidsapp'],
          entry_points={
                 "console_scripts": [
                         'mialsuperresolutiontoolkit_bidsapp = pymialsrtk.cli.mialsuperresolutiontoolkit_bidsapp:main'
                 ]
          },
          license='Modified BSD License',
          package_data=package_data,
          classifiers=[c.strip() for c in """\
            Development Status :: 4 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Scientific/Engineering :: Image Processing
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],
          maintainer='Medical Image Analysis Laboratory, University Hospital of Lausanne and the MIALSRTK developers',
          maintainer_email='sebastien.tourbier@alumni.epfl.ch',
          packages=setuptools.find_packages(),
          # packages=setuptools.find_packages(exclude=['*/*/Network_checkpoints/*/*']),
          install_requires=install_requires,
          dependency_links=dependency_links,
          python_requires='==3.6',
          # requires=["numpy (>=1.2)", "nibabel (>=2.0.0)", "pybids (>=0.9.1)"],
          **extra_args)


if __name__ == "__main__":
    main(**extra_setuptools_args)