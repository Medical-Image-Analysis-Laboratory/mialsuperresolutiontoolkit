"""This file contains MIALSRTK package information."""

_version_major = 2
_version_minor = 0
_version_micro = 2
_version_extra = ''

__release_date__ = '22.11.2021'

__minor_version__ = "%s.%s" % (_version_major,
                               _version_minor)

__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

# __current_year__ = datetime.datetime.now().strftime("%Y")
__current_year__ = '2021'

__author__ = 'The MIAL Super-Resolution ToolKit developers'
__copyright__ = 'Copyright 2016-{}, Medical Image Analysis Laboratory (MIAL), Lausanne'.format(__current_year__)
__credits__ = ('Contributors: please check the ``.zenodo.json`` file at the top-level folder'
               'of the repository')
__license__ = '3-clause BSD'
__maintainer__ = 'Sebastien Tourbier'
__email__ = 'sebastien.tourbier@alumni.epfl.ch'
__status__ = 'Prototype'

__packagename__ = 'mialsuperresolutiontoolkit'

__url__ = 'https://github.com/sebastientourbier/{name}'.format(name=__packagename__)

DOWNLOAD_URL = (
    'https://github.com/sebastientourbier/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

DOCKER_HUB = 'sebastientourbier/mialsuperresolutiontoolkit'
