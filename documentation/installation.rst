.. _installation:

************************************
Installation Instructions for Users
************************************

.. warning:: This software is for research purposes only and shall not be used for
             any clinical use. This software has not been reviewed or approved by
             the Food and Drug Administration or equivalent authority, and is for
             non-clinical, IRB-approved Research Use Only. In no event shall data
             or images generated through the use of the Software be used in the
             provision of patient care.

Installation of the `MIALSRTK` processing tools and pipelines has been facilitated through the distribution of a BIDSApp relying on
the `Docker <https://www.docker.com/>`_ and `Singularity <https://sylabs.io/>`_ software container technologies, so in order to run `MIALSRTK`, Docker or Singularity must be installed (see instructions in :ref:`manual-install-docker`).

Once Docker or Singularity is installed, the recommended way to run `MIALSRTK` is to use the corresponding ``mialsuperresolutiontoolkit`` wrapper.
Installation instructions for the wrappers can be found in :ref:`manual-install-wrapper`, which requires as prerequisites having Python3 (see :ref:`manual-install-python`) installed and an Internet connection.

If you need a finer control over the Docker/Singularity container execution, or you feel comfortable with the Docker/Singularity Engine, download instructions for the `MIALSRTK BIDS App` can be found in :ref:`manual-install-bidsapp`.


.. _manual-install-docker:

Prerequisites
==============

To run `MIALSRTK` you will need to have either Docker or Singularity containerization engine installed.

While Docker enables `MIALSRTK` to be run on all major operating systems where you have root privileges, Singularity allows you to run `MIALSRTK` on Linux systems where you might not have root privileges such as a High Performance Computing cluster.

Please check https://docs.docker.com/get-started/overview/ and https://sylabs.io/guides/3.7/user-guide/introduction.html If you want to learn more about Docker and Singularity.


Installation of Docker Engine
------------------------------

* Install Docker Engine corresponding to your system:

  * For Ubuntu 14.04/16.04/18.04, follow the instructions from https://docs.docker.com/install/linux/docker-ce/ubuntu/

  * For Mac OSX (>=10.10.3), get the .dmg installer from https://store.docker.com/editions/community/docker-ce-desktop-mac

  * For Windows (>=10), get the installer from https://store.docker.com/editions/community/docker-ce-desktop-windows

.. note:: The MIALSRTK BIDSApp has been tested only on Ubuntu and MacOSX. For Windows users, it might be required to make few patches in the Dockerfile.

* Set Docker to be managed as a non-root user

  * Open a terminal

  * Create the docker group::

    $ sudo groupadd docker

  * Add the current user to the docker group::

    $ sudo usermod -G docker -a $USER

  * Reboot

    After reboot, test if docker is managed as non-root::

      $ docker run hello-world


Installation of Singularity Engine
-----------------------------------

* Install singularity following instructions from the official documentation webpage at https://sylabs.io/guides/3.7/user-guide/quick_start.html#quick-installation-steps

.. note::
    If you need to make the request to install Singularity on your HPC, Singularity provides a nice template at https://singularity.lbl.gov/install-request#installation-request to facilitate it.


.. _manual-install-bidsapp:

MIALSRTK Container Image Download
==================================

Running Docker?
---------------

* Open a terminal

* Get the latest release (|vrelease|) of the BIDS App:

  .. parsed-literal::

    $ docker pull sebastientourbier/mialsuperresolutiontoolkit:|vrelease|

* To display all docker images available::

  $ docker images

 You should see the docker image "mialsuperresolutiontoolkit" with tag "|vrelease|" is now available.

* You are ready to use the Docker image of `MIALSRTK` from the terminal. See its `commandline usage <usage.html>`_.

Running Singularity?
--------------------

* Open a terminal

* Get the latest release (|vrelease|) of the BIDS App:

  .. parsed-literal::

    $ singularity pull library://tourbier/default/mialsuperresolutiontoolkit:|vrelease|

* You are ready to use the Singularity image of `MIALSRTK`. See its `commandline usage <usage.html>`_.


The lightweight MIALSRTK wrappers
==================================

.. _manual-install-python:

Prerequisites
---------------

The wrappers requires a Python3 environment. We recommend you tu use miniconda3 for which the installer corresponding to your 32/64bits MacOSX/Linux/Win system can be downloaded from https://conda.io/miniconda.html.

.. _manual-install-wrapper:

Wrappers Installation
---------------------

Once Python3 is installed, the ``mialsuperresolutiontoolkit_docker`` and ``mialsuperresolutiontoolkit_singularity`` wrappers can be installed via `pip` as follows:

* Open a terminal

* Installation with `pip`:

  .. parsed-literal::

     $ pip install |pypirelease|

* You are ready to use the ``mialsuperresolutiontoolkit_docker`` and ``mialsuperresolutiontoolkit_singularity`` wrappers. See their `commandline usages <wrapperusage>`_.

.. important::

    On Mac and Windows, if you want to track the carbon emission incurred by the processing with the `--track_carbon_footprint` option flag, you will need to install the `Intel Power Gadget` tool available `here <https://www.intel.com/content/www/us/en/developer/articles/tool/power-gadget.html>`.

Help/Questions
--------------

Code bugs can be reported by creating a new `GitHub Issue <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.
