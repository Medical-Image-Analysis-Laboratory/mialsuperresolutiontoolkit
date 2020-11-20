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
the Docker software container technology, so in order to run the `MIALSRTK BIDS App`, Docker must be installed (see instructions in :ref:`manual-install-docker`).

Once Docker is installed, the recommended way to run the `MIALSRTK BIDS App` is to use the ``mialsuperresolutiontoolkit_bidsapp`` wrapper.
Installation instructions for the wrapper are found in :ref:`manual-install-wrapper`, which requires Python (see :ref:`manual-install-python`) and an Internet connection.

If you need a finer control over the Docker container execution, or you feel comfortable with the Docker Engine, download instructions for the `MIALSRTK BIDS App` can be found in :ref:`manual-install-bidsapp`.

Make sure that you have installed all the following prerequisites.


The MIALSRTK BIDSApp
===============================

.. _manual-install-docker:

Prerequisites
-------------

* Installed Docker Engine corresponding to your system:

  * For Ubuntu 14.04/16.04/18.04, follow the instructions from https://docs.docker.com/install/linux/docker-ce/ubuntu/

  * For Mac OSX (>=10.10.3), get the .dmg installer from https://store.docker.com/editions/community/docker-ce-desktop-mac

  * For Windows (>=10), get the installer from https://store.docker.com/editions/community/docker-ce-desktop-windows

.. note:: The MIALSRTK BIDSApp has been tested only on Ubuntu and MacOSX. For Windows users, it might be required to make few patches in the Dockerfile.


* Docker managed as a non-root user

  * Open a terminal

  * Create the docker group::

    $ sudo groupadd docker

  * Add the current user to the docker group::

    $ sudo usermod -G docker -a $USER

  * Reboot

    After reboot, test if docker is managed as non-root::

      $ docker run hello-world


.. _manual-install-bidsapp:

Installation of the BIDS App
-----------------------------

* Open a terminal

* Get the latest release (|release|) of the BIDS App:

  .. parsed-literal::

    $ docker pull sebastientourbier/mialsuperresolutiontoolkit-bidsapp:|release|

* To display all docker images available::

  $ docker images

 You should see the docker image "mialsuperresolutiontoolkit-bidsapp" with tag "|release|" is now available.

* You are ready to use the MIALSRTK BIDS App from the terminal. See its `commandline usage <usage.html>`_.


The lightweight MIALSRTK BIDSApp wrapper
========================================

.. _manual-install-python:

Prerequisites
---------------

The wrapper requires a Python3 environment. We recommend you tu use miniconda3 for which the installer corresponding to your 32/64bits MacOSX/Linux/Win system can be downloaded from https://conda.io/miniconda.html.

.. _manual-install-wrapper:

Installation
-------------

Once Python3 is installed, the ``mialsuperresolutiontoolkit_bidsapp`` could be installed via `pip` as follows:

* Open a terminal

* Installation with `pip`:

  .. code-block:: console

     $ pip install -e git+https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit#egg=pymialsrtk

* You are ready to use the ``mialsuperresolutiontoolkit_bidsapp`` wrapper. See its `commandline usage <wrapperusage>`_.

Help/Questions
--------------

Code bugs can be reported by creating a new `GitHub Issue <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.
