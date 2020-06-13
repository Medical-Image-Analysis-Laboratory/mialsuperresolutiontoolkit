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


* Installation instructions for the MIALSRTK BIDS App are found in :ref:`manual-install-bidsapp`

..
	The steps to add the NeuroDebian repository are explained here::

		$ firefox http://neuro.debian.net/

Make sure that you have installed the following prerequisites.

The MIALSRTK BIDSApp
===============================

Prerequisites
-------------

* Installed Docker Engine corresponding to your system:

  * For Ubuntu 14.04/16.04/18.04, follow the instructions from the web page::

    $ firefox https://docs.docker.com/install/linux/docker-ce/ubuntu/

  * For Mac OSX (>=10.10.3), get the .dmg installer from the web page::

    $ firefox https://store.docker.com/editions/community/docker-ce-desktop-mac

  * For Windows (>=10), get the installer from the web page::

    $ firefox https://store.docker.com/editions/community/docker-ce-desktop-windows

.. note:: Connectome Mapper 3 BIDSApp has been tested only on Ubuntu and MacOSX. For Windows users, it might be required to make few patches in the Dockerfile.


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

Installation
---------------------------------------

Installation of the MIALSRTK pipeline has been facilitated through the distribution of a BIDSApp relying on the Docker software container technology.

* Open a terminal

* Get the latest release (|release|) of the BIDS App:

  .. parsed-literal::

    $ docker pull sebastientourbier/mialsuperresolutiontoolkit-bidsapp:|release|

* To display all docker images available::

  $ docker images

You should see the docker image "mialsuperresolutiontoolkit-bidsapp" with tag "|release|" is now available.

* You are ready to use the MIALSRTK BIDS App from the terminal. See its `commandline usage <usage.html>`_.


Help/Questions
--------------

Code bugs can be reported by creating a new `GitHub Issue <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.
