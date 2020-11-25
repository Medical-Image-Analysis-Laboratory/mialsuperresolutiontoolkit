**************
Changes
**************

Version 2.0.0
--------------

Date: November 25, 2020

This corresponds to the first release of the second version of the MIAL Super-Resolution Toolkit, which has evolved massively over the last years in terms of the underlying codebase and the scope of the functionality provided, following recent advances in standardization of neuroimaging data organization and processing workflows.


Major changes
=============

* Adoption of the `Brain Imaging Data Structure standard <https://bids.neuroimaging.io/>`_ for data organization and the sample dataset available in data/ has been modified accordingly. (See :ref:`BIDS and BIDS App standards <cmpbids>` for more details)

* MIALSRTK is going to Python with the creation of the ``pymialsrtk`` workflow library which extends the `Nipype dataflow library <https://nipype.readthedocs.io/en/latest/>`_ with the implementation of interfaces to all C++ MIALSRTK tools connected in a common workflow to perform super-resolution reconstruction of fetal brain MRI with data provenance and execution detail recordings. (See :ref:`API Documentation <api-doc>`)

* Docker image encapsulating MIALSRTK is distributed as a BIDS App, a standard for containerized workflow that handles BIDS datasets with a set of predefined commandline input argument. (See :ref:`BIDS App Commadline Usage <cmdusage>` for more details)

* Main documentation of MIALSRTK is rendered using readthedocs at https://mialsrtk.readthedocs.io/.


New feature
=============

* ``pymialsrtk``  implements an automatic brain extraction (masking) module based on a 2D U-Net (Ronneberger et al. [Ref1]_) using the pre-trained weights from Salehi et al. [Ref2]_ (See `pull request 4<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/4>`_). It is integrated in the BIDS App workflow by default.

.. [Ref1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_

.. [Ref2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

* ``pymialsrtk``  implements a module for automatic stack reference selection and ordering (masking) based on the tracking of the brain mask centroid slice by slice (See `pull request 34<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/34>`_)

* ``pymialsrtk`` implements for convenience a Python wrapper that generates the Docker command line of the BIDS App for you,
prints it out for reporting purposes, and then executes it without further action needed (See `pull request 47<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/47>`_)


Software development life cycle
================================

* Adopt CircleCI for continuous integration testing and run the following regression tests:

	* Test 01: Run BIDS App on the sample `data/` BIDS dataset with the ``--manual_masks`` option.
	
	* Test 02: Run BIDS App on the sample `data/` BIDS dataset with automated brain extraction (masking).

	See `CircleCI project page <https://app.circleci.com/pipelines/github/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit>`_.

* Use `Codacy <https://www.codacy.com/>`_ to support code reviews and monitor code quality over time.

* Use `coveragepy <https://coverage.readthedocs.io/en/coverage-5.2/>`_  in CircleCI during regression tests of the BIDS app and create code coverage reports published on our `Codacy project page <https://app.codacy.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/dashboard>`_.


More...
========

Please check `pull request 2<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/2>`_, `pull request 4<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/4>`_, `pull request 34<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/34>`_, `pull request 39<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/39>`_, `pull request 47<https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/47>`_ for more change details and development discussions.
