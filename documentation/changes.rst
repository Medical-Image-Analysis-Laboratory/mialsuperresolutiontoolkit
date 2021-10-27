**************
Changes
**************


Version 2.0.2
--------------

Date: October 28, 2021

This corresponds to the release of MIAL Super-Resolution Toolkit 2.0.2,
that includes in particular the following changes.

New feature
=============

- ``pymialsrtk`` enables to fix the maximal amount of memory (in Gb) that could be used by the
  pipelines at execution with the ``--memory MEMORY_Gb`` option flag.
  (See `pull request 92 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/92>`_).

- ``pymialsrtk`` generates a HTML processing report for each subject in `sub-<label>/report/sub-<label>.html`.
  It includes the following:
    - Pipeline/workflow configuration summary
    - Nipype workflow execution graph
    - Links to the log and the profiling output report
    - Plots for the quality check of the automatic reordering step based on the motion index.
    - Three orthogonal cuts of the reconstructed image
    - Computing environment summary
  (See pull requests `97 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/97>`_, `102 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/102>`_, and `103 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/103>`_).

Major change
=============

* The method `pymialsrtk.postprocess.binarize_image()` has been modified and encapsulated in a new interface
  called `pymialsrtk.postprocess.BinarizeImage`.

Python update
===============

* From `3.6.8` to `3.7.10`

New package
==============

* pandas `1.1.5`
* sphinxcontrib-apidoc ``0.3.0`` (required to build documentation)
* sphinxcontrib-napoleon ``0.7`` (required to build documentation)

Package update
===============

* traits from `5.1.2` to ``6.3.0``
* nipype from `1.6.0` to ``1.7.0``
* nilearn from `0.7.1` to ``0.8.1``
* numpy from `1.16.6` to ``1.21.3``
* scikit-learn from `0.20` to ``1.0.1``
* scikit-image from `0.14` to ``0.16.2``

Bug fix
========

* Correct the output filename of the high-resolution brain mask sunk
  in ``mialsrtk-<variant>/sub-<label>/anat``
  (See `pull request 92 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/92>`_).

* The following Sphinx extension packages were added to the conda environment, that were required if one wish
  to build the documentation locally:
    * sphinxcontrib-apidoc ``0.3.0``
    * sphinxcontrib-napoleon ``0.7``

* ``mialsrtkImageReconstruction`` updates the reference image used for
  slice-to-volume registration using the high-resolution image reconstructed
  by SDI at the previous iteration.

Note
====

It was not possible to update the version of tensorflow for the moment.
All versions of tensorflow greater than 1.14 are in fact compiled with
a version of GCC much more recent than the one available in Ubuntu 14.04.
This seems to cause unresponsiveness of the `preprocess.BrainExtraction`
interface node which can get stuck while getting access to the CPU device.

More...
========

Please check `pull request 70 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/70>`_
and  `pull request 110 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/110>`_
for more change details and development discussions.


Version 2.0.1
--------------

Date: December 24, 2020

This corresponds to the release of MIAL Super-Resolution Toolkit 2.0.1,
that includes in particular the following changes.

Major change
=============

* Review `setup.py` for publication of future release of `pymialsrtk` to PyPI (See `pull request 59 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/59>`_).
* Review creation of entrypoint scripts of the container for compatibility with Singularity (See `pull request 60 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/60>`_).
* Use `MapNode` for all interfaces that apply a processing independently to a list of images (See `pull request 68 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/68>`_).
* Use the nipype sphinx extension to generate API documentation (See `pull request 65 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/65>`_).
* Review the `--manual` option flag which takes as input a directory with brain masks (See `pull request 51 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/51>`_).

New feature
=============

* ``pymialsrtk`` enables to skip different steps in the super-resolution pipeline (See `pull request 63 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/63>`_).
* Support of Singularity to execute MIALSTK on high-performance computing cluster (See `pull request 60 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/60>`_).
* ``pymialsrtk`` implements for convenience a Python wrapper that generates the Singularity command line of the BIDS App for you, prints it out for reporting purposes, and then executes it without further action needed (See `pull request 61 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/61>`_).

Software development life cycle
================================

* Add `test-python-install` job to CircleCI to test the creation of the distribution wheel to PyPI and test its installation via `pip` (See `pull request 34 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/34>`_).
* Add `deploy-pypi-release` job to CircleCI to publish the package of a new release to PyPI (See `pull request 59 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/59>`_).
* Add `build-singularity`, `test-singularity`, `deploy-singularity-latest`, and `deploy-singularity-release` jobs in CircleCI to build, test and deploy a Singularity image of `MIALSRTK` to `Sylabs.io <https://sylabs.io>`_ (See `pull request 34 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/34>`_). The tests includes:

	* Test 03: Run BIDS App on the sample `data/` BIDS dataset with the ``--manual_masks`` option without code coverage.
	* Test 04: Run BIDS App on the sample `data/` BIDS dataset with automated brain extraction (masking) without code coverage.

More...
========

Please check `pull request 53 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/53>`_ for more change details and development discussions.


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

* ``pymialsrtk``  implements an automatic brain extraction (masking) module based on a 2D U-Net (Ronneberger et al. [Ref1]_) using the pre-trained weights from Salehi et al. [Ref2]_ (See `pull request 4 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/4>`_). It is integrated in the BIDS App workflow by default.

.. [Ref1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_

.. [Ref2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

* ``pymialsrtk``  implements a module for automatic stack reference selection and ordering (masking) based on the tracking of the brain mask centroid slice by slice (See `pull request 34 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/34>`_)

* ``pymialsrtk`` implements for convenience a Python wrapper that generates the Docker command line of the BIDS App for you,
prints it out for reporting purposes, and then executes it without further action needed (See `pull request 47 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/47>`_)

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

Please check `pull request 2 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/2>`_ and `pull request 4 <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/4>`_ for more change details and development discussions.
