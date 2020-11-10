**************
Changes
**************

Version 2.0.0
--------------

Date: Month XX, 2020 (To be updated)

This corresponds to the first release of the second version of the MIAL Super-Resolution Toolkit, which has evolved massively over the last years in terms of the underlying codebase and the scope of the functionality provided, following recent advances in standardization of neuroimaging data organization and processing workflows.


*Major changes*

* Adoption of the Brain Imaging Data Structure standard for data organization (See :ref:`BIDS and BIDS App standards <cmpbids>` for more details)

* Creation of the ``pymialsrtk`` Python workflow library, built on top of the Nipype dataflow library which implements interfaces to all C++ MIALSRTK tools connected in a common workflow to perform super-resolution reconstruction of fetal brain MRI with data provenance and execution detail recordings. (See :ref:` API Documentation <api-doc>`)

* Docker image encapsulting MIALSRTK now distributed as a `BIDS App`, a standard for containerized workflow that handles BIDS datasets with a set of predefined commandline input argument. (See :ref:`BIDS App Commadline Usage <cmdusage>` for more details)


*New feature*

* Automatic brain extraction (masking) based on a 2D U-Net (Ronneberger et al. [1]_) using the pre-trained weights from Salehi et al. [2]_.

References
.. [1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_
.. [2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_


*Software development life cycle*

* Use `Codacy <https://www.codacy.com/>`_ to support code reviews and monitor code quality over time.

* Use `coveragepy <https://coverage.readthedocs.io/en/coverage-5.2/>`_  in CircleCI during regression tests of the BIDS app and create code coverage reports published on our `Codacy project page <https://app.codacy.com/gh/connectomicslab/connectomemapper3/dashboard>`_.

* Adopt CircleCI for continuous integration testing of the BIDS App that runs the following regression tests:
	* Test 01: Run BIDS App on the sample `data/` BIDS dataset with the ``--manual_masks`` option.
	* Test 02: Run BIDS App on the sample `data/` BIDS dataset with automated brain extraction (masking).


*Documentation*

* Installation instructions and Python API documentation of ```` are now generated on readthedocs at: https://mialsrtk.readthedocs.io/


Please check the `pull request 2 page <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/2>`_ and `pull request 4 page <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pull/4>`_for more change details and development discussions.
