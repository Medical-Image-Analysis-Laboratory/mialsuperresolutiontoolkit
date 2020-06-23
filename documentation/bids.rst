
.. _cmpbids:

*******************************************
BIDS and BIDS App standards
*******************************************

``MIALSRTK BIDS App`` adopts the :abbr:`BIDS (Brain Imaging Data Structure)` standard for data organization and is developed following the BIDS App standard. This means that ``MIALSRTK BIDS App`` handles dataset formatted following the BIDS App standard and provides a processing workflow containerized in Docker container image (promoting portability and reproduciblity) that can be run with a set of arguments defined by the BIDS App standard directly from the terminal or a script (See :ref:`cmdusage` section for more details). 

For more information about BIDS and BIDS-Apps, please consult the `BIDS Website <https://bids.neuroimaging.io/>`_, the `Online BIDS Specifications <https://bids-specification.readthedocs.io/en/stable/>`_, and the `BIDSApps Website <https://bids-apps.neuroimaging.io/>`_. `HeuDiConv <https://github.com/nipy/heudiconv>`_ can assist you in converting DICOM brain imaging data to BIDS. A nice tutorial can be found @ `BIDS Tutorial Series: HeuDiConv Walkthrough <http://reproducibility.stanford.edu/bids-tutorial-series-part-2a/>`_ .

.. _bidsexample:

BIDS dataset schema
=======================

The BIDS App accepts BIDS datasets that adopt the following organization, naming, and file formats::

    ds-example/
        
        README
        CHANGES
        participants.tsv
        dataset_description.json
        
        sub-01/
            anat/
                sub-01_run-1_T2w.nii.gz
                sub-01_run-1_T2w.json
                sub-01_run-2_T2w.nii.gz
                sub-01_run-2_T2w.json
                ...
        
        ...

        sub-<subject_label>/
            anat/
                sub-<subject_label>_run-1_T2w.nii.gz
                sub-<subject_label>_run-1_T2w.json
                sub-<subject_label>_run-2_T2w.nii.gz
                sub-<subject_label>_run-2_T2w.json
                ...
            ...
        ...

        code/
            participants_params.json

where ``participants_params.json`` is the MIALSRTK BIDS App configuration file, which following a specific schema (See :ref:`config schema <config>`), and which defines multiple processing parameters (such as the ordered list of scans or the weight of regularization).

.. important:: 
    Before using any BIDS App, we highly recommend you to validate your BIDS structured dataset with the free, online `BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.
