
.. _cmpbids:

*******************************************
MIALSRTK BIDS App and the BIDS standard
*******************************************

``MIALSRTK BIDS App`` adopts the :abbr:`BIDS (Brain Imaging Data Structure)` standard for data organization and is developed following the BIDS App standard.

This means it can be executed by running the BIDS App container image directly from the terminal or a script (See :ref:`cmdusage` section for more details). 

For more information about BIDS and BIDS-Apps, please consult the `BIDS Website <https://bids.neuroimaging.io/>`_, the `Online BIDS Specifications <https://bids-specification.readthedocs.io/en/stable/>`_, and the `BIDSApps Website <https://bids-apps.neuroimaging.io/>`_. `HeuDiConv <https://github.com/nipy/heudiconv>`_ can assist you in converting DICOM brain imaging data to BIDS. A nice tutorial can be found @ `BIDS Tutorial Series: HeuDiConv Walkthrough <http://reproducibility.stanford.edu/bids-tutorial-series-part-2a/>`_ .

.. _bidsexample:

Example BIDS dataset
=======================

For instance, a BIDS dataset should adopt the following organization, naming, and file formats:::

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
                sub-<subject_label>_T1w.nii.gz
                sub-<subject_label>_T1w.json
            ...
        ...

        code/
            participants_params.json

The BIDS App use a specific pipeline configuration file (here `participants_params.json`) which uses the following schema:::

    {
      "01": [
        { "sr-id":1,
          "stacksOrder": [1, 3, 5, 2, 4, 6],
          "paramTV": { 
            "lambdaTV": 0.75, 
            "deltatTV": 0.01 }
        }]
      "02": [
        { "sr-id":1,
          "stacksOrder": [3, 1, 2, 4],
          "paramTV": { 
            "lambdaTV": 0.7, 
            "deltatTV": 0.01 }
        }]
      ...
    } 

where:
* `stacksOrder` define the list and order od scans to be used in the reconstruction
* `lambdaTV` (regularization) and `deltaTV` (optimization time step) are parameters of the TV super-resolution algorithm

.. important:: 
    Before using any BIDS App, we highly recommend you to validate your BIDS structured dataset with the free, online `BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.
