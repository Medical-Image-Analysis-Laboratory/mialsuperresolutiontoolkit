*****************************************
Outputs of MIALSRTK BIDS App
*****************************************

Processed, or derivative, data are outputed to ``<bids_dataset/derivatives>/`` and follow the :abbr:`BIDS (Brain Imaging Data Structure)` v1.4.1 standard (see `BIDS Derivatives <https://bids-specification.readthedocs.io/en/v1.4.1/05-derivatives/01-introduction.html>`_) whenever possible.  

BIDS derivatives entities
--------------------------

.. tabularcolumns:: |l|p{5cm}|

+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| **Entity**               | **Description**                                                                                                     |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| ``sub-<subject_label>``  | Label to distinguish different subject                                                                              |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| ``ses-<session_label>``  | Label to distinguish different T2w scan acquisition session                                                         |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| ``run-<run_label>``      | Label to distinguish different T2w scans                                                                            |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| ``rec-<recon_label>``    | Label to distinguish images reconstructed using scattered data interpolation (SDI) or super-resolution (SR) methods |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+
| ``id-<srr_id>``          | Label to distinguish outputs of multiple reconstructions with different configuration                               |
+--------------------------+---------------------------------------------------------------------------------------------------------------------+

See `Original BIDS Entities Appendix <https://bids-specification.readthedocs.io/en/v1.4.1/99-appendices/09-entities.html>`_ for more description.

.. note:: A new entity ``id-<srr_id>`` has been introduced to distinguish between outputs when the pipeline is run with multiple configurations (such a new order of scans) on the same subject.

Main MIALSRTK BIDS App Derivatives
==========================================

Main outputs produced by MIALSRTK BIDS App are written to ``<bids_dataset/derivatives>/pymialsrtk-<variant>/sub-<subject_label>/(_ses-<session_label>/)``. An execution log of the full workflow is saved as `sub-<subject_label>(_ses-<session_label>)_id-<srr_id>_log.txt``.

Anatomical derivatives
------------------------
* Anatomical derivatives are placed in each subject's ``anat/`` subfolder, including:

    * The brain masks of the T2w scans:

        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-01_id-<srr_id>_desc-brain_mask.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-02_id-<srr_id>_desc-brain_mask.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-03_id-<srr_id>_desc-brain_mask.nii.gz``
        - ...

    * The preprocessed T2w scans used for slice motion estimation and scattered data interpolation (SDI) reconstruction:

        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-01_id-<srr_id>_desc-preprocSDI_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-02_id-<srr_id>_desc-preprocSDI_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-03_id-<srr_id>_desc-preprocSDI_T2w.nii.gz``
        - ...
        
    * The preprocessed T2w scans used for super-resolution reconstruction:

        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-01_id-<srr_id>_desc-preprocSR_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-02_id-<srr_id>_desc-preprocSR_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_run-03_id-<srr_id>_desc-preprocSR_T2w.nii.gz``
        - ...
   
    * The high-resolution image reconstructed by SDI:

        - ``anat/sub-<subject_label>(_ses-<session_label>)_rec-SDI_id-<srr_id>_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_rec-SDI_id-<srr_id>_T2w.json``

    * The high-resolution image reconstructed by SDI:

        - ``anat/sub-<subject_label>(_ses-<session_label>)_rec-SR_id-<srr_id>_T2w.nii.gz``
        - ``anat/sub-<subject_label>(_ses-<session_label>)_rec-SR_id-<srr_id>_T2w.json``

* The slice-to-volume registration transform of each T2W scans estimated during slice motion estimation and SDI reconstruction and used in the super-resolution forward model are placed in each subject's ``xfm/`` subfolder:

    - ``xfm/sub-<subject_label>(_ses-<session_label>)_run-1_id-<srr_id>_T2w_from-origin_to-SDI_mode-image_xfm.txt``
    - ``xfm/sub-<subject_label>(_ses-<session_label>)_run-2_id-<srr_id>_T2w_from-origin_to-SDI_mode-image_xfm.txt``
    - ``xfm/sub-<subject_label>(_ses-<session_label>)_run-3_id-<srr_id>_T2w_from-origin_to-SDI_mode-image_xfm.txt``
    - ...

Nipype Workflow Derivatives
==========================================

The execution of the Nipype workflow (pipeline) involves the creation of a number of intermediate outputs which are written to ``<bids_dataset/derivatives>/nipype/sub-<subject_label>/rec-<sr_id>/srr_pipeline``: 

.. image:: images/nipype_wf_derivatives.png
    :width: 888
    :align: center

To enhance transparency on how data is processed, outputs include a pipeline execution graph saved as ``srr_pipeline/graph.png`` which summarizes all processing nodes involves in the given processing pipeline:

.. image:: images/nipype_wf_graph.png
    :width: 888
    :align: center

Execution details (data provenance) of each interface (node) of a given pipeline are reported in ``srr_pipeline/<interface_name>/_report/report.rst``

.. image:: images/nipype_node_report.png
    :width: 888
    :align: center
