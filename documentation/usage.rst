.. _cmdusage:

***********************
Commandline Usage
***********************

`MIALSRTK` adopts the :abbr:`BIDS (Brain Imaging Data Structure)` standard for data organization and takes as principal input the path of the dataset that is to be processed. The input dataset is required to be in *valid BIDS format*, and it must include *at least one T2w scan with anisotropic resolution per anatomical direction*. See :ref:`BIDS and BIDS App standards <cmpbids>` page that provides links for more information about BIDS and BIDS-Apps as well as an example for dataset organization and naming.


Commandline Arguments
=============================

The command to run the `MIALSRTK` follows the `BIDS-Apps <https://github.com/BIDS-Apps>`_ definition standard with an additional option for loading the pipeline configuration file.

.. argparse::
		:ref: pymialsrtk.parser.get_parser
		:prog: mialsuperresolutiontoolkit-bidsapp


.. _config:

BIDS App configuration file
-----------------------------

The BIDS App configuration file specified by the input flag `--param_file` adopts the following JSON schema::

    {
      "01": [
        { "sr-id": 1,
          ("session": 01,)
          "stacks": [1, 3, 5, 2, 4, 6],
          "paramTV": { 
            "lambdaTV": 0.75, 
            "deltatTV": 0.01 }
        },
        { "sr-id": 2,
          ("session": 01,)
          "stacks": [2, 3, 5, 4],
          "ga": 25,
          "paramTV": { 
            "lambdaTV": 0.75, 
            "deltatTV": 0.01 },
          "custom_interfaces":
            {
            "skip_svr": true,
            "do_refine_hr_mask": false,
            "skip_stacks_ordering": false,
            "do_anat_orientation": true
            }
        }]
      "02": [
        { "sr-id": 1,
          ("session": 01,)
          "stacks": [3, 1, 2, 4],
          "paramTV": { 
            "lambdaTV": 0.7, 
            "deltatTV": 0.01 }
        }]
      ...
    } 

where:
    * ``"sr-id"`` (mandatory) allows to distinguish between runs with different configurations of the same acquisition set.

    * ``"stacks"`` (optional) defines the list of scans to be used in the reconstruction. The specified order is considered if ``"skip_stacks_ordering"`` is False

    * ``"paramTV"`` (optional): ``"lambdaTV"`` (regularization), ``"deltaTV"`` (optimization time step),
``"num_iterations"``, ``"num_primal_dual_loops"``, ``"num_bregman_loops"``, ``"step_scale"``, ``"gamma"`` are parameters of the TV super-resolution algorithm.

    * ``"session"`` (optional) It MUST be specified if you have a BIDS dataset composed of multiple sessions with the *sub-XX/ses-YY* structure.

    * ``"ga"`` (optional but mandatory when ``do_anat_orientation`` is true) subject's gestational age in weeks.
    
    * ``"run_type"`` (optional): defines the type of run that should be done. It can be set between `sr` (super-resolution) and `preprocessing` (preprocessing-only). (default is ``"sr"``)
    
    * ``"custom_interfaces"`` (optional): indicates whether optional interfaces of the pipeline should be performed.

        * ``"skip_svr"`` (optional) the Slice-to-Volume Registration should be skipped in the image reconstruction. (default is False)

        * ``"do_refine_hr_mask"`` (optional) indicates whether a refinement of the HR mask should be performed. (default is False)
        
        * ``"skip_preprocessing"`` (optional) indicates whether the preprocessing stage should be skipped. A minimal preprocessing is still computed: the field-of-view is reduced based on the brain masks and the LR series are masked on the ROI. (default is False)
        .. note:: This option requires input images to be normalised in the range [0,255] prior to running the code with this option. The projection step of the TV algorithm will otherwise clip values to 255. 
        * ``"do_nlm_denoising"`` (optional) indicates whether the NLM denoising preprocessing should be performed prior to motion estimation. (default is False)

        * ``"do_reconstruct_labels"`` (optional) indicates whether the reconstruction of LR label maps should be performed together with T2w images. (default is False)

        * ``"skip_stacks_ordering"`` (optional) indicates whether the order of stacks specified in ``"stacks"`` should be kept or re-computed. (default is False)

        * ``"do_anat_orientation"`` (optional) indicates whether the alignement into anatomical planes should be performed. If True, path to a directory containing STA atlas (Gholipour et al., 2017 [1]_, [2]_) must be mounted to `/sta`. (default is False)
        
        * ``"preproc_do_registration"`` (optional) indicates whether the Slice-to-Volume Registration should be computed in the ``"preprocessing"`` run (default is False).

        * ``"do_multi_parameters"`` (optional) enables running the super-resolution reconstruction with lists of parameters. The algorithm will
        then run a grid search over all combinations of parameters. (default is False)
        
        * ``"do_srr_assessment"`` (optional) enables comparing the quality of the super-resolution reconstruction with a reference image. (default is False)
        If True, it will require a reference isotropic T2w image, mask and labels located in the data folder.
                
References
----------
.. [1] Gholipour et al.; A normative spatiotemporal MRI atlas of the fetal brain for automatic segmentation and analysis of early brain growth, Scientific Reports 7, Article number: 476 (2017). `(link to article)<http://www.nature.com/articles/s41598-017-00525-w>`_ .
.. [2] `(link to download) <http://crl.med.harvard.edu/research/fetal_brain_atlas/>`_

.. important:: 
    Before using any BIDS App, we highly recommend you to validate your BIDS structured dataset with the free, online `BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.


Running `MIALSRTK`
===================

You can run the `MIALSRTK` using the lightweight Docker or Singularity wrappers we created for convenience or you can interact directly with the Docker / Singularity Engine via the docker or singularity run command. (See :ref:`installation`)

.. role:: raw-html(raw)
   :format: html

.. admonition:: New

    You can now be aware about the adverse impact of your processing on the environment :raw-html:`&#x1F30D;`!

    With the new `--track_carbon_footprint` option of the `mialsuperresolutiontoolkit_docker` and `mialsuperresolutiontoolkit_singularity` BIDS App python wrappers, you can use `codecarbon <https://codecarbon.io/>`_ to estimate the amount of carbon dioxide (CO2) produced to execute the code by the computing resources and save the results in ``<bids_dir>/code/emissions.csv``.

    Then, to visualize, interpret and track the evolution of the CO2 emissions incurred, you can use the visualization tool of `codecarbon` aka `carbonboard` that takes as input the `.csv` created::

        carbonboard --filepath="<bids_dir>/code/emissions.csv" --port=xxxx


.. _wrapperusage:

With the wrappers
-------------------

When you run ``mialsuperresolutiontoolkit_docker``, it will generate a Docker command line for you, print it out for reporting purposes, and then execute it without further action needed, e.g.:

    .. code-block:: console

       $ mialsuperresolutiontoolkit_docker \
            /home/localadmin/data/ds001 /media/localadmin/data/ds001/derivatives \
            participant --participant_label 01 \
            --param_file /home/localadmin/data/ds001/code/participants_params.json \
            --track_carbon_footprint \
            (--openmp_nb_of_cores 4) \
            (--nipype_nb_of_cores 4)


When you run ``mialsuperresolutiontoolkit_singularity``, it will generate a Singularity command line for you, print it out for reporting purposes, and then execute it without further action needed, e.g.:

    .. code-block:: console

       $ mialsuperresolutiontoolkit_singularity \
            /home/localadmin/data/ds001 /media/localadmin/data/ds001/derivatives \
            participant --participant_label 01 \
            --param_file /home/localadmin/data/ds001/code/participants_params.json \
            --track_carbon_footprint \
            (--openmp_nb_of_cores 4) \
            (--nipype_nb_of_cores 4)


With the Docker / Singularity Engine
--------------------------------------

If you need a finer control over the container execution, or you feel comfortable with the Docker or Singularity Engine, avoiding the extra software layer of the wrapper might be a good decision.

For instance, the previous call to the ``mialsuperresolutiontoolkit_docker`` wrapper corresponds to:

  .. parsed-literal::

    $ docker run -t --rm -u $(id -u):$(id -g) \\
            -v /home/localadmin/data/ds001:/bids_dir \\
            -v /media/localadmin/data/ds001/derivatives:/output_dir \\
            (-v /path/to/CRL_Fetal_Brain_Atlas:/sta \\)
            sebastientourbier/mialsuperresolutiontoolkit:|vrelease| \\
            /bids_dir /output_dir participant --participant_label 01 \\
            --param_file /bids_dir/code/participants_params.json \\
            (--openmp_nb_of_cores 4) \\
            (--nipype_nb_of_cores 4)

.. note:: We use the `-v /path/to/local/folder:/path/inside/container` docker run option to access local files and folders inside the container such that the local directory of the input BIDS dataset (here: ``/home/localadmin/data/ds001``) and the output directory (here: ``/media/localadmin/data/ds001/derivatives``) used to process are mapped to the folders ``/bids_dir`` and ``/output_dir`` in the container respectively.

The previous call to the ``mialsuperresolutiontoolkit_singularity`` wrapper corresponds to:

  .. parsed-literal::

    $ singularity run --containall \\
            --bind /home/localadmin/data/ds001:/bids_dir \\
            --bind /media/localadmin/data/ds001/derivatives:/output_dir \\
            library://tourbier/default/mialsuperresolutiontoolkit:|vrelease| \\
            /bids_dir /output_dir participant --participant_label 01 \\
            --param_file /bids_dir/code/participants_params.json \\
            (--openmp_nb_of_cores 4) \\
            (--nipype_nb_of_cores 4)

.. note:: Similarly as with Docker, we use the `--bind /path/to/local/folder:/path/inside/container` singularity run option to access local files and folders inside the container such that the local directory of the input BIDS dataset (here: ``/home/localadmin/data/ds001``) and the output directory (here: ``/media/localadmin/data/ds001/derivatives``) used to process are mapped to the folders ``/bids_dir`` and ``/output_dir`` in the container respectively.


Debugging
=========

Logs are outputted into
``<output dir>/nipype/sub-<participant_label>/anatomical_pipeline/rec<srId>/pypeline.log``.


Support, bugs and new feature requests
=======================================

All bugs, concerns and enhancement requests for this software are managed on GitHub and can be submitted at `https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.


Not running on a local machine? - Data transfer
===============================================

If you intend to run `MIALSRTK` on a remote system, you will need to
make your data available within that system first. Comprehensive solutions such as `Datalad
<http://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands. Datalad also performs version control over your data.
