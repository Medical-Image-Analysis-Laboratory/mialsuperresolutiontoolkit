.. _cmdusage:

***********************
Commandline Usage
***********************

``MIALSRTK BIDS App`` adopts the :abbr:`BIDS (Brain Imaging Data Structure)` standard for data organization and takes as principal input the path of the dataset that is to be processed. The input dataset is required to be in valid `BIDS` format, and it must include at least one T2w scan with anisotropic resolution per anatomical direction. See :ref:`cmpbids` page that provides links for more information about BIDS and BIDS-Apps as well as an example for dataset organization and naming.

Commandline Arguments
=============================

The command to run the ``MIALSRTK BIDS App`` follows the `BIDS-Apps <https://github.com/BIDS-Apps>`_ definition standard with an additional option for loading the pipeline configuration file.

.. argparse::
		:ref: pymialsrtk.parser.get_parser
		:prog: mialsuperresolutiontoolkit-bidsapp

.. _config:

The pipeline configuration file adopts a specific schema which is the following:::

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
    * ``stacksOrder`` define the list and order od scans to be used in the reconstruction

    * ``lambdaTV`` (regularization) and ``deltaTV` (optimization time step) are parameters of the TV super-resolution algorithm

.. important:: 
    Before using any BIDS App, we highly recommend you to validate your BIDS structured dataset with the free, online `BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.

Participant Level Analysis
===========================
To run the docker image in participant level mode (for one participant):

  .. parsed-literal::

    $ docker run -t --rm -u $(id -u):$(id -g) \\
            -v /home/localadmin/data/ds001:/bids_dir \\
            -v /media/localadmin/data/ds001/derivatives:/output_dir \\
            sebastientourbier/mialsuperresolutiontoolkit-bidsapp:|release| \\
            /bids_dir /output_dir participant --participant_label 01 \\(--session_label 01 \\)
          	--param_file /bids_dir/code/participants_params.json \\
            (--number_of_cores 1)

.. note:: The local directory of the input BIDS dataset (here: ``/home/localadmin/data/ds001``) and the output directory (here: ``/media/localadmin/data/ds001/derivatives``) used to process have to be mapped to the folders ``/bids_dir`` and ``/output_dir`` respectively using the ``-v`` docker run option. 

Debugging
=========

Logs are outputted into
``<output dir>/nipype/sub-<participant_label>/anatomical_pipeline/rec<srId>/pypeline.log``.

Support, bugs and new feature requests
=======================================

All bugs, concerns and enhancement requests for this software are managed on GitHub and can be submitted at `https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.


Not running on a local machine? - Data transfer
===============================================

If you intend to run the ``MIALSRTK BIDS App`` on a remote system, you will need to
make your data available within that system first. Comprehensive solutions such as `Datalad
<http://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands. Datalad also performs version control over your data.
