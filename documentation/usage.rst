.. _cmdusage:

***********************
Commandline Usage
***********************

`MIALSRTK BIDS App` adopts the :abbr:`BIDS (Brain Imaging Data Structure)` standard for data organization and takes as principal input the path of the dataset that is to be processed. The input dataset is required to be in *valid BIDS format*, and it must include *at least one T2w scan with anisotropic resolution per anatomical direction*. See :ref:`BIDS and BIDS App standards <cmpbids>` page that provides links for more information about BIDS and BIDS-Apps as well as an example for dataset organization and naming.


Commandline Arguments
=============================

The command to run the `MIALSRTK BIDS App` follows the `BIDS-Apps <https://github.com/BIDS-Apps>`_ definition standard with an additional option for loading the pipeline configuration file.

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
          "stacksOrder": [1, 3, 5, 2, 4, 6],
          "paramTV": { 
            "lambdaTV": 0.75, 
            "deltatTV": 0.01 }
        }],
      "01": [
        { "sr-id": 2,
          ("session": 01,)
          "stacksOrder": [2, 3, 5, 4],
          "paramTV": { 
            "lambdaTV": 0.75, 
            "deltatTV": 0.01 }
        }]
      "02": [
        { "sr-id": 1,
          ("session": 01,)
          "stacksOrder": [3, 1, 2, 4],
          "paramTV": { 
            "lambdaTV": 0.7, 
            "deltatTV": 0.01 }
        }]
      ...
    } 

where:
    * ``"sr-id"`` allows to distinguish between runs with different configurations of the same acquisition set.

    * ``"stacksOrder"`` defines the list and order od scans to be used in the reconstruction.

    * ``"lambdaTV"`` (regularization) and `deltaTV` (optimization time step) are parameters of the TV super-resolution algorithm.

    * ``"session"`` MUST be specified if you have a BIDS dataset composed of multiple sessions with the *sub-XX/ses-YY* structure.


.. important:: 
    Before using any BIDS App, we highly recommend you to validate your BIDS structured dataset with the free, online `BIDS Validator <http://bids-standard.github.io/bids-validator/>`_.


Running the `MIALSRTK BIDS App`
==================================

You can run the `MIALSRTK BIDS App` using a lightweight wrapper we created for convenience or you can interact directly with the Docker Engine via the docker run command line. (See :ref:`installation`)

.. _wrapperusage:

With the ``mialsuperresolutiontoolkit_bidsapp`` wrapper
--------------------------------------------------------

When you run ``mialsuperresolutiontoolkit_bidsapp``, it will generate a Docker command line for you,
print it out for reporting purposes, and then execute it without further action needed, e.g.:

    .. code-block:: console

       $ mialsuperresolutiontoolkit_bidsapp \\
            /home/localadmin/data/ds001 /media/localadmin/data/ds001/derivatives \\
            participant --participant_label 01 \\
            --param_file /home/localadmin/data/ds001/code/participants_params.json \\
            (--openmp_nb_of_cores 4) \\
            (--nipype_nb_of_cores 4)


Directly with the Docker Engine
--------------------------------

If you need a finer control over the container execution, or you feel comfortable with the Docker Engine, avoiding the extra software layer of the wrapper might be a good decision. In this case, previous call to the ``mialsuperresolutiontoolkit_bidsapp`` wrapper corresponds to:

  .. parsed-literal::

    $ docker run -t --rm -u $(id -u):$(id -g) \\
            -v /home/localadmin/data/ds001:/bids_dir \\
            -v /media/localadmin/data/ds001/derivatives:/output_dir \\
            sebastientourbier/mialsuperresolutiontoolkit-bidsapp:|release| \\
            /bids_dir /output_dir participant --participant_label 01 \\
            --param_file /bids_dir/code/participants_params.json \\
            (--openmp_nb_of_cores 4) \\
            (--nipype_nb_of_cores 4)

.. note:: The local directory of the input BIDS dataset (here: ``/home/localadmin/data/ds001``) and the output directory (here: ``/media/localadmin/data/ds001/derivatives``) used to process have to be mapped to the folders ``/bids_dir`` and ``/output_dir`` respectively using the `-v` docker run option. 


Debugging
=========

Logs are outputted into
``<output dir>/nipype/sub-<participant_label>/anatomical_pipeline/rec<srId>/pypeline.log``.


Support, bugs and new feature requests
=======================================

All bugs, concerns and enhancement requests for this software are managed on GitHub and can be submitted at `https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.


Not running on a local machine? - Data transfer
===============================================

If you intend to run the `MIALSRTK BIDS App` on a remote system, you will need to
make your data available within that system first. Comprehensive solutions such as `Datalad
<http://www.datalad.org/>`_ will handle data transfers with the appropriate
settings and commands. Datalad also performs version control over your data.
