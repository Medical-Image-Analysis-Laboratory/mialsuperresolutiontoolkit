# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Entrypoint point script of the BIDS APP."""

import os
import sys
import json
from glob import glob
# from traits.api import *

import multiprocessing

# Import the super-resolution pipeline
from pymialsrtk.parser import get_parser
from pymialsrtk.pipelines.anatomical import SRReconPipeline
from pymialsrtk.pipelines.anatomical import PreprocessingPipeline


def return_default_nb_of_cores(nb_of_cores, openmp_proportion=2):
    """Function that returns the number of cores used by OpenMP and Nipype by default.

    Given ``openmp_proportion``, the proportion of cores dedicated to OpenMP threads,
    ``openmp_nb_of_cores`` and ``nipype_nb_of_cores`` are set by default to the following:

    .. code-block:: python

        openmp_nb_of_cores = nb_of_cores // openmp_proportion
        nipype_nb_of_cores = nb_of_cores // openmp_nb_of_cores

    where ``//`` is the integer division operator.

    Parameters
    ----------
    nb_of_cores : int
        Number of cores available on the computer
    openmp_proportion : int
        Proportion of cores dedicated to OpenMP threads

    Returns
    -------
    openmp_nb_of_cores : int
        Number of cores used by default by openmp

    nipype_nb_of_cores : int
        Number of cores used by default by openmp
    """
    openmp_nb_of_cores = nb_of_cores // openmp_proportion
    nipype_nb_of_cores = nb_of_cores // openmp_nb_of_cores

    return openmp_nb_of_cores, nipype_nb_of_cores


def check_and_return_valid_nb_of_cores(openmp_nb_of_cores, nipype_nb_of_cores, openmp_proportion=2):
    """Function that checks and returns a valid number of cores used by OpenMP and Nipype.

    If the number of cores available is exceeded by one of these variables or by the multiplication of the two,
    ``openmp_nb_of_cores`` and ``nipype_nb_of_cores`` are reset by the :func:`return_default_nb_of_cores()` to the following:

    .. code-block:: python

        openmp_nb_of_cores = nb_of_cores // 2
        nipype_nb_of_cores = nb_of_cores - openmp_nb_of_cores

    Parameters
    ----------
    openmp_nb_of_cores : int
        Number of cores used by openmp that was initially set

    nipype_nb_of_cores : int
        Number of cores used by Niype that was initially set

    Returns
    -------
    openmp_nb_of_cores : int
        Valid number of cores used by openmp

    nipype_nb_of_cores : int
        Valid number of cores used by Niype
    """
    nb_of_cores = multiprocessing.cpu_count()

    # Handles all the scenarios for values of nb_of_cores, openmp_nb_of_cores
    # and nipype_nb_of_cores and make the correction if needed.
    if nb_of_cores == 1:
        openmp_nb_of_cores = 1
        nipype_nb_of_cores = 1
    else:
        if openmp_nb_of_cores == 0 and nipype_nb_of_cores == 0:
            openmp_nb_of_cores, nipype_nb_of_cores = return_default_nb_of_cores(nb_of_cores, openmp_proportion)

        elif openmp_nb_of_cores > 0 and nipype_nb_of_cores == 0:
            if openmp_nb_of_cores > nb_of_cores:
                print(f"WARNING: Value of {openmp_nb_of_cores} set by"
                      f"'--openmp_nb_of_cores' is bigger than the number of "
                      f"cores available ({nb_of_cores}) and will be reset.")
                openmp_nb_of_cores = nb_of_cores
                nipype_nb_of_cores = 1
            else:
              openmp_nb_of_cores = openmp_nb_of_cores
              nipype_nb_of_cores = nb_of_cores // openmp_nb_of_cores

        elif openmp_nb_of_cores == 0 and nipype_nb_of_cores > 0:
            if nipype_nb_of_cores > nb_of_cores:
                print(f"WARNING: Value of {nipype_nb_of_cores} set by"
                      f"'--nipype_nb_of_cores' is bigger than the number of "
                      "cores available ({nb_of_cores}) and will be reset.")
                nipype_nb_of_cores = nb_of_cores
                openmp_nb_of_cores = 1
            else:
              nipype_nb_of_cores = nipype_nb_of_cores
              openmp_nb_of_cores = nb_of_cores // nipype_nb_of_cores

        elif openmp_nb_of_cores > 0 and nipype_nb_of_cores > 0:
            if nipype_nb_of_cores > nb_of_cores:
                if openmp_nb_of_cores > nb_of_cores:
                    print(f"WARNING: Value of {nipype_nb_of_cores} and "
                          f"{openmp_nb_of_cores} set by '--openmp_nb_of_cores'"
                          f"and '--nipype_nb_of_cores' are both bigger than "
                          f"the number of cores available ({nb_of_cores})"
                          f"and will be reset.")
                    openmp_nb_of_cores, nipype_nb_of_cores = \
                        return_default_nb_of_cores(nb_of_cores,
                                                   openmp_proportion)
                else:
                    if (openmp_nb_of_cores * nipype_nb_of_cores) > nb_of_cores:
                        print(f"WARNING: Multiplication of "
                              f"{nipype_nb_of_cores} and {openmp_nb_of_cores} "
                              "set by '--nipype_nb_of_cores' and "
                              "'--nipype_nb_of_cores' is bigger than the "
                              "number of cores available ({nb_of_cores}) "
                              "and will be reset.")
                        openmp_nb_of_cores, nipype_nb_of_cores = \
                            return_default_nb_of_cores(nb_of_cores,
                                                       openmp_proportion)
            else:
                if openmp_nb_of_cores > nb_of_cores:
                    print(f"WARNING: Value of {openmp_nb_of_cores} set by "
                          f"'--openmp_nb_of_cores' is bigger than the number "
                          f"of cores available ({nb_of_cores}) and will "
                          "be reset.")
                    openmp_nb_of_cores, nipype_nb_of_cores = \
                        return_default_nb_of_cores(nb_of_cores,
                                                   openmp_proportion)
                else:
                    if (openmp_nb_of_cores * nipype_nb_of_cores) > nb_of_cores:
                        print(f"WARNING: Multiplication of "
                              f"{nipype_nb_of_cores} and {openmp_nb_of_cores}"
                              "set by '--nipype_nb_of_cores' and "
                              "'--nipype_nb_of_cores' is bigger than the "
                              "number of cores available ({nb_of_cores}) "
                              "and will be reset.")
                        openmp_nb_of_cores, nipype_nb_of_cores = \
                            return_default_nb_of_cores(nb_of_cores,
                                                       openmp_proportion)

    return openmp_nb_of_cores, nipype_nb_of_cores


def main(bids_dir, output_dir,
         subject,
         session,
         run_type,
         p_ga=None,
         p_stacks=None,
         param_TV=None,
         sr_id=None,
         masks_derivatives_dir="",
         labels_derivatives_dir='',
         masks_desc=None,
         dict_custom_interfaces=None,
         verbose=None,
         nipype_number_of_cores=1,
         openmp_number_of_cores=1,
         memory=0
         ):
    """Main function that creates and executes the workflow of the BIDS App on
    one subject.

    It creates an instance of the class :class:`pymialsrtk.pipelines.anatomical
    srr.AnatomicalPipeline`, which is then used to create and execute the
    workflow of the super-resolution reconstruction pipeline.

    Parameters
    ----------
    bids_dir : string
        BIDS root directory (required)

    output_dir : string
        Output derivatives directory (required)

    subject : string
        Subject ID (in the form ``sub-XX``)

    session : string
        Session ID if applicable (in the form ``ses-YY``)

    p_stacks : list(int)
        List of stack to be used in the reconstruction. The specified order is
        kept if `skip_stacks_ordering` is True.

    param_TV dict : {"deltatTV": float, "lambdaTV": float,
                     "primal_dual_loops": int}
        Dictionary of Total-Variation super-resolution optimizer parameters

    sr_id : string
        ID of the reconstruction useful to distinguish when multiple
        reconstructions with different order of stacks are run on the
        same subject

    masks_derivatives_dir : string
        directory basename in BIDS directory derivatives where to search
        for masks (optional)

    masks_desc : string
        BIDS description tag of masks to use (optional)

    dict_custom_interfaces : {"do_refine_hr_mask": False,
        "do_nlm_denoising": False, "skip_stacks_ordering": False,
        "preproc_do_registration": False}
        Dictionary that customize the workflow (skip interfaces).

    nipype_number_of_cores : int
        Number of cores / CPUs used by the Nipype worflow execution engine

    openmp_number_of_cores : int
        Number of threads used by OpenMP

    memory : int
        Maximal amount of memory used by the workflow
        (Default: 0, workflow uses all available memory)

    """

    if param_TV is None:
        param_TV = dict()

    subject = "sub-" + subject
    if session is not None:
        session = "ses-" + session

    if sr_id is None:
        sr_id = 1
    # Initialize an instance of AnatomicalPipeline
    if run_type == "sr":
        pipeline = SRReconPipeline(
            bids_dir,
            output_dir,
            subject,
            p_ga,
            p_stacks,
            sr_id,
            session,
            param_TV,
            masks_derivatives_dir,
            labels_derivatives_dir,
            masks_desc,
            p_dict_custom_interfaces=dict_custom_interfaces,
            p_verbose=verbose,
            p_openmp_number_of_cores=openmp_number_of_cores,
            p_nipype_number_of_cores=nipype_number_of_cores
            )
    elif run_type == "preprocessing":
        pipeline = PreprocessingPipeline(
            bids_dir,
            output_dir,
            subject,
            p_ga,
            p_stacks,
            sr_id,
            session,
            masks_derivatives_dir,
            masks_desc,
            p_dict_custom_interfaces=dict_custom_interfaces,
            p_verbose=verbose,
            p_openmp_number_of_cores=openmp_number_of_cores,
            p_nipype_number_of_cores=nipype_number_of_cores,
            )
    else:
        raise ValueError(f"Invalid run_type {run_type}."
                         f"Please choose from ('sr,'preprocessing').")

    # Create the super resolution Nipype workflow
    pipeline.create_workflow()

    # Execute the workflow
    res = pipeline.run(memory=memory)

    return res


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    openmp_nb_of_cores = args.openmp_nb_of_cores
    nipype_nb_of_cores = args.nipype_nb_of_cores

    # Check values set for the number of cores and reset them if invalid
    openmp_nb_of_cores, nipype_nb_of_cores = \
        check_and_return_valid_nb_of_cores(openmp_nb_of_cores,
                                           nipype_nb_of_cores)
    print(f"INFO: Number of cores used by Nipype engine "
          f"set to {nipype_nb_of_cores}")

    os.environ["OMP_NUM_THREADS"] = str(openmp_nb_of_cores)
    print("INFO: Environment variable OMP_NUM_THREADS set to: "
          "{}".format(os.environ["OMP_NUM_THREADS"]))

    with open(args.param_file, 'r') as f:
        participants_params = json.load(f)

    subjects_to_analyze = []
    # only for a subset of subjects
    if args.participant_label:
        subjects_to_analyze = args.participant_label
    # for all subjects
    else:
        subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
        subjects_to_analyze = [
            subject_dir.split("-")[-1] for subject_dir in subject_dirs
        ]
    failed_dict = {}
    for sub in subjects_to_analyze:
        failed_dict[sub] = []
        if sub in participants_params.keys():

            sr_list = participants_params[sub]
            print(sr_list)

            for sr_params in sr_list:

                sr_id = sr_params["sr-id"] if "sr-id" in sr_params.keys() \
                    else None
                ses = sr_params["session"] if "session" in sr_params.keys() \
                    else None
                ga = sr_params["ga"] if "ga" in sr_params.keys() else None
                stacks = sr_params["stacks"] if "stacks" in sr_params.keys() \
                    else None
                param_TV = sr_params["paramTV"] if "paramTV" in \
                    sr_params.keys() else None
                masks_desc = sr_params["masks_desc"] if "masks_desc" in \
                    sr_params.keys() else None

                dict_custom_interfaces = sr_params["custom_interfaces"] \
                    if "custom_interfaces" in sr_params.keys() else None

                if sr_id is None:
                    e = f'Subject {sub} was not processed ' \
                        f'because of missing parameters.'
                    failed_dict[sub] += [e]
                    print(e)
                    continue
                try:
                    res = main(bids_dir=args.bids_dir,
                               output_dir=args.output_dir,
                               subject=sub,
                               session=ses,
                               run_type=args.run_type,
                               p_ga=ga,
                               p_stacks=stacks,
                               param_TV=param_TV,
                               sr_id=sr_id,
                               masks_derivatives_dir=args.masks_derivatives_dir,
                               labels_derivatives_dir=args.labels_derivatives_dir,
                               masks_desc=masks_desc,
                               dict_custom_interfaces=dict_custom_interfaces,
                               verbose=args.verbose,
                               nipype_number_of_cores=nipype_nb_of_cores,
                               openmp_number_of_cores=openmp_nb_of_cores,
                               memory=args.memory)
                except Exception as e:
                    e = f"Subject {sub} with parameters {sr_params} failed "\
                        f"with message \n\t {e}"
                    failed_dict[sub] += [e]
                    print(e)
        else:
            e = f"Subject {sub} was not processed because of missing configuration."
            failed_dict[sub] += [e]
            print(e)

    if not all([v==[] for v in failed_dict.values()]):
        print(f"WARNING: Some runs failed.")
        for sub, v in failed_dict.items():
            for error in v:
                print("-> ", error)

    sys.exit(0)
