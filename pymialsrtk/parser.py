# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""MIALSRTK BIDS App Commandline Parser."""

from pymialsrtk.info import __version__
from pymialsrtk.info import __release_date__


def get_parser():
    """Create and return the parser object of the BIDS App."""

    import argparse

    p = argparse.ArgumentParser(description="Argument parser of the MIALSRTK BIDS App")

    p.add_argument("bids_dir", help="The directory with the input dataset " "formatted according to the BIDS standard.")

    p.add_argument(
        "output_dir",
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the "
        "participant level analysis.",
    )

    p.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. " "Only participant is available",
        choices=["participant"],
    )

    p.add_argument(
        "--participant_label",
        help="The label(s) of the participant(s) that should be analyzed. "
        "The label corresponds to sub-<participant_label> from the BIDS spec "
        '(so it does not include "sub-"). If this parameter is not '
        "provided all subjects should be analyzed. Multiple "
        "participants can be specified with a space separated list.",
        nargs="+",
    )

    p.add_argument(
        "--param_file",
        help="Path to a JSON file containing subjects' exams "
        "information and super-resolution total variation parameters.",
        default="/bids_dir/code/participants_params.json",
        type=str,
    )

    p.add_argument(
        "--openmp_nb_of_cores",
        help="Specify number of cores used by OpenMP threads "
        "Especially useful for NLM denoising and slice-to-volume registration. "
        "(Default: 0, meaning it will be determined automatically)",
        default=0,
        type=int,
    )

    p.add_argument(
        "--nipype_nb_of_cores",
        help="Specify number of cores used by the Niype workflow library to distribute "
        "the execution of independent processing workflow nodes (i.e. interfaces) "
        "(Especially useful in the case of slice-by-slice bias field correction and "
        "intensity standardization steps for example). "
        "(Default: 0, meaning it will be determined automatically)",
        default=0,
        type=int,
    )

    p.add_argument(
        "--memory",
        help="Limit the workflow to using the amount of specified memory [in gb] "
        "(Default: 0, the workflow memory consumption is not limited)",
        default=0,
        type=int,
    )

    p.add_argument(
        "--masks_derivatives_dir",
        help="Use manual brain masks found in "
        "``<output_dir>/<masks_derivatives_dir>/`` directory",
    )

    p.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"BIDS-App MIALSRTK version {__version__} (Released: {__release_date__})",
    )
    return p
