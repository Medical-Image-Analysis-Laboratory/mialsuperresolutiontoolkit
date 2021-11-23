#!/usr/bin/env python
#
# Copyright © 2016-2020
# Medical Image Analysis Laboratory,
# University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland,
# and Contributors
#
#  This software is distributed under the open-source license Modified BSD.

"""This module defines the `mialsuperresolutiontoolkit_bidsapp_docker` script that wraps calls to the Docker BIDS APP image."""

# General imports
import sys
from pathlib import Path
import logging

# Carbon footprint
from codecarbon import EmissionsTracker

# Own imports
from pymialsrtk.info import __version__
from pymialsrtk.parser import get_parser
from pymialsrtk.interfaces.utils import run


def create_docker_cmd(args):
    """Function that creates and returns the BIDS App docker run command.

    Parameters
    ----------
    args : dict
        Dictionary of parsed input argument in the form::

            {
                'bids_dir': "/path/to/bids/dataset/directory",
                'output_dir': "/path/to/output/directory",
                'param_file': "/path/to/configuration/parameter/file",
                'analysis_level': "participant",
                'participant_label': ['01', '02', '03'],
                'openmp_nb_of_cores': 1,
                'nipype_nb_of_cores': 1,
                'masks_derivatives_dir': 'manual_masks'
            }

    Returns
    -------
    cmd : string
        String containing the command to be run via `subprocess.run()`
    """
    # Docker run command prelude
    cmd = 'docker run -t --rm '
    cmd += '-u $(id -u):$(id -g) '
    cmd += f'-v {args.bids_dir}:/bids_dir '
    cmd += f'-v {args.output_dir}:/output_dir '
    cmd += f'-v {args.param_file}:/bids_dir/code/participants_params.json '
    cmd += f'sebastientourbier/mialsuperresolutiontoolkit-bidsapp:v{__version__} '

    # Standard BIDS App inputs
    cmd += '/bids_dir '
    cmd += '/output_dir '
    cmd += f'{args.analysis_level} '
    cmd += '--participant_label '
    for label in args.participant_label:
        cmd += f'{label} '

    # MIALSRTK BIDS App inputs
    cmd += '--param_file /bids_dir/code/participants_params.json '
    if args.masks_derivatives_dir and args.masks_derivatives_dir != '':
        cmd += f'--masks_derivatives_dir {args.masks_derivatives_dir} '
    cmd += f'--openmp_nb_of_cores {args.openmp_nb_of_cores} '
    cmd += f'--nipype_nb_of_cores {args.nipype_nb_of_cores}'

    return cmd


def main():
    """Main function that creates and executes the BIDS App docker command.

    Returns
    -------
    exit_code : {0, 1}
        An exit code given to `sys.exit()` that can be:

            * '0' in case of successful completion

            * '1' in case of an error
    """
    # Create and parse arguments
    logging.getLogger("codecarbon").disabled = True  # Comment this line for debug
    parser = get_parser()
    parser.add_argument(
        "--codecarbon_output_dir",
        help="Directory path in which `codecarbon` saves a CSV file called "
             "`emissions.csv` reporting carbon footprint details of the overall run "
             "(Defaults to user’s home directory)",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    # Create the docker run command
    cmd = create_docker_cmd(args)

    # Create and start the carbon footprint tracker
    tracker = EmissionsTracker(
        project_name=f"MIALSRTK{__version__}-docker",
        output_dir=args.codecarbon_output_dir if args.codecarbon_output_dir is not None else str(Path.home()),
        measure_power_secs=15,
    )
    tracker.start()

    # Execute the docker run command
    try:
        print(f'... cmd: {cmd}')
        run(cmd)
        exit_code = 0
    except Exception as e:
        print('Failed')
        print(e)
        exit_code = 1

    emissions: float = tracker.stop()
    print("########### CARBON FOOTPRINT OF RUN ###########")
    print(f"Total emissions: {emissions} kg")
    print("###############################################")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
