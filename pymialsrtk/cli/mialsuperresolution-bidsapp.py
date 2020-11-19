#!/usr/bin/env python
#
# Copyright © 2016-2020
# Medical Image Analysis Laboratory,
# University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland,
# and Contributors
#
#  This software is distributed under the open-source license Modified BSD.

"""This module defines the mialsuperresolutiontoolkit-bidsapp script that wraps calls to the BIDS APP."""

# General imports
import os
import sys

# Own imports
from pymialsrtk.info import __version__
from pymialsrtk.parser import get_parser
from pymialsrtk.interfaces.utils import run


def create_docker_cmd(args):
    """Function that creates and returns the BIDS App docker run command.

    Parameters
    ----------
    args : dict
        Dictionary of parsed input argument in the form:
        {
            'bids_dir': "/path/to/bids/dataset/directory",
            'output_dir': "/path/to/output/directory",
            'analysis_level': "participant",
            'participant_label': ['01', '02', '03'],
            'openmp_nb_of_cores': 1,
            'nipype_nb_of_cores': 1
        }

    Returns
    -------
    cmd : string
        String containing the command to be run via `subprocess.run()`
    """
    cmd = 'docker run -t --rm '
    cmd += f'-u {os.geteuid()}:{os.getegid()} '
    cmd += f'-v :/bids_dir '
    cmd += f'-v :/output_dir '
    cmd += f'sebastientourbier/mialsuperresolutiontoolkit-bidsapp:v{__version__} '
    cmd += f'{args.bids_dir} '
    cmd += f'{args.output_dir} '
    cmd += f'{args.analysis_level} '
    cmd += f'--participant_label '
    for label in args.participant_label:
        cmd += '{label} '
    cmd += f'--param_file {args.param_file} '
    cmd += f'--openmp_nb_of_cores {args.openmp_nb_of_cores} '
    cmd += f'--nipype_nb_of_cores {args.nipype_nb_of_cores}'

    return cmd


def main(args):
    """Main function that creates and executes the BIDS App (Docker or Singularity) command.

    Parameters
    ----------
    args : dict
        Dictionary of parsed input argument in the form {'key': 'value'}

    Returns
    -------
    exit_code : 0 or 1
        An exit code given to sys.exit() with:
        - '0' that indicates completion without error
        - '1' that indicates there was an error
    """

    cmd = create_docker_cmd(args)

    try:
        print(f'... cmd: {cmd}')
        run(cmd)
        exit_code = 0
    except Exception as e:
        print('Failed')
        print(e)
        exit_code = 1

    return exit_code


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    sys.exit(main(args))
