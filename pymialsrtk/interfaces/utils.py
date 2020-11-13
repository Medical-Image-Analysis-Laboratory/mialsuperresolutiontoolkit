# Copyright © 2016-2020 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK utils functions."""

import os
import subprocess


def run(command, env=None, cwd=None):
    """Function calls by each MIALSRTK interface.

    It runs the command specified as input via ``subprocess.run()``.

    Parameters
    ----------
    command <string>
        String containing the command to be executed (required)

    env <os.environ>
        Specify a custom os.environ

    cwd <Directory>
        Specify a custom current working directory

    Examples
    --------
    >>> cmd = 'btkNLMDenoising -i "/path/to/in_file" -o "/path/to/out_file" -b 0.1'
    >>> run(cmd)

    """

    merged_env = os.environ

    if cwd is None:
        cwd = os.getcwd()

    if env is not None:
        merged_env.update(env)

    # Python 3.7
    # process = subprocess.run(command, shell=True,
    # 	env=merged_env, cwd=cwd, capture_output=True)

    # Python 3.6 (No capture_output)
    process = subprocess.run(command,
                             shell=True,
                             env=merged_env,
                             cwd=cwd)
    return process


def sort_ascending(p_files):
    """Function used to sort images at the input of a nipype node.

    Parameters
    ----------
    p_files <list<string>>
        List of image paths to be sorting in ascending order

    Examples
    --------
    >>> in_files = ['sub-01_run-2_T2w.nii.gz', 'sub-01_run-5_T2w.nii.gz', 'sub-01_run-3_T2w.nii.gz', 'sub-01_run-1_T2w.nii.gz']
    >>> sort_ascending(in_files)

    """
    p_files.sort()
    return p_files