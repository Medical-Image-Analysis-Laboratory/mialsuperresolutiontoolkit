# Copyright © 2016-2020 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK utils functions"""

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
    >>> run(self, cmd)

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
