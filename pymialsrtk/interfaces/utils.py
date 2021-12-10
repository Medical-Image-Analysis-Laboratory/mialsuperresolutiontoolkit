# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
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
    command : string
        String containing the command to be executed (required)

    env : os.environ
        Specify a custom os.environ

    cwd : Directory
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
    p_files : list
        List of image paths to be sorted in ascending order

    Examples
    --------
    >>> in_files = ['sub-01_run-2_T2w.nii.gz', 'sub-01_run-5_T2w.nii.gz', 'sub-01_run-3_T2w.nii.gz', 'sub-01_run-1_T2w.nii.gz']
    >>> sort_ascending(in_files)

    """
    from operator import itemgetter
    import os
    path_basename = []
    for f in p_files:
        path_basename.append((os.path.basename(f), f))

    path_basename = sorted(path_basename, key=itemgetter(0))
    # p_files.sort()
    return [f[1] for f in path_basename]


def reorder_by_run_ids(p_files, p_order):
    """Function used to reorder images by their run-id IDS tag.

    The images are sorted according to the input parameters.
    If more images are available, they remaining are sorted in scending order.

    Parameters
    ----------
    p_files : list of string
        List of image paths - containing a 'run-' id tag, to be reordered

    p_order : list of int
        List of expecting run id order.

    Examples
    --------
    >>> in_files = ['sub-01_run-2_T2w.nii.gz', 'sub-01_run-5_T2w.nii.gz', 'sub-01_run-3_T2w.nii.gz', 'sub-01_run-1_T2w.nii.gz']
    >>> my_order = [1,5,3]
    >>> reorder_by_run_ids(in_files, my_order)

    """
    orig_order = [[int(f.split('_run-')[1].split('_')[0]), f] for f in p_files]

    id_and_files_ordered = []
    for s in p_order:
        id_and_files_ordered.append([[s, f[1]] for f in orig_order if ('run-' + str(s)) in f[1]][0])

    # # Todo: this if statement could be ignored to remove extra series.
    # if len(p_files) > len(p_order):
    #     remainings = [s for s in orig_order if s[0] not in p_order]
    #     remainings.sort()
    #     id_and_files_ordered = id_and_files_ordered + remainings

    return [i[1] for i in id_and_files_ordered]


def get_emission_car_miles_equivalent(emissions):
    """Return the equivalent of CO2 emissions [Kg] in terms of kms traveled by an average car.

    References
    ----------
    https://github.com/mlco2/codecarbon/blob/c6aebb9681186a71573748e381b6a3c9731de2d3/codecarbon/viz/data.py#L53

    """
    return "{:.0f}".format((emissions / 0.409) * 1.60934)


def get_emission_tv_time_equivalent(emissions):
    """Return the equivalent of CO2 emissions [Kg] in terms of kms traveled by an average car.

    References
    ----------
    https://github.com/mlco2/codecarbon/blob/c6aebb9681186a71573748e381b6a3c9731de2d3/codecarbon/viz/data.py#L66

    """
    tv_time_in_minutes = emissions * (1 / 0.097) * 60
    tv_time = "{:.0f} minutes".format(tv_time_in_minutes)
    if tv_time_in_minutes >= 60:
        time_in_hours = tv_time_in_minutes / 60
        tv_time = "{:.0f} hours".format(time_in_hours)
        if time_in_hours >= 24:
            time_in_days = time_in_hours / 24
            tv_time = "{:.0f} days".format(time_in_days)
    return tv_time
