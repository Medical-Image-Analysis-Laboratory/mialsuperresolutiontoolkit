# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the output of super-resolution reconstruction pipeline."""

import os
import traceback
from glob import glob
import pathlib

from traits.api import *

from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.utils as utils

from nipype import config
from nipype import logging as nipype_logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_srr_output_stage(name="srr_output_stage"):
    """Create a output management workflow
    for srr pipeline
    Parameters
    ----------
    ::
        name : name of workflow (default: preproc_stage)
    Inputs::

    Outputs::

    Example
    -------
    >>>
    """


    srr_output_stage = pe.Workflow(name=name)


    return srr_output_stage
