# Copyright © 2016-2021 Medical Image Analysis Laboratory,
# University Hospital Center and University of Lausanne (UNIL-CHUV),
# Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the high-resolution reconstruction stage of the low-resolution labelmaps
in the super-resolution reconstruction pipeline."""

import os
import traceback
from glob import glob
import pathlib

from traits.api import *

from nipype.interfaces.base import traits, \
    TraitedSpec, File, \
    InputMultiPath, OutputMultiPath, \
    BaseInterface, BaseInterfaceInputSpec

from nipype.pipeline import Node, MapNode, Workflow

import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.utils as utils

from nipype import config
from nipype import logging as nipype_logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_recon_labelmap_stage(sub_ses, name="recon_labels_stage"):
    """Create a SR reconstruction workflow
    for tissue label maps.
    Parameters
    ----------
    ::
        name : name of workflow (default: recon_labels_stage)
    Inputs::
        inputnode.input_labels : Input LR label maps (list of filenames)
        inputnode.input_masks : Input mask images (list of filenames)
        inputnode.input_transforms : Input tranforms (list of filenames)
        inputnode.input_reference : Input HR reference image (filename)
        inputnode.label_ids : Label IDs to reconstruct (list of integer)
        inputnode.stacks_order : Order of stacks in
                                the reconstruction (list of integer)
    Outputs::
        outputnode.output_labelmap : HR labelmap (filename)
    Example
    -------
    >>>
    >>>
    >>>
    >>>
    >>>
    >>>
    >>>
    >>>
    """

    recon_labels_stage = pe.Workflow(name=name)

    """
    Set up input/output nodes to define the workflow requirements.
    """

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['input_labels', 'input_masks', 'input_transforms',
                    'input_reference',
                    'label_ids', 'stacks_order']),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['output_labelmap']
        ),
        name='outputnode')

    labelmap_splitter = MapNode(
        interface=preprocess.SplitLabelMaps(),
        iterfield=['in_labelmap'],
        name='labelmap_splitter'
    )

    labels_merge_lr_maps = Node(
        interface=preprocess.PathListsMerger(),
        joinsource="labelmap_splitter",
        joinfield="inputs",
        name='labels_merge_lr_maps'
    )

    labels_reconstruct_hr_maps = MapNode(
        interface=reconstruction.MialsrtkSDIComputation(),
        iterfield=['label_id'],
        name='labels_reconstruct_hr_maps'
    )
    labels_reconstruct_hr_maps.inputs.sub_ses = sub_ses

    labels_merge_hr_maps = Node(
        interface=preprocess.PathListsMerger(),
        joinsource="labels_reconstruct_hr_maps",
        joinfield="inputs",
        name='labels_merge_hr_maps'
    )

    labels_majorityvoting = Node(interface=postprocess.MergeMajorityVote(),
                                 name='labels_majorityvoting')

    recon_labels_stage.connect(inputnode, "label_ids",
                               labelmap_splitter, "all_labels")
    recon_labels_stage.connect(inputnode, "input_labels",
                               labelmap_splitter, "in_labelmap")

    recon_labels_stage.connect(labelmap_splitter, "out_labels",
                               labels_merge_lr_maps, "inputs")

    recon_labels_stage.connect(inputnode, "label_ids",
                               labels_reconstruct_hr_maps, "label_id")
    recon_labels_stage.connect(inputnode, "stacks_order",
                               labels_reconstruct_hr_maps, "stacks_order")
    recon_labels_stage.connect(inputnode, "input_reference",
                               labels_reconstruct_hr_maps, "input_reference")
    recon_labels_stage.connect(inputnode, "input_transforms",
                               labels_reconstruct_hr_maps, "input_transforms")
    recon_labels_stage.connect(inputnode, "input_masks",
                               labels_reconstruct_hr_maps, "input_masks")
    recon_labels_stage.connect(labels_merge_lr_maps, "outputs",
                               labels_reconstruct_hr_maps, "input_images")

    recon_labels_stage.connect(labels_reconstruct_hr_maps, "output_hr",
                               labels_merge_hr_maps, "inputs")

    recon_labels_stage.connect(labels_merge_hr_maps, "outputs",
                               labels_majorityvoting, "input_images")

    recon_labels_stage.connect(labels_majorityvoting, "output_image",
                               outputnode, 'output_labelmap')

    return recon_labels_stage
