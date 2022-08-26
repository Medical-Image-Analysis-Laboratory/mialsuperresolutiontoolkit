# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the input of super-resolution
reconstruction pipeline.
"""

import os
import pkg_resources

from traits.api import *
from nipype.interfaces.base import (TraitedSpec, File, InputMultiPath,
                                    OutputMultiPath, BaseInterface,
                                    BaseInterfaceInputSpec)
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface
from pymialsrtk.interfaces import preprocess


def create_input_stage(p_bids_dir,
                       p_subject,
                       p_session,
                       p_use_manual_masks,
                       p_masks_desc,
                       p_masks_derivatives_dir,
                       p_skip_stacks_ordering,
                       p_stacks,
                       p_do_srr_assessment,
                       name="input_stage"
                       ):
    """Create a input management workflow
    for srr pipeline
    Parameters
    ----------
    ::
        name : name of workflow (default: input_stage)
        p_bids_dir
        p_subject
        p_session
        p_use_manual_masks
        p_masks_desc
        p_masks_derivatives_dir
        p_skip_stacks_ordering
        p_stacks
        p_do_srr_assessment

    Inputs::

    Outputs::
        outputnode.t2ws_filtered
        outputnode.masks_filtered
        outputnode.stacks_order
        outputnode.report_image
        outputnode.motion_tsv
        outputnode.ground_truth (optional, if p_do_srr_assessment=True)
    Example
    -------
    >>>
    """

    input_stage = pe.Workflow(name=name)

    sub_ses = p_subject
    sub_path = p_subject
    if p_session is not None:
        sub_ses = ''.join([sub_ses, '_', p_session])
        sub_path = os.path.join(p_subject, p_session)

    output_fields = ['t2ws_filtered', 'masks_filtered', 'stacks_order']

    if not p_skip_stacks_ordering:
        output_fields += ['report_image', 'motion_tsv']

    if p_do_srr_assessment:
        output_fields += ['ground_truth']

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=output_fields
        ),
        name='outputnode'
    )
    if p_use_manual_masks:
        dg = pe.Node(
            interface=DataGrabber(outfields=['T2ws', 'masks']),
            name='data_grabber'
        )
        dg.inputs.base_directory = p_bids_dir
        dg.inputs.template = '*'
        dg.inputs.raise_on_empty = False
        dg.inputs.sort_filelist = True

        t2ws_template = os.path.join(sub_path, 'anat',
                                     sub_ses + '*_run-*_T2w.nii.gz'
                                     )
        if p_masks_desc is not None:
            masks_template = os.path.join(
                'derivatives', p_masks_derivatives_dir,
                sub_path, 'anat',
                '_'.join([sub_ses, '*_run-*', '_desc-'+p_masks_desc,
                          '*mask.nii.gz'])
            )
        else:
            masks_template = os.path.join(
                'derivatives', p_masks_derivatives_dir,
                sub_path, 'anat',
                '_'.join([sub_ses, '*run-*', '*mask.nii.gz'])
            )
        dg.inputs.field_template = dict(T2ws=t2ws_template,
                                        masks=masks_template)

        brainMask = pe.MapNode(
            interface=IdentityInterface(fields=['out_file']),
            name='brain_masks_bypass',
            iterfield=['out_file'])

        if p_stacks is not None:
            custom_masks_filter = pe.Node(
                interface=preprocess.FilteringByRunid(),
                name='custom_masks_filter')

            custom_masks_filter.inputs.stacks_id = p_stacks

    else:
        dg = pe.Node(interface=DataGrabber(outfields=['T2ws']),
                     name='data_grabber')

        dg.inputs.base_directory = p_bids_dir
        dg.inputs.template = '*'
        dg.inputs.raise_on_empty = False
        dg.inputs.sort_filelist = True

        dg.inputs.field_template = dict(
            T2ws=os.path.join(sub_path, 'anat',
                              sub_ses+'*_run-*_T2w.nii.gz'))

        if p_stacks is not None:
            t2ws_filter_prior_masks = pe.Node(
                interface=preprocess.FilteringByRunid(),
                name='t2ws_filter_prior_masks')
            t2ws_filter_prior_masks.inputs.stacks_id = p_stacks

        brainMask = pe.MapNode(interface=preprocess.BrainExtraction(),
                               name='brainExtraction',
                               iterfield=['in_file'])

        brainMask.inputs.in_ckpt_loc = pkg_resources.resource_filename(
            "pymialsrtk",
            os.path.join("data",
                         "Network_checkpoints",
                         "Network_checkpoints_localization",
                         "Unet.ckpt-88000.index")
        ).split('.index')[0]
        brainMask.inputs.threshold_loc = 0.49
        brainMask.inputs.in_ckpt_seg = pkg_resources.resource_filename(
            "pymialsrtk",
            os.path.join("data",
                         "Network_checkpoints",
                         "Network_checkpoints_segmentation",
                         "Unet.ckpt-20000.index")
        ).split('.index')[0]
        brainMask.inputs.threshold_seg = 0.5

    t2ws_filtered = pe.Node(interface=preprocess.FilteringByRunid(),
                            name='t2ws_filtered')
    masks_filtered = pe.Node(interface=preprocess.FilteringByRunid(),
                             name='masks_filtered')

    if not p_skip_stacks_ordering:
        stacksOrdering = pe.Node(interface=preprocess.StacksOrdering(),
                                 name='stackOrdering')
    else:
        stacksOrdering = pe.Node(
            interface=IdentityInterface(fields=['stacks_order']),
            name='stackOrdering')
        stacksOrdering.inputs.stacks_order = p_stacks

    if p_do_srr_assessment:

        gt = pe.Node(
            interface=DataGrabber(outfields=['gt']),
            name='gt_grabber')

        gt.inputs.base_directory = p_bids_dir
        gt.inputs.template = '*'
        gt.inputs.raise_on_empty = False
        gt.inputs.sort_filelist = True

        gt_template = os.path.join(
            p_subject,
            'anat',
            sub_ses + '_desc-iso_T2w.nii.gz'
        )
        if p_session is not None:
            gt_template = os.path.join(
                p_subject,
                p_session,
                'anat',
                sub_ses + '_desc-iso_T2w.nii.gz'
            )
        gt.inputs.field_template = dict(gt=gt_template)


    if p_use_manual_masks:
        if p_stacks is not None:
            input_stage.connect(dg, "masks", custom_masks_filter,
                                "input_files")
            input_stage.connect(custom_masks_filter, "output_files",
                                brainMask, "out_file")
        else:
            input_stage.connect(dg, "masks", brainMask, "out_file")
    else:
        if p_stacks is not None:
            input_stage.connect(dg, "T2ws", t2ws_filter_prior_masks,
                                "input_files")
            input_stage.connect(t2ws_filter_prior_masks, "output_files",
                                brainMask, "in_file")
        else:
            input_stage.connect(dg, "T2ws", brainMask, "in_file")

    if not p_skip_stacks_ordering:
        input_stage.connect(brainMask, "out_file",
                            stacksOrdering, "input_masks")

    input_stage.connect(stacksOrdering, "stacks_order",
                        t2ws_filtered, "stacks_id")
    input_stage.connect(dg, "T2ws",
                        t2ws_filtered, "input_files")
    input_stage.connect(stacksOrdering, "stacks_order",
                        masks_filtered, "stacks_id")
    input_stage.connect(brainMask, "out_file",
                        masks_filtered, "input_files")

    input_stage.connect(masks_filtered, "output_files",
                        outputnode, "masks_filtered")
    input_stage.connect(t2ws_filtered, "output_files",
                        outputnode, "t2ws_filtered")
    input_stage.connect(stacksOrdering, "stacks_order",
                        outputnode, "stacks_order")

    if not p_skip_stacks_ordering:
        input_stage.connect(stacksOrdering, "report_image",
                            outputnode, "report_image")
        input_stage.connect(stacksOrdering, "motion_tsv",
                            outputnode, "motion_tsv")

    if p_do_srr_assessment:
        input_stage.connect(gt, "gt",
                            outputnode, "ground_truth")

    return input_stage
