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


def create_input_stage(bids_dir,
                       subject,
                       session,
                       use_manual_masks,
                       m_masks_desc,
                       m_masks_derivatives_dir,
                       m_skip_stacks_ordering,
                       m_stacks,
                       p_do_multi_parameters,
                       name="input_stage"
                       ):
    """Create a input management workflow
    for srr pipeline
    Parameters
    ----------
    ::
        name : name of workflow (default: input_stage)
    Inputs::

    Outputs::

    Example
    -------
    >>>
    """

    input_stage = pe.Workflow(name=name)

    sub_ses = subject
    if session is not None:
        sub_ses = ''.join([sub_ses, '_', session])

    output_fields = [
        't2ws_filtered',
        'masks_filtered',
        'stacks_order',
        'report_image',
        'motion_tsv'
    ]
    if p_do_multi_parameters: output_fields += ['ground_truth']

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name='outputnode')

    if use_manual_masks:
        dg = pe.Node(
            interface=DataGrabber(outfields=['T2ws', 'masks']),
            name='data_grabber'
        )
        dg.inputs.base_directory = bids_dir
        dg.inputs.template = '*'
        dg.inputs.raise_on_empty = False
        dg.inputs.sort_filelist = True

        if session is not None:
            t2ws_template = os.path.join(
                subject, session, 'anat',
                '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])
            )
            if m_masks_desc is not None:
                masks_template = os.path.join(
                    'derivatives', m_masks_derivatives_dir, subject, session,
                    'anat', '_'.join([sub_ses, '*_run-*',
                                      '_desc-'+m_masks_desc, '*mask.nii.gz'])
                )
            else:
                masks_template = os.path.join(
                    'derivatives', m_masks_derivatives_dir, subject, session,
                    'anat', '_'.join([sub_ses, '*run-*', '*mask.nii.gz'])
                )
        else:
            t2ws_template = os.path.join(subject, 'anat',
                                         sub_ses + '*_run-*_T2w.nii.gz')

            if m_masks_desc is not None:
                masks_template = os.path.join(
                    'derivatives', m_masks_derivatives_dir, subject, session,
                    'anat', '_'.join([sub_ses, '*_run-*',
                                      '_desc-'+m_masks_desc, '*mask.nii.gz'])
                )
            else:
                masks_template = os.path.join(
                    'derivatives', m_masks_derivatives_dir, subject, 'anat',
                    sub_ses + '*_run-*_*mask.nii.gz'
                )

        dg.inputs.field_template = dict(T2ws=t2ws_template,
                                        masks=masks_template)

        brainMask = pe.MapNode(
            interface=IdentityInterface(fields=['out_file']),
            name='brain_masks_bypass',
            iterfield=['out_file'])

        if m_stacks is not None:
            custom_masks_filter = pe.Node(
                interface=preprocess.FilteringByRunid(),
                name='custom_masks_filter')

            custom_masks_filter.inputs.stacks_id = m_stacks

    else:
        dg = pe.Node(interface=DataGrabber(outfields=['T2ws']),
                     name='data_grabber')

        dg.inputs.base_directory = bids_dir
        dg.inputs.template = '*'
        dg.inputs.raise_on_empty = False
        dg.inputs.sort_filelist = True

        dg.inputs.field_template = dict(
            T2ws=os.path.join(subject, 'anat',
                              sub_ses+'*_run-*_T2w.nii.gz'))
        if session is not None:
            dg.inputs.field_template = dict(
                T2ws=os.path.join(subject, session, 'anat',
                                  '_'.join([sub_ses, '*run-*',
                                            '*T2w.nii.gz'])))

        if m_stacks is not None:
            t2ws_filter_prior_masks = pe.Node(
                interface=preprocess.FilteringByRunid(),
                name='t2ws_filter_prior_masks')
            t2ws_filter_prior_masks.inputs.stacks_id = m_stacks

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

    if not m_skip_stacks_ordering:
        stacksOrdering = pe.Node(interface=preprocess.StacksOrdering(),
                                 name='stackOrdering')
    else:
        stacksOrdering = pe.Node(
            interface=IdentityInterface(fields=['stacks_order']),
            name='stackOrdering')
        stacksOrdering.inputs.stacks_order = m_stacks

    if p_do_multi_parameters:

        gt = pe.Node(
            interface=DataGrabber(outfields=['gt']),
            name='gt_grabber')

        gt.inputs.base_directory = bids_dir
        gt.inputs.template = '*'
        gt.inputs.raise_on_empty = False
        gt.inputs.sort_filelist = True

        gt_template = os.path.join(
            subject,
            'anat',
            sub_ses + '_desc-iso_T2w.nii.gz'
        )
        if session is not None:
            gt_template = os.path.join(
                subject,
                session,
                'anat',
                sub_ses + '_desc-iso_T2w.nii.gz'
            )
        gt.inputs.field_template = dict(gt=gt_template)



    if use_manual_masks:
        if m_stacks is not None:
            input_stage.connect(dg, "masks", custom_masks_filter,
                                "input_files")
            input_stage.connect(custom_masks_filter, "output_files",
                                brainMask, "out_file")
        else:
            input_stage.connect(dg, "masks", brainMask, "out_file")
    else:
        if m_stacks is not None:
            input_stage.connect(dg, "T2ws", t2ws_filter_prior_masks,
                                "input_files")
            input_stage.connect(t2ws_filter_prior_masks, "output_files",
                                brainMask, "in_file")
        else:
            input_stage.connect(dg, "T2ws", brainMask, "in_file")

    if not m_skip_stacks_ordering:
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
    input_stage.connect(stacksOrdering, "report_image",
                        outputnode, "report_image")
    input_stage.connect(stacksOrdering, "motion_tsv",
                        outputnode, "motion_tsv")

    input_stage.connect(gt, "gt",
                        outputnode, "ground_truth")

    return input_stage
