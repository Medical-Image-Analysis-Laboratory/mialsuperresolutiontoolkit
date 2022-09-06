# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
# This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution
reconstruction pipeline."""

from traits.api import *

from nipype.pipeline import engine as pe

from nipype.interfaces import utility as util
from nipype.interfaces.ants \
    import RegistrationSynQuick, ApplyTransforms

import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.preprocess as preprocess

import pymialsrtk.workflows.postproc_stage as postproc_stage
import nipype
import logging

def create_srr_assessment_stage(
        p_do_multi_parameters=False,
        p_do_reconstruct_labels=False,
        p_input_srtv_node=None,
        p_verbose=False,
        p_openmp_number_of_cores=1,
        name='srr_assessment_stage'
):
    """Create an assessment workflow to compare
    a SR-reconstructed image and a reference target.

    Parameters
    ----------
    ::
        name : name of workflow (default: sr_assessment_stage)
        p_do_multi_parameters : boolean
            weither multiple SR are to be assessed
            with different TV parameters(default: False)
        p_input_srtv_node : string
            when p_do_multi_parameters is set, name of the sourcenode
            from which metrics must be merged
        p_openmp_number_of_cores : integer
            number of threads possible
            for ants registration (default : 1)

    Inputs::
        input_reference_image
        input_reference_mask
        input_reference_labelmap
        input_sr_image
        input_sdi_image
        input_TV_parameters
    Outputs::
        outputnode.output_metrics
    Example
    -------
    """

    srr_assessment_stage = pe.Workflow(name=name)

    if not p_verbose:
        # Removing verbose by removing the output altogether as we
        # cannot control the verbosity level in the ANTs call
        logging.getLogger(f'nipype.workflow.{name}').setLevel(0)

    # Set up a node to define all inputs required for the
    # preprocessing workflow
    input_fields = [
        'input_ref_image',
        'input_ref_mask',
        'input_ref_labelmap',
        'input_sr_image',
        'input_sdi_image',
        'input_TV_parameters'
    ]

    if p_do_reconstruct_labels:
        input_fields += ['input_sr_labelmap']

    output_fields = ['output_metrics', 'output_metrics_labels']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=output_fields
        ),
        name='outputnode')

    proc_reference = postproc_stage.create_postproc_stage(
        p_ga=None,
        p_do_anat_orientation=False,
        p_do_reconstruct_labels=False,
        p_verbose=p_verbose,
        name='proc_reference'
    )

    crop_reference = pe.Node(
        interface=preprocess.ReduceFieldOfView(),
        name='crop_reference'
    )

    registration_quick = pe.Node(
        interface=RegistrationSynQuick(),
        name='registration_quick'
    )
    registration_quick.inputs.num_threads = p_openmp_number_of_cores
    registration_quick.inputs.transform_type = 'r'
    registration_quick.environ = {'PATH': '/opt/conda/bin'}
    registration_quick.terminal_output = 'file_stderr'
    
    apply_transform = pe.Node(
        interface=ApplyTransforms(),
        name='apply_transform'
    )
    apply_transform.inputs.num_threads = p_openmp_number_of_cores
    apply_transform.environ = {'PATH': '/opt/conda/bin'}
    apply_transform.terminal_output = 'file_stderr'

    if p_do_reconstruct_labels:
        apply_transform_labels = pe.Node(
            interface=ApplyTransforms(),
            name='apply_transform_labels'
        )
        apply_transform_labels.inputs.num_threads = p_openmp_number_of_cores
        apply_transform_labels.environ = {'PATH': '/opt/conda/bin'}
        apply_transform_labels.terminal_output = 'file_stderr'
        apply_transform_labels.interpolation = 'NearestNeighbor'

    mask_sr = pe.Node(
        interface=preprocess.MialsrtkMaskImage(),
        name='mask_sr'
    )

    sr_image_metrics = pe.Node(
        postprocess.ImageMetrics(),
        name='sr_image_metrics'
    )

    if p_do_multi_parameters:
        concat_sr_image_metrics = pe.JoinNode(
            interface=postprocess.ConcatenateImageMetrics(),
            joinfield=['input_metrics', 'input_metrics_labels'],
            joinsource=p_input_srtv_node,
            name='concat_sr_image_metrics'
        )

    srr_assessment_stage.connect(inputnode, 'input_ref_image',
                                proc_reference, 'inputnode.input_image')
    srr_assessment_stage.connect(inputnode, 'input_ref_mask',
                                proc_reference, 'inputnode.input_mask')

    srr_assessment_stage.connect(proc_reference, 'outputnode.output_image',
                                crop_reference, 'input_image')
    srr_assessment_stage.connect(proc_reference, 'outputnode.output_mask',
                                crop_reference, 'input_mask')
    srr_assessment_stage.connect(inputnode, 'input_ref_labelmap',
                                crop_reference, "input_label")

    srr_assessment_stage.connect(inputnode, 'input_sdi_image',
                                registration_quick, 'moving_image')
    srr_assessment_stage.connect(crop_reference, 'output_image',
                                registration_quick, 'fixed_image')

    srr_assessment_stage.connect(inputnode, 'input_sr_image',
                                apply_transform, 'input_image')
    srr_assessment_stage.connect(crop_reference, 'output_image',
                                apply_transform, 'reference_image')
    srr_assessment_stage.connect(registration_quick, 'out_matrix',
                                apply_transform, 'transforms')

    if p_do_reconstruct_labels:
        srr_assessment_stage.connect(inputnode, 'input_sr_labelmap',
                                    apply_transform_labels, 'input_image')
        srr_assessment_stage.connect(crop_reference, 'output_image',
                                    apply_transform_labels, 'reference_image')
        srr_assessment_stage.connect(registration_quick, 'out_matrix',
                                    apply_transform_labels, 'transforms')

    srr_assessment_stage.connect(apply_transform, 'output_image',
                                mask_sr, 'in_file')
    srr_assessment_stage.connect(crop_reference, 'output_mask',
                                mask_sr, 'in_mask')

    srr_assessment_stage.connect(mask_sr, 'out_im_file',
                                sr_image_metrics, 'input_image')
    srr_assessment_stage.connect(crop_reference, 'output_image',
                                sr_image_metrics, 'input_ref_image')
    srr_assessment_stage.connect(crop_reference, 'output_label',
                                sr_image_metrics, 'input_ref_labelmap')
    srr_assessment_stage.connect(inputnode, 'input_TV_parameters',
                                sr_image_metrics, 'input_TV_parameters')

    if p_do_multi_parameters:
        srr_assessment_stage.connect(sr_image_metrics, 'output_metrics',
                                    concat_sr_image_metrics, 'input_metrics')
        srr_assessment_stage.connect(sr_image_metrics, 'output_metrics_labels',
                                    concat_sr_image_metrics,
                                    'input_metrics_labels')

        srr_assessment_stage.connect(concat_sr_image_metrics, 'output_csv',
                                    outputnode, 'output_metrics')
        srr_assessment_stage.connect(concat_sr_image_metrics,
                                    'output_csv_labels',
                                    outputnode, 'output_metrics_labels')

    else:
        srr_assessment_stage.connect(sr_image_metrics, 'output_metrics',
                                    outputnode, 'output_metrics')

        srr_assessment_stage.connect(sr_image_metrics, 'output_metrics_labels',
                                    outputnode, 'output_metrics_labels')

    return srr_assessment_stage
