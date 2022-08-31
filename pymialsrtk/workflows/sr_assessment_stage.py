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


def create_sr_assessment_stage(
        p_do_multi_parameters=False,
        p_do_reconstruct_labels=False,
        p_input_srtv_node=None,
        p_openmp_number_of_cores=1,
        name='sr_assessment_stage'
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

    sr_assessment_stage = pe.Workflow(name=name)

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

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['output_metrics']
        ),
        name='outputnode')

    mask_reference = pe.Node(
        interface=preprocess.MialsrtkMaskImage(),
        name='mask_reference'
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


    quality_metrics = pe.Node(
        postprocess.QualityMetrics(),
        name='quality_metrics'
    )

    if p_do_multi_parameters:
        concat_quality_metrics = pe.JoinNode(
            interface=postprocess.ConcatenateQualityMetrics(),
            joinfield='input_metrics',
            joinsource=p_input_srtv_node,
            name='concat_quality_metrics'
        )

    sr_assessment_stage.connect(inputnode, 'input_ref_image',
                                mask_reference, 'in_file')
    sr_assessment_stage.connect(inputnode, 'input_ref_mask',
                                mask_reference, 'in_mask')

    sr_assessment_stage.connect(mask_reference, 'out_im_file',
                                crop_reference, 'input_image')
    sr_assessment_stage.connect(inputnode, 'input_ref_mask',
                                crop_reference, 'input_mask')
    sr_assessment_stage.connect(inputnode, 'input_ref_labelmap',
                                crop_reference, "input_label")

    sr_assessment_stage.connect(inputnode, 'input_sdi_image',
                                registration_quick, 'moving_image')
    sr_assessment_stage.connect(crop_reference, 'output_image',
                                registration_quick, 'fixed_image')

    sr_assessment_stage.connect(inputnode, 'input_sr_image',
                                apply_transform, 'input_image')
    sr_assessment_stage.connect(crop_reference, 'output_image',
                                apply_transform, 'reference_image')
    sr_assessment_stage.connect(registration_quick, 'out_matrix',
                                apply_transform, 'transforms')

    if p_do_reconstruct_labels:
        sr_assessment_stage.connect(inputnode, 'input_sr_labelmap',
                                    apply_transform_labels, 'input_image')
        sr_assessment_stage.connect(crop_reference, 'output_image',
                                    apply_transform_labels, 'reference_image')
        sr_assessment_stage.connect(registration_quick, 'out_matrix',
                                    apply_transform_labels, 'transforms')

    sr_assessment_stage.connect(apply_transform, 'output_image',
                                mask_sr, 'in_file')
    sr_assessment_stage.connect(crop_reference, 'output_mask',
                                mask_sr, 'in_mask')

    sr_assessment_stage.connect(mask_sr, 'out_im_file',
                                quality_metrics, 'input_image')
    sr_assessment_stage.connect(crop_reference, 'output_image',
                                quality_metrics, 'input_ref_image')
    sr_assessment_stage.connect(crop_reference, 'output_label',
                                quality_metrics, 'input_ref_labelmap')
    sr_assessment_stage.connect(inputnode, 'input_TV_parameters',
                                quality_metrics, 'input_TV_parameters')

    if p_do_multi_parameters:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                                    concat_quality_metrics, 'input_metrics')

        sr_assessment_stage.connect(concat_quality_metrics, 'output_csv',
                                    outputnode, 'output_metrics')
    else:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                                    outputnode, 'output_metrics')

    return sr_assessment_stage
