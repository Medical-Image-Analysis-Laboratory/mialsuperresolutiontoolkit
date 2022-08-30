# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
# This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution
reconstruction pipeline."""

from traits.api import *

from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.postprocess as postprocess


def create_sr_assessment_stage(
        p_multi_parameters=False,
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
        p_multi_parameters : boolean
            weither multiple SR are to be assessed
            with different TV parameters(default: False)
        p_input_srtv_node : string
            when p_do_multi_parameters is set, name of the sourcenode
            from which metrics must be merged
        p_openmp_number_of_cores : integer
            number of threads possible
            for ants registration (default : 1)

    Inputs::
        inputnode.input_reference_image
        inputnode.input_reference_mask
        inputnode.input_reference_labelmap
        inputnode.input_image
        inputnode.input_TV_parameters
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
        'input_image',
        'input_TV_parameters'
    ]

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['output_metrics']
        ),
        name='outputnode')

    quality_metrics = pe.Node(
        postprocess.QualityMetrics(),
        name='quality_metrics'
    )
    quality_metrics.inputs.in_num_threads = p_openmp_number_of_cores

    z_debug = pe.Node(
        interface=util.IdentityInterface(
            fields=["output_warped_image"]
        ),
        name='z_debug'
    )

    if p_multi_parameters:
        concat_quality_metrics = pe.JoinNode(
            interface=postprocess.ConcatenateQualityMetrics(),
            joinfield='input_metrics',
            joinsource=p_input_srtv_node,
            name='concat_quality_metrics'
        )

    sr_assessment_stage.connect(inputnode, 'input_image',
                                quality_metrics, 'input_image')
    sr_assessment_stage.connect(inputnode, 'input_ref_image',
                                quality_metrics, 'input_ref_image')
    sr_assessment_stage.connect(inputnode, 'input_ref_mask',
                                quality_metrics, 'input_ref_mask')
    sr_assessment_stage.connect(inputnode, 'input_ref_labelmap',
                                quality_metrics, 'input_ref_labelmap')

    sr_assessment_stage.connect(inputnode, 'input_TV_parameters',
                                quality_metrics, 'input_TV_parameters')

    sr_assessment_stage.connect(quality_metrics, 'output_warped_image',
                                z_debug, 'output_warped_image')

    if p_multi_parameters:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                                    concat_quality_metrics, 'input_metrics')

        sr_assessment_stage.connect(concat_quality_metrics, 'output_csv',
                                    outputnode, 'output_metrics')
    else:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                                    outputnode, 'output_metrics')

    return sr_assessment_stage
