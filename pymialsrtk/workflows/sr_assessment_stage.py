# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
# This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution
reconstruction pipeline."""

from traits.api import *
from nipype.interfaces.base import (TraitedSpec, File,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterface, BaseInterfaceInputSpec)
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils


def create_sr_assessment_stage(
        p_multi_parameters=False,
        p_input_srtv_node=None,
        name="sr_assessment_stage"
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

    Inputs::
        inputnode.input_ground_truth
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
        'input_ground_truth',
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
    quality_metrics.inputs.in_num_threads = 3  # TODO

    if p_multi_parameters:
        concatenate_quality_metrics = pe.JoinNode(
            interface=postprocess.ConcatenateQualityMetrics(),
            joinfield='input_metrics',
            joinsource=p_input_srtv_node,
            name='concatenate_quality_metrics'
        )

    sr_assessment_stage.connect(inputnode, 'input_image',
                        quality_metrics, 'input_image')
    sr_assessment_stage.connect(inputnode, "input_ground_truth",
                        quality_metrics, 'input_ground_truth')

    sr_assessment_stage.connect(inputnode, "input_TV_parameters",
                                quality_metrics, 'input_TV_parameters')

    if p_multi_parameters:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                            concatenate_quality_metrics, 'input_metrics')

        sr_assessment_stage.connect(concatenate_quality_metrics, 'output_csv',
                            outputnode, 'output_metrics')
    else:
        sr_assessment_stage.connect(quality_metrics, 'output_metrics',
                            outputnode, 'output_metrics')

    return sr_assessment_stage
