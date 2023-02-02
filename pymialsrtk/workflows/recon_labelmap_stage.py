# Copyright © 2016-2023 Medical Image Analysis Laboratory,
# University Hospital Center and University of Lausanne (UNIL-CHUV),
# Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the high-resolution reconstruction stage of low-resolution labelmaps in the super-resolution reconstruction pipeline."""

from traits.api import *

import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.reconstruction as reconstruction

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_recon_labelmap_stage(
    p_sub_ses, p_verbose=False, name="recon_labels_stage"
):
    """Create a SR reconstruction workflow for tissue label maps.

    Parameters
    ----------
    p_sub_ses : string
        String containing subject-session information for output formatting
    p_verbose : boolean
        Whether verbosity should be enabled
        (default: `False`)
    name : string
        Name of workflow
        (default: "recon_labels_stage")

    Inputs
    ------
    input_labels : list of items which are a pathlike object or string representing a file
        Input LR label maps
    input_masks : list of items which are a pathlike object or string representing a file
        Input mask images
    input_transforms : list of items which are a pathlike object or string representing a file
        Input tranforms
    input_reference : pathlike object or string representing a file
        Input HR reference image
    label_ids : list of integer
        Label IDs to reconstruct
    stacks_order : list of integer
        Order of stacks in the reconstruction

    Outputs
    -------
    output_labelmap : pathlike object or string representing a file
        HR labelmap

    Example
    -------
    >>> from pymialsrtk.pipelines.workflows import recon_labelmap_stage as rec_label
    >>> recon_labels_stage = rec_label.create_recon_labelmap_stage(
            p_sub_ses=p_sub_ses,
            p_verbose=p_verbose
        )
    >>> recon_labels_stage.inputs.input_labels = [
            'sub-01_run-1_labels.nii.gz',
            'sub-01_run-2_labels.nii.gz'
        ]
    >>> recon_labels_stage.inputs.input_masks = [
            'sub-01_run-1_T2w.nii_mask.gz',
            'sub-01_run-2_T2w.nii_mask.gz'
        ]
    >>> recon_labels_stage.inputs.input_transforms = [
            'sub-01_run-1_transforms.txt',
            'sub-01_run-2_transforms.txt'
        ]
    >>> recon_labels_stage.inputs.input_reference = 'sub-01_desc-GT_T2w.nii.gz'
    >>> recon_labels_stage.inputs.label_ids = 'sub-01_desc-GT_T2w.nii.gz'
    >>> recon_labels_stage.inputs.stacks_order = [2,1]
    >>> recon_labels_stage.run()  # doctest: +SKIP

    """

    recon_labels_stage = pe.Workflow(name=name)

    recon_labels_stage.config["remove_unnecessary_outputs"] = False

    # Set up input/output nodes to define the workflow requirements.

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=[
                "input_labels",
                "input_masks",
                "input_transforms",
                "input_reference",
                "label_ids",
                "stacks_order",
            ]
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["output_labelmap"]),
        name="outputnode",
    )

    labels_split_lr_labelmap = pe.MapNode(
        interface=preprocess.SplitLabelMaps(),
        iterfield=["in_labelmap"],
        name="labels_split_lr_labelmap",
    )

    labels_merge_ids = pe.Node(
        interface=preprocess.ListsMerger(),
        joinsource="labelmap_splitter",
        joinfield="inputs",
        name="labels_merge_ids",
    )

    labels_merge_lr_maps = pe.Node(
        interface=preprocess.ListsMerger(),
        joinsource="labelmap_splitter",
        joinfield="inputs",
        name="labels_merge_lr_maps",
    )

    labels_reconstruct_hr_maps = pe.MapNode(
        interface=reconstruction.MialsrtkSDIComputation(
            sub_ses=p_sub_ses,
            verbose=p_verbose,
        ),
        iterfield=["label_id"],
        name="labels_reconstruct_hr_maps",
    )

    labels_merge_hr_maps = pe.Node(
        interface=preprocess.ListsMerger(),
        joinsource="labels_reconstruct_hr_maps",
        joinfield="inputs",
        name="labels_merge_hr_maps",
    )

    labels_majorityvoting = pe.Node(
        interface=postprocess.MergeMajorityVote(), name="labels_majorityvoting"
    )

    recon_labels_stage.connect(
        inputnode, "input_labels", labels_split_lr_labelmap, "in_labelmap"
    )

    recon_labels_stage.connect(
        labels_split_lr_labelmap, "out_labels", labels_merge_ids, "inputs"
    )

    recon_labels_stage.connect(
        labels_split_lr_labelmap,
        "out_labelmaps",
        labels_merge_lr_maps,
        "inputs",
    )

    recon_labels_stage.connect(
        labels_merge_ids, "outputs", labels_reconstruct_hr_maps, "label_id"
    )

    recon_labels_stage.connect(
        labels_merge_lr_maps,
        "outputs",
        labels_reconstruct_hr_maps,
        "input_images",
    )

    recon_labels_stage.connect(
        inputnode, "stacks_order", labels_reconstruct_hr_maps, "stacks_order"
    )
    recon_labels_stage.connect(
        inputnode,
        "input_reference",
        labels_reconstruct_hr_maps,
        "input_reference",
    )
    recon_labels_stage.connect(
        inputnode,
        "input_transforms",
        labels_reconstruct_hr_maps,
        "input_transforms",
    )
    recon_labels_stage.connect(
        inputnode, "input_masks", labels_reconstruct_hr_maps, "input_masks"
    )

    recon_labels_stage.connect(
        labels_reconstruct_hr_maps,
        "output_sdi",
        labels_merge_hr_maps,
        "inputs",
    )

    recon_labels_stage.connect(
        labels_merge_hr_maps, "outputs", labels_majorityvoting, "input_images"
    )

    recon_labels_stage.connect(
        labels_majorityvoting, "output_image", outputnode, "output_labelmap"
    )

    return recon_labels_stage
