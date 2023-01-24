# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the input of super-resolution reconstruction pipeline."""

import os
import pkg_resources

from traits.api import *

from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.utility import IdentityInterface
from pymialsrtk.interfaces import preprocess


def create_input_stage(
    p_bids_dir,
    p_sub_ses,
    p_sub_path,
    p_use_manual_masks,
    p_masks_desc,
    p_masks_derivatives_dir,
    p_labels_derivatives_dir,
    p_skip_stacks_ordering,
    p_do_reconstruct_labels,
    p_stacks,
    p_do_srr_assessment,
    p_verbose,
    name="input_stage",
):
    """Create a input management workflow for srr pipeline.

    Parameters
    ----------
    name : :str:
        name of workflow (default: input_stage)
    p_bids_dir : :str:
            Path to the bids directory
    p_sub_ses : :str:
            String containing subject-session information.
    p_use_manual_masks : :bool:
        Whether manual masks are used
    p_masks_desc : :str:
        BIDS description tag of masks to use (optional)
    p_masks_derivatives_dir : :str:
        Path to the directory of the manual masks.
    p_skip_stacks_ordering : :bool:
        Whether stacks ordering should be skipped. If true, uses the order
        provided in `p_stacks`.
    p_stacks : :list: of :int:
        List of stack to be used in the reconstruction. The specified order is
        kept if `skip_stacks_ordering` is True.
    p_do_srr_assessment : :obj:`bool`
        If super-resolution assessment should be done.

    Outputs
    -------
    outputnode.t2ws_filtered : :list: of filenames
        Low-resolution T2w images
    outputnode.masks_filtered : :list: of filenames
        Low-resolution T2w masks
    outputnode.stacks_order : :list: of ids
        Order in which the stacks should be processed
    outputnode.report_image : filename
        Output PNG image for report
    outputnode.motion_tsv : filename
        Output TSV file with results used to create `report_image`
    outputnode.ground_truth : filename
        Ground truth image used for `srr_assessment`
        (optional, if `p_do_srr_assessment=True)

    Example
    -------
    >>> from pymialsrtk.pipelines.workflows import input_stage
    >>> input_mgmt_stage = input_stage.create_input_stage(
            p_bids_dir="bids_data",
            p_sub_ses="sub-01_ses-1",
            p_sub_path="sub-01/ses-1/anat",
            p_use_manual_masks=False,
            p_skip_stacks_ordering=False,
            p_do_srr_assessment=False,
            name="input_mgmt_stage",
        )
    >>> input_mgmt_stage.run()  # doctest: +SKIP

    """

    input_stage = pe.Workflow(name=name)

    output_fields = ["t2ws_filtered", "masks_filtered", "stacks_order"]

    if not p_skip_stacks_ordering:
        output_fields += ["report_image", "motion_tsv"]

    if p_do_srr_assessment:
        output_fields += [
            "hr_reference_image",
            "hr_reference_mask",
            "hr_reference_labels",
        ]

    if p_do_reconstruct_labels:
        output_fields += ["labels_filtered"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode",
    )

    dg_fields = ["T2ws"]
    if p_use_manual_masks:
        dg_fields += ["masks"]
    if p_do_reconstruct_labels:
        dg_fields += ["labels"]

    dg = pe.Node(
        interface=DataGrabber(
            outfields=dg_fields,
            base_directory=p_bids_dir,
            template="*",
            raise_on_empty=True,
            sort_filelist=True,
        ),
        name="data_grabber",
    )

    dict_templates = {}

    t2ws_template = os.path.join(
        p_sub_path, "anat", p_sub_ses + "*_run-*_T2w.nii.gz"
    )
    dict_templates["T2ws"] = t2ws_template

    if p_use_manual_masks:
        if p_masks_desc is not None:
            masks_template = os.path.join(
                "derivatives",
                p_masks_derivatives_dir,
                p_sub_path,
                "anat",
                "_".join(
                    [
                        p_sub_ses,
                        "*_run-*",
                        "_desc-" + p_masks_desc,
                        "*mask.nii.gz",
                    ]
                ),
            )
        else:
            masks_template = os.path.join(
                "derivatives",
                p_masks_derivatives_dir,
                p_sub_path,
                "anat",
                "_".join([p_sub_ses, "*run-*", "*mask.nii.gz"]),
            )

        dict_templates["masks"] = masks_template

        if p_do_reconstruct_labels:
            labels_template = os.path.join(
                "derivatives",
                p_labels_derivatives_dir,
                p_sub_path,
                "anat",
                "_".join([p_sub_ses, "*run-*", "*labels.nii.gz"]),
            )
            dict_templates["labels"] = labels_template

    dg.inputs.field_template = dict_templates

    if not p_use_manual_masks:
        brainMask = pe.MapNode(
            interface=preprocess.BrainExtraction(),
            name="brainExtraction",
            iterfield=["in_file"],
        )

        brainMask.inputs.in_ckpt_loc = pkg_resources.resource_filename(
            "pymialsrtk",
            os.path.join(
                "data",
                "Network_checkpoints",
                "Network_checkpoints_localization",
                "Unet.ckpt-88000.index",
            ),
        ).split(".index")[0]
        brainMask.inputs.threshold_loc = 0.49
        brainMask.inputs.in_ckpt_seg = pkg_resources.resource_filename(
            "pymialsrtk",
            os.path.join(
                "data",
                "Network_checkpoints",
                "Network_checkpoints_segmentation",
                "Unet.ckpt-20000.index",
            ),
        ).split(".index")[0]
        brainMask.inputs.threshold_seg = 0.5

    check_input = pe.Node(
        preprocess.CheckAndFilterInputStacks(),
        name="filter_input",
    )

    check_input.inputs.stacks_id = p_stacks if p_stacks else []

    if not p_skip_stacks_ordering:
        stacksOrdering = pe.Node(
            interface=preprocess.StacksOrdering(
                sub_ses=p_sub_ses, verbose=p_verbose
            ),
            name="stackOrdering",
        )
    else:
        stacksOrdering = pe.Node(
            interface=IdentityInterface(fields=["stacks_order"]),
            name="stackOrdering",
        )
        stacksOrdering.inputs.stacks_order = p_stacks

    if p_do_srr_assessment:

        rg = pe.Node(
            interface=DataGrabber(
                outfields=["T2w", "mask", "labels"],
                base_directory=p_bids_dir,
                template="*",
                raise_on_empty=True,
                sort_filelist=True,
            ),
            name="reference_grabber",
        )

        t2w_template = os.path.join(
            p_sub_path, "anat", p_sub_ses + "_desc-iso_T2w.nii.gz"
        )

        mask_template = os.path.join(
            p_sub_path, "anat", p_sub_ses + "_desc-iso_mask.nii.gz"
        )
        labels_template = os.path.join(
            p_sub_path, "anat", p_sub_ses + "_desc-iso_labels.nii.gz"
        )

        rg.inputs.field_template = dict(
            T2w=t2w_template, mask=mask_template, labels=labels_template
        )
    input_stage.connect(dg, "T2ws", check_input, "input_images")

    if p_use_manual_masks:
        # Directly connect the input_masks to the output and stack ordering
        input_stage.connect(dg, "masks", check_input, "input_masks")
        input_stage.connect(
            check_input, "output_masks", outputnode, "masks_filtered"
        )
        if not p_skip_stacks_ordering:
            input_stage.connect(
                check_input, "output_masks", stacksOrdering, "input_masks"
            )
    else:
        # Compute the masks, map them to the output and stack ordering
        input_stage.connect(check_input, "output_images", brainMask, "in_file")
        input_stage.connect(
            brainMask, "out_file", outputnode, "masks_filtered"
        )
        if not p_skip_stacks_ordering:
            input_stage.connect(
                brainMask, "out_file", stacksOrdering, "input_masks"
            )

    input_stage.connect(
        check_input, "output_images", outputnode, "t2ws_filtered"
    )

    input_stage.connect(
        stacksOrdering, "stacks_order", outputnode, "stacks_order"
    )

    if p_do_reconstruct_labels:
        input_stage.connect(dg, "labels", check_input, "input_labels")
        input_stage.connect(
            check_input, "output_labels", outputnode, "labels_filtered"
        )

    if not p_skip_stacks_ordering:
        input_stage.connect(
            stacksOrdering, "report_image", outputnode, "report_image"
        )
        input_stage.connect(
            stacksOrdering, "motion_tsv", outputnode, "motion_tsv"
        )

    if p_do_srr_assessment:
        input_stage.connect(rg, "T2w", outputnode, "hr_reference_image")
        input_stage.connect(rg, "mask", outputnode, "hr_reference_mask")
        input_stage.connect(rg, "labels", outputnode, "hr_reference_labels")

    return input_stage
