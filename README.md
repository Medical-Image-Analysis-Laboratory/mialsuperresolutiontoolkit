# ![MIALSRTK logo](https://raw.githubusercontent.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/dev-pgd-hk/documentation/images/mialsrtk-logo.png)
---

Copyright © 2016-2023 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

This software is distributed under the open-source BSD 3-Clause License. See [LICENSE](LICENSE.txt) file for details.

---
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?include_prereleases) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4290209.svg)](https://doi.org/10.5281/zenodo.4290209) [![Docker Pulls](https://img.shields.io/docker/pulls/sebastientourbier/mialsuperresolutiontoolkit?label=docker%20pulls)](https://hub.docker.com/repository/docker/sebastientourbier/mialsuperresolutiontoolkit) [![Build Status](https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit.svg?branch=master)](https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit) [![CircleCI](https://circleci.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit.svg?style=shield)](https://app.circleci.com/pipelines/github/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit) [![Code Coverage](https://app.codacy.com/project/badge/Coverage/a27593d6fae7436eb2cd65b80f3342c3)](https://www.codacy.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?utm_source=github.com&utm_medium=referral&utm_content=Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit&utm_campaign=Badge_Coverage) [![Documentation Status](https://readthedocs.org/projects/mialsrtk/badge/?version=latest)](https://mialsrtk.readthedocs.io/en/latest/?badge=latest) [![Code Quality](https://app.codacy.com/project/badge/Grade/a27593d6fae7436eb2cd65b80f3342c3)](https://www.codacy.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit&amp;utm_campaign=Badge_Grade) [![Github All Contributors](https://img.shields.io/github/all-contributors/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)](#credits-) 

The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) provides a set of C++ and Python tools necessary to perform motion-robust super-resolution fetal MRI reconstruction. 

The original C++ MIALSRTK library includes all algorithms and methods for brain extraction, intensity standardization, motion estimation and super-resolution. It uses the CMake build system and depends on the open-source image processing Insight ToolKit (ITK) library, the command line parser TCLAP library and OpenMP for multi-threading.

MIALSRTK has been recently extended with the `pymialsrtk` Python3 library following recent advances in standardization of neuroimaging data organization and processing workflows such as the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) and [BIDS App](https://bids-apps.neuroimaging.io/) standards. This library has a modular architecture built on top of the Nipype dataflow library which consists of (1) processing nodes that interface with each of the MIALSRTK C++ tools and (2) a processing pipeline that links the interfaces in a common workflow. 

The processing pipeline with all dependencies including the C++ MIALSRTK tools are encapsulated in a Docker image container, which handles datasets organized following the BIDS standard and is distributed as a `BIDS App` @ [Docker Hub](https://store.docker.com/community/images/sebastientourbier/mialsuperresolutiontoolkit-bidsapp). For execution on high-performance computing cluster, a Singularity image is also made freely available @ [Sylabs Cloud](https://cloud.sylabs.io/library/_container/5fe46eb7517f0358917ab76c). To facilitate the use of Docker or Singularity, `pymialsrtk` provides two Python commandline wrappers (`mialsuperresolutiontoolkit_docker` and `mialsuperresolutiontoolkit_singularity`) that can generate and run the appropriate command.

All these design considerations allow us not only to (1) represent the entire processing pipeline as an *execution graph, where each MIALSRTK C++ tools are connected*, but also to (2) provide a *mecanism to record data provenance and execution details*, and to (3) easily customize the BIDS App to suit specific needs as interfaces with *new tools can be added with relatively little effort* to account for additional algorithms.

### Resources

*   **BIDS App and `pymialsrtk` documentation:** [https://mialsrtk.readthedocs.io/](https://mialsrtk.readthedocs.io/)

*   **Source:** [https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)

*   **Bug reports:** [https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues)

*   **For C++ developers/contributors:** 
    *   [Installation instructions on Ubuntu](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_ubuntu.md) / [Installation instructions on MACOSX](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_mac.md)
    *   [C++ code documentation](https://htmlpreview.github.io/?https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/doxygen_html/index.html)

## Installation

*   Install Docker or Singularity engine

*   In a Python 3.7 environment, install `pymialsrtk` with `pip`:

        pip install pymialsrtk

*   You are ready to use MIALSRTK BIDS App wrappers! 

## Usage

`mialsuperresolutiontoolkit_docker` and `mialsuperresolutiontoolkit_singularity` python wrappers to the MIALSRTK BIDS App have the following command line arguments:

    $ mialsuperresolutiontoolkit_[docker|singularity] -h
    
    usage: mialsuperresolutiontoolkit_[docker|singularity] [-h]
                                         [--run_type {sr,preprocessing}]
                                         [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                                         [--param_file PARAM_FILE]
                                         [--openmp_nb_of_cores OPENMP_NB_OF_CORES]
                                         [--nipype_nb_of_cores NIPYPE_NB_OF_CORES]
                                         [--memory MEMORY]
                                         [--masks_derivatives_dir MASKS_DERIVATIVES_DIR]
                                         [--labels_derivatives_dir LABELS_DERIVATIVES_DIR]
                                         [--all_outputs] [-v] [--verbose]
                                         [--track_carbon_footprint]
                                         bids_dir output_dir {participant}

    Argument parser of the MIALSRTK BIDS App Python wrapper
    
    positional arguments:
      bids_dir              The directory with the input dataset formatted
                            according to the BIDS standard.
      output_dir            The directory where the output files should be stored.
                            If you are running group level analysis this folder
                            should be prepopulated with the results of the
                            participant level analysis.
      {participant}         Level of the analysis that will be performed. Only
                            participant is available
    
    optional arguments:
      -h, --help            show this help message and exit
      --run_type {sr,preprocessing}
                            Type of pipeline that is run. Can choose between
                            running the super-resolution pipeline (`sr`) or only
                            preprocessing (`preprocessing`).
      --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                            The label(s) of the participant(s) that should be
                            analyzed. The label corresponds to
                            sub-<participant_label> from the BIDS spec (so it does
                            not include "sub-"). If this parameter is not provided
                            all subjects should be analyzed. Multiple participants
                            can be specified with a space separated list.
      --param_file PARAM_FILE
                            Path to a JSON file containing subjects' exams
                            information and super-resolution total variation
                            parameters.
      --openmp_nb_of_cores OPENMP_NB_OF_CORES
                            Specify number of cores used by OpenMP threads
                            Especially useful for NLM denoising and slice-to-
                            volume registration. (Default: 0, meaning it will be
                            determined automatically)
      --nipype_nb_of_cores NIPYPE_NB_OF_CORES
                            Specify number of cores used by the Niype workflow
                            library to distribute the execution of independent
                            processing workflow nodes (i.e. interfaces)
                            (Especially useful in the case of slice-by-slice bias
                            field correction and intensity standardization steps
                            for example). (Default: 0, meaning it will be
                            determined automatically)
      --memory MEMORY       Limit the workflow to using the amount of specified
                            memory [in gb] (Default: 0, the workflow memory
                            consumption is not limited)
      --masks_derivatives_dir MASKS_DERIVATIVES_DIR
                            Use manual brain masks found in
                            ``<output_dir>/<masks_derivatives_dir>/`` directory
      --labels_derivatives_dir LABELS_DERIVATIVES_DIR
                            Use low-resolution labelmaps found in
                            ``<output_dir>/<labels_derivatives_dir>/`` directory.
      --all_outputs         Whether or not all outputs should be kept(e.g.
                            preprocessed LR images)
      -v, --version         show program's version number and exit
      --verbose             Verbose mode
      --track_carbon_footprint
                            Track carbon footprint with `codecarbon
                            <https://codecarbon.io/>`_ and save results in a CSV
                            file called ``emissions.csv`` in the
                            ``<bids_dir>/code`` directory.

## Credits 

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sebastientourbier"><img src="https://avatars3.githubusercontent.com/u/22279770?v=4?s=100" width="100px;" alt="Sébastien Tourbier"/><br /><sub><b>Sébastien Tourbier</b></sub></a><br /><a href="#design-sebastientourbier" title="Design">🎨</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Code">💻</a> <a href="#infra-sebastientourbier" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues?q=author%3Asebastientourbier" title="Bug reports">🐛</a> <a href="#example-sebastientourbier" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Documentation">📖</a> <a href="#ideas-sebastientourbier" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3Asebastientourbier" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pdedumast"><img src="https://avatars2.githubusercontent.com/u/19345763?v=4?s=100" width="100px;" alt="Priscille de Dumast"/><br /><sub><b>Priscille de Dumast</b></sub></a><br /><a href="#design-pdedumast" title="Design">🎨</a> <a href="#example-pdedumast" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Code">💻</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Documentation">📖</a> <a href="#ideas-pdedumast" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3Apdedumast" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hamzake"><img src="https://avatars2.githubusercontent.com/u/27707790?v=4?s=100" width="100px;" alt="hamzake"/><br /><sub><b>hamzake</b></sub></a><br /><a href="#example-hamzake" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Code">💻</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Documentation">📖</a> <a href="#ideas-hamzake" title="Ideas, Planning, & Feedback">🤔</a></td>
       <td align="center" valign="top" width="14.28%"><a href="http://t-sanchez.github.io"><img src="https://avatars.githubusercontent.com/u/14288440?v=4?s=100" width="100px;" alt="Thomas Sanchez"/><br /><sub><b>Thomas Sanchez</b></sub></a><br /><a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues?q=author%3At-sanchez" title="Bug reports">🐛</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=t-sanchez" title="Code">💻</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=t-sanchez" title="Documentation">📖</a> <a href="#example-t-sanchez" title="Examples">💡</a> <a href="#ideas-t-sanchez" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-t-sanchez" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3At-sanchez" title="Reviewed Pull Requests">👀</a></td>
       <td align="center" valign="top" width="14.28%"><a href="https://github.com/helenelajous"><img src="https://avatars.githubusercontent.com/u/58977568?v=4?s=100" width="100px;" alt="Hélène Lajous"/><br /><sub><b>Hélène Lajous</b></sub></a><br /><a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues?q=author%3Ahelenelajous" title="Bug reports">🐛</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=helenelajous" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3Ahelenelajous" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://wp.unil.ch/connectomics"><img src="https://avatars.githubusercontent.com/u/411192?v=4?s=100" width="100px;" alt="Patric Hagmann"/><br /><sub><b>Patric Hagmann</b></sub></a><br /><a href="#data-pahagman" title="Data">🔣</a> <a href="#fundingFinding-pahagman" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/meribach"><img src="https://avatars3.githubusercontent.com/u/2786897?v=4?s=100" width="100px;" alt="Meritxell Bach"/><br /><sub><b>Meritxell Bach</b></sub></a><br /><a href="#fundingFinding-meribach" title="Funding Finding">🔍</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
