# ![MIALSRTK logo](https://raw.githubusercontent.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/dev-pgd-hk/documentation/images/mialsrtk-logo.png)
---

Copyright © 2016-2020 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

This software is distributed under the open-source BSD 3-Clause License. See [LICENSE](LICENSE.txt) file for details.

---
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?include_prereleases) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4290209.svg)](https://doi.org/10.5281/zenodo.4290209) [![Docker Pulls](https://img.shields.io/docker/pulls/sebastientourbier/mialsuperresolutiontoolkit?label=docker%20pulls%20%28v1%29)](https://hub.docker.com/repository/docker/sebastientourbier/mialsuperresolutiontoolkit) [![Build Status](https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit.svg?branch=master)](https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit) [![CircleCI](https://circleci.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit.svg?style=shield)](https://app.circleci.com/pipelines/github/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit) [![Code Coverage](https://app.codacy.com/project/badge/Coverage/a27593d6fae7436eb2cd65b80f3342c3)](https://www.codacy.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?utm_source=github.com&utm_medium=referral&utm_content=Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit&utm_campaign=Badge_Coverage) [![Documentation Status](https://readthedocs.org/projects/mialsrtk/badge/?version=latest)](https://mialsrtk.readthedocs.io/en/latest/?badge=latest) [![Code Quality](https://app.codacy.com/project/badge/Grade/a27593d6fae7436eb2cd65b80f3342c3)](https://www.codacy.com/gh/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit&amp;utm_campaign=Badge_Grade) [![Github All Contributors](https://img.shields.io/github/all-contributors/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)](#credits-) 

The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) provides a set of C++ and Python tools necessary to perform motion-robust super-resolution fetal MRI reconstruction. 

The original C++ MIALSRTK library includes all algorithms and methods for brain extraction, intensity standardization, motion estimation and super-resolution. It uses the CMake build system and depends on the open-source image processing Insight ToolKit (ITK) library, the command line parser TCLAP library and OpenMP for multi-threading.

MIALSRTK has been recently extended with the `pymialsrtk` Python3 library following recent advances in standardization of neuroimaging data organization and processing workflows such as the [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/) and [BIDS App](https://bids-apps.neuroimaging.io/) standards. This library has a modular architecture built on top of the Nipype dataflow library which consists of (1) processing nodes that interface with each of the MIALSRTK C++ tools and (2) a processing pipeline that links the interfaces in a common workflow. 

The processing pipeline with all dependencies including the C++ MIALSRTK tools are encapsulated in a Docker image container, which is now distributed as a `BIDS App` which handles datasets organized following the BIDS standard. 

All these design considerations allow us not only to (1) represent the entire processing pipeline as an *execution graph, where each MIALSRTK C++ tools are connected*, but also to (2) provide a *mecanism to record data provenance and execution details*, and to (3) easily customize the BIDS App to suit specific needs as interfaces with *new tools can be added with relatively little effort* to account for additional algorithms.

The Docker image of the BIDS App is freely available @ [Docker store](https://store.docker.com/community/images/sebastientourbier/mialsuperresolutiontoolkit-bidsapp).  

## Resources

*   **BIDS App and `pymialsrtk` documentation:** [https://mialsrtk.readthedocs.io/](https://mialsrtk.readthedocs.io/)

*   **Source:** [https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit)

*   **Bug reports:** [https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues](https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues)

*   **For C++ developers/contributors:** 
    *   [Installation instructions on Ubuntu](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_ubuntu.md) / [Installation instructions on MACOSX](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_mac.md)
    *   [C++ code documentation](https://htmlpreview.github.io/?https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/doxygen_html/index.html)

## Usage

The BIDS App of MIALSRTK has the following command line arguments:

    $ docker run -it sebastientourbier/mialsuperresolutiontoolkit-bidsapp --help

    usage: run.py [-h]
                  [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                  [--param_file PARAM_FILE] [--manual] [-v]
                  bids_dir output_dir {participant}

    Entrypoint script to the MIALSRTK BIDS App

    positional arguments:
      bids_dir              The directory with the input dataset formatted
                            according to the BIDS standard.
      output_dir            The directory where the output files should be stored.
                            If you are running group level analysis this folder
                            should be prepopulated with the results of
                            theparticipant level analysis.
      {participant}         Level of the analysis that will be performed. Only
                            participant is available

    optional arguments:
      -h, --help            show this help message and exit
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
      --manual              Use manual brain masks found in
                            <output_dir>/manual_masks/ directory
      -v, --version         show program's version number and exit

## Credits 

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/sebastientourbier"><img src="https://avatars3.githubusercontent.com/u/22279770?v=4" width="100px;" alt=""/><br /><sub><b>Sébastien Tourbier</b></sub></a><br /><a href="#design-sebastientourbier" title="Design">🎨</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Code">💻</a> <a href="#example-sebastientourbier" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Documentation">📖</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3Asebastientourbier" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/pdedumast"><img src="https://avatars2.githubusercontent.com/u/19345763?v=4" width="100px;" alt=""/><br /><sub><b>Priscille de Dumast</b></sub></a><br /><a href="#example-pdedumast" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Code">💻</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/hamzake"><img src="https://avatars2.githubusercontent.com/u/27707790?v=4" width="100px;" alt=""/><br /><sub><b>hamzake</b></sub></a><br /><a href="#example-hamzake" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Code">💻</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/meribach"><img src="https://avatars3.githubusercontent.com/u/2786897?v=4" width="100px;" alt=""/><br /><sub><b>Meritxell Bach</b></sub></a><br /><a href="#fundingFinding-meribach" title="Funding Finding">🔍</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
