![MIALSRTK logo](https://cloud.githubusercontent.com/assets/22279770/24004342/5e78836a-0a66-11e7-8b7d-058961cfe8e8.png)
---

Copyright © 2016-2017 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

This software is distributed under the open-source BSD 3-Clause License. See LICENSE file for details.

---
[![DOI](https://zenodo.org/badge/85210898.svg)](https://zenodo.org/badge/latestdoi/85210898) [![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-) 

   
The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ image processing tools necessary to perform motion-robust super-resolution fetal MRI reconstruction. This toolkit, supported by the Swiss National Science Foundation (grant SNSF-141283), includes all algorithms and methods for brain extraction [1], intensity standardization [1,2], motion estimation and super-resolution [2] developed during my PhD. It uses the CMake build system and depends on the open-source image processing Insight ToolKit (ITK) library, the command line parser TCLAP library and OpenMP for multi-threading. The USAGE message of each tool can be obained using either the *-h* or *--help* flag. 

A Docker image is provided to facilitate the deployment and freely available @ [Docker store](https://store.docker.com/community/images/sebastientourbier/mialsuperresolutiontoolkit).  

* Please acknowledge this software in any work reporting results using MIALSRTK by citing the following:

[1] S. Tourbier, X. Bresson, P. Hagmann, M. B. Cuadra. (2019, March 19). sebastientourbier/mialsuperresolutiontoolkit: MIAL Super-Resolution Toolkit v1.0 (Version v1.0). Zenodo. http://doi.org/10.5281/zenodo.2598448

[2] S. Tourbier, C. Velasco-Annis, V. Taimouri, P. Hagmann, R. Meuli, S. K. Warfield, M. B. Cuadra,
A. Gholipour, *Automated template-based brain localization and extraction for fetal brain MRI
reconstruction*, Neuroimage (2017) In Press. [DOI](https://doi.org/10.1016/j.neuroimage.2017.04.004)

[3] S. Tourbier, X. Bresson, P. Hagmann, R. Meuli, M. B. Cuadra, *An efficient total variation
algorithm for super-resolution in fetal brain MRI with adaptive regularization*, Neuroimage 118
(2015) 584-597. [DOI](https://doi.org/10.1016/j.neuroimage.2015.06.018)

# Documentation #

* FOR USERS: [How to run the Docker image](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/userguide_docker.md)
* FOR DEVELOPERS/CONTRIBUTORS: [Installation instructions on Ubuntu](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_ubuntu.md) / [Installation instructions on MACOSX](https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/devguide_mac.md)
* [Doxygen source code documentation](https://htmlpreview.github.io/?https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/doxygen_html/index.html)

# Credits #

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/sebastientourbier"><img src="https://avatars3.githubusercontent.com/u/22279770?v=4" width="100px;" alt=""/><br /><sub><b>Sébastien Tourbier</b></sub></a><br /><a href="#design-sebastientourbier" title="Design">🎨</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Code">💻</a> <a href="#example-sebastientourbier" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=sebastientourbier" title="Documentation">📖</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/pulls?q=is%3Apr+reviewed-by%3Asebastientourbier" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/pdedumast"><img src="https://avatars2.githubusercontent.com/u/19345763?v=4" width="100px;" alt=""/><br /><sub><b>Priscille de Dumast</b></sub></a><br /><a href="#example-pdedumast" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=pdedumast" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/hamzake"><img src="https://avatars2.githubusercontent.com/u/27707790?v=4" width="100px;" alt=""/><br /><sub><b>hamzake</b></sub></a><br /><a href="#example-hamzake" title="Examples">💡</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Tests">⚠️</a> <a href="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/commits?author=hamzake" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/meribach"><img src="https://avatars3.githubusercontent.com/u/2786897?v=4" width="100px;" alt=""/><br /><sub><b>Meritxell Bach</b></sub></a><br /><a href="#fundingFinding-meribach" title="Funding Finding">🔍</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
