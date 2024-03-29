.. _contributing:

*************
Contributing 
*************

This project follows the `all contributors specification <https://allcontributors.org/>`_. 
Contributions in many different ways are welcome!

Contribution Types
------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

MIALSRTK could always use more documentation, whether as part of the
official MIALSRTK docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to create an issue at https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `MIALSRTK` for local development.

1. Fork the `mialsuperresolutiontoolkit` repo on GitHub.

2. Clone your fork locally::

    git clone git@github.com:your_name_here/mialsuperresolutiontoolkit.git
    cd mialsuperresolutiontoolkit

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

4. Now you can make your changes locally. If you add a new node in a pipeline or a completely new pipeline, we encourage you to rebuild the BIDS App Docker image (See :ref:`BIDS App build instructions <instructions_bisapp_build>`) and test it on the sample dataset (`mialsuperresolutiontoolkit/data/`). 

.. note::
	Please keep your commit the most specific to a change it describes. It is highly advice to track unstaged files with ``git status``, add a file involved in the change to the stage one by one with ``git add <file>``. The use of ``git add .`` is highly disencouraged. When all the files for a given change are staged, commit the files with a brieg message using ``git commit -m "[COMMIT_TYPE]: Your detailed description of the change."`` that describes your change and where ``[COMMIT_TYPE]`` can be ``[FIX]`` for a bug fix, ``[ENH]`` for a new feature, ``[MAINT]`` for code maintenance and typo fix, ``[DOC]`` for documentation, ``[CI]`` for continuous integration testing.

5. When you're done making changes, push your branch to GitHub::

    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. If the pull request adds functionality, the docs should be updated (See :ref:`documentation build instructions <instructions_docs_build>`). 

2. The pull request should work for Python 3.6. Check
   https://app.circleci.com/pipelines/github/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit
   and make sure that the tests pass.

.. _instructions_bisapp_build:

How to build the BIDS App locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Go to the clone directory of your fork and run the script `build_bidsapp.sh` ::

    cd mialsuperresolutiontoolkit
    sh build_bidsapp.sh

Note that the tag of the version of the image will be extracted from ``pymialsrtk/info.py`` where you might need to change the version to not overwrite an other existing image with the same version.

.. _instructions_pymialsrtk_install:

How to install `pyMIALSTK` locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the `MIALSRTK` conda environment `pymialsrtk-env` that provides a Python 3.6 environment::

    cd mialsuperresolutiontoolkit
    conda env create -f docker/bidsapp/environment.yml

2. Activate the `pymialsrtk-env` conda environment and install `pymialsrtk` ::

    conda activate pymialsrtk-env
    pip install .

.. _instructions_docs_build:

How to build the documentation locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install the `MIALSRTK` conda environment `pymialsrtk-env` with sphinx and all extensions to generate the documentation::

    cd mialsuperresolutiontoolkit
    conda env create -f docker/bidsapp/environment.yml

2. Activate the MIALSRTK conda environment `pymialsrtk-env` and install `pymialsrtk` ::

    conda activate pymialsrtk-env
    pip install .

3. Run the script `build_sphinx_docs.sh` to generate the HTML documentation in ``documentation/_build/html``::

    bash build_sphinx_docs.sh

.. note::
	Make sure to have activated the conda environment `pymialsrtk-env` before running the script `build_sphinx_docs.sh`.

Not listed as a contributor?
----------------------------

This is easy, `MIALSRTK` has the `all contributors bot <https://allcontributors.org/docs/en/bot/usage>`_ installed.

Just comment on Issue or Pull Request (PR), asking `@all-contributors` to add you as contributor::

    @all-contributors please add <github_username> for <contributions>

`<contribution>`: See the `Emoji Key Contribution Types Reference <https://github.com/all-contributors/all-contributors/blob/master/docs/emoji-key.md>`_ for a list of valid `contribution` types.

The all-contributors bot will create a PR to add you in the README and reply with the pull request details.

When the PR is merged you will have to make an extra Pull Request where you have to:

    1. add your entry in the `.zenodo.json` (for that you will need an ORCID ID - https://orcid.org/). Doing so, you will appear as a contributor on Zenodo in the future version releases of MIALSRTK. Zenodo is used by MIALSRTK to publish and archive each of the version release with a unique Digital Object Identifier (DOI), which can then be used for citation.

    2. update the content of the table in `documentation/contributors.rst` with the new content generated by the bot in the README. Doing so, you will appear in the :ref:`Contributing Page <contributing>`.

------------

This document has been inspired and adapted from `these great contributing guidelines <https://github.com/dPys/MIALSRTK/edit/master/docs/contributing.rst>`_.