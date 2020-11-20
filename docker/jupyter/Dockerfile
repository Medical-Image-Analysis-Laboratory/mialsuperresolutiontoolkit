# Use an initial image, where all MIAL Super-Resolution BIDSApp dependencies are installed, as a parent image

ARG MAIN_DOCKER
FROM $MAIN_DOCKER

###############################
## A little Docker magic here
# Force bash always
#RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# Default miniconda installation
ENV CONDA_ENV_PATH /opt/conda
ENV MY_CONDA_PY3ENV "pymialsrtk-env"
# This is how you will activate this conda environment
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"

# Pull the environment name out of the environment.yml
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Install pymialsrtk inside installed conda environment (see environment.yml)
#RUN /bin/bash -c "$CONDA_ACTIVATE && conda update jupyter_core jupyter_client"

# Install jupyter extensions 
RUN bash -c '$CONDA_ACTIVATE && jupyter nbextension enable exercise2/main && jupyter nbextension enable spellchecker/main'
RUN bash -c '$CONDA_ACTIVATE && jupyter nbextension install --py jupyter_highlight_selected_word && jupyter nbextension enable highlight_selected_word/main'

# Install niwidget from nipy and create the jupyter notebook kernel linked to the conda environment
RUN bash -c "$CONDA_ACTIVATE && pip install git+git://github.com/nipy/niwidgets && python3 -m ipykernel install --prefix=/opt/conda --name $MY_CONDA_PY3ENV"

RUN mkdir -p /app/notebooks
COPY . /app/notebooks

WORKDIR /app/notebooks

# RUN printf "#!/bin/bash \n %s && jupyter-notebook --allow-root --no-browser --ip=\"0.0.0.0\" --NotebookApp.token=\'mial\'" "$CONDA_ACTIVATE" > /app/launch_jupyter_notebook.sh
RUN printf "#!/bin/bash \n %s && jupyter-lab --allow-root --no-browser --ip=\"0.0.0.0\" --NotebookApp.token=\'mial\'" "$CONDA_ACTIVATE" > /app/launch_jupyter_lab.sh
# RUN chmod 755 /app/launch_jupyter_notebook.sh
RUN chmod 755 /app/launch_jupyter_lab.sh

# ENTRYPOINT ["/app/launch_jupyter_notebook.sh"]
ENTRYPOINT ["/app/launch_jupyter_lab.sh"]

#ENTRYPOINT ["/opt/mialsuperresolutiontoolkit/run.py"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

#Metadata
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="MIAL Super-Resolution Toolkit Jupyter Notebooks"
LABEL org.label-schema.description="Provides a jupyter notebook environment for developing notebooks"
LABEL org.label-schema.url="https://mialsuperresolutiontoolkit.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run --rm -v ~/data/bids_dataset:/tmp -t sebastientourbier/mialsuperresolutionbidsapp:${VERSION}"

