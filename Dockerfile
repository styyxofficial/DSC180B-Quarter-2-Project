# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt update
RUN apt-get update && apt-get install -y aria2 nmap traceroute git

# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

RUN pip install --no-cache-dir --ignore-installed networkx pandas scikit-learn matplotlib plotly einops
RUN pip install --no-cache-dir --ignore-installed ONE-api ibllib
RUN pip install --no-cache-dir --ignore-installed "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
RUN pip install --no-cache-dir --ignore-installed git+https://github.com/yuanz271/vlgpax.git
RUN pip install --no-cache-dir --ignore-installed --force-reinstall numpy==1.23.1
RUN pip install --no-cache-dir --ignore-installed pyOpenSSL --upgrade

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]