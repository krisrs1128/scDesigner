FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# # setup miniconda
# RUN apt-get update && apt-get install -y wget && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Miniconda3-latest-Linux-x86_64.sh && \
#     echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc && \
#     /opt/conda/bin/conda init bash

# install necessary packages
# RUN /opt/conda/bin/conda install -c conda-forge anndata formulaic copulas pytorch-lightning muon mudata pamona altair
RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge anndata formulaic copulas lightning muon mudata altair
RUN pip install pamona
