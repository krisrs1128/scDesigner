FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y sudo curl git less vim
RUN mkdir -p /home/user && chmod 777 /home/user
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/user/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/home/user/miniconda/bin:${PATH}"
RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge anndata formulaic copulas lightning muon mudata altair papermill
RUN pip install pamona
RUN pip install --upgrade pip ipython ipykernel