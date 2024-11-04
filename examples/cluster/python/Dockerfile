FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y curl git less
RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge anndata formulaic copulas lightning muon mudata altair papermill
RUN pip install pamona
RUN pip install --upgrade pip ipython ipykernel
RUN apt-get install vim