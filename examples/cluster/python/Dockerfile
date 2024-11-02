FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN conda install -c conda-forge mamba
RUN mamba install -c conda-forge anndata formulaic copulas lightning muon mudata altair
RUN pip install pamona
