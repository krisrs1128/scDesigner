FROM bioconductor/tidyverse

RUN R -e "install.packages(c('glue', 'reticulate'), repos='https://cloud.r-project.org/')"
RUN R -e "reticulate::py_install('scipy')"