#!/bin/bash

# condor_submit -i requirements.submit

# download relevant data and code
wget https://drive.google.com/uc?export=download&id=1dhgi6oEQmyex_nRfFYUkRY0c2q0E3BgP
wget https://drive.google.com/uc?export=download&id=1xLB9NMT859kpz-iTzkNlPDL32dDPyLOx
wget https://drive.google.com/uc?export=download&id=1VBJ3lcUbG-1V-2o_saRgW4tgmpPPkdbO
git clone git@github.com:krisrs1128/scDesigner.git

# reorganize and extract the data
mv scalability_configurations.csv scDesigner/examples/data/
mv million_cells* scDesigner/examples/data/

cd scDesigner/examples/data/
tar -zxvf million_cells.tar.gz

# run the script
cd ../
Rscript -e "rmarkdown::render('scalability_study.Rmd', params = list(config=${process}))"

# collect the data
cp *.csv $CONDOR_RUN_DIR