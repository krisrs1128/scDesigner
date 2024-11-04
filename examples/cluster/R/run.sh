#!/bin/bash

# condor_submit -i requirements.submit

# download relevant data and code
wget https://drive.google.com/uc?export=download&id=1dhgi6oEQmyex_nRfFYUkRY0c2q0E3BgP
git clone git@github.com:krisrs1128/scDesigner.git

# run the script
cd scDesigner/examples/
Rscript -e "rmarkdown::render('scalability_study.Rmd', params = list(config=1))"

# collect the data
cp *.csv ../../