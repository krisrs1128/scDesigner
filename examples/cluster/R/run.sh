#!/bin/bash

# download relevant data and code
wget -O scalability_configurations.csv "https://drive.google.com/uc?export=download&id=1dhgi6oEQmyex_nRfFYUkRY0c2q0E3BgP"
git clone https://krisrs1128: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@github.com/krisrs1128/scDesigner.git

# reorganize and extract the data
mkdir scDesigner/examples/data
mv scalability_configurations.csv scDesigner/examples/data/
mv /staging/ksankaran/million_cells.tar.gz scDesigner/examples/data/

cd scDesigner/examples/data/
tar -zxvf million_cells.tar.gz

# run the script and collect data
cd ../
config=$(($1 + 1))
Rscript -e "rmarkdown::render('scalability_study.Rmd', params = list(config=$config))"
cp *.csv $_CONDOR_SCRATCH_DIR