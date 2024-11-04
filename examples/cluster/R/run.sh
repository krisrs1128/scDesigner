#!/bin/bash

# condor_submit -i requirements.submit

# download relevant data and code
wget -O scalability_configurations.csv "https://drive.google.com/uc?export=download&id=1dhgi6oEQmyex_nRfFYUkRY0c2q0E3BgP"
wget -O million_cells.tar.gz "https://uwmadison.box.com/shared/static/fooedjcbf1roafwuy9kmr810rfvxztlc.gz"
git clone https://krisrs1128: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@github.com/krisrs1128/scDesigner.git

# reorganize and extract the data
mkdir scDesigner/examples/data
mv scalability_configurations.csv scDesigner/examples/data/
mv million_cells* scDesigner/examples/data/

cd scDesigner/examples/data/
tar -zxvf million_cells.tar.gz

# run the script and collect data
cd ../
Rscript -e "rmarkdown::render('scalability_study.Rmd', params = list(config=${ProcID}))"
cp *.csv $CONDOR_SCRATCH_DIR