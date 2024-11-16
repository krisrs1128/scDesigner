#!/bin/bash

# download relevant data and code
export HOME=/home/user
export NUMBA_CACHE_DIR=$_CONDOR_SCRATCH_DIR
export PYTORCH_KERNEL_CACHE_PATH=$_CONDOR_SCRATCH_DIR
curl -L "https://drive.google.com/uc?export=download&id=1dhgi6oEQmyex_nRfFYUkRY0c2q0E3BgP" > scalability_configurations.csv
curl -L "https://uwmadison.box.com/shared/static/9ucgdbgbt119tx1b9y3v9u7wjm9ef920.h5ad" > million_cells.h5ad
git clone https://krisrs1128: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@github.com/krisrs1128/scDesigner.git

mkdir scDesigner/examples/data
mv scalability_configurations.csv scDesigner/examples/data/
mv million_cells.h5ad scDesigner/examples/data/

# run the script
cd scDesigner/examples
config=$1
python3 -m scalability_study --config $config
cp *.csv $_CONDOR_SCRATCH_DIR
tar -zcvf lightning_logs_$config.tar.gz lightning_logs
mv lightning_logs_$config.tar.gz $_CONDOR_SCRATCH_DIR