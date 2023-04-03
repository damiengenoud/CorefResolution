#!/bin/bash
module --quiet purge 
module load PyTorch/1.8.1-fosscuda-2020b
python3 -m venv $USERWORK/venvs/trans_nopt --clear 
source $USERWORK/venvs/trans_nopt/bin/activate
python3 -m pip install jsonlines
python3 -m pip install toml
python3 -m pip install transformers==3.2.0
# /cluster/work/users/egilron/litbank4wl-coref/saga-i3.sh