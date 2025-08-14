#!/usr/env/bin bash

# generate data from command line
python -m cytodatagen \
    --add-switch-effect \
    --add-signal-effect \
    --add-batch-effect \
    --add-noise-effect \
    --add-sinh-effect \
    -o artifacts/cytodata --format fcs \
    --verbose
