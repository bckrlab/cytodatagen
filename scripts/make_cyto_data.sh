#!/usr/env/bin bash

python -m cytodatagen \
    -o artifacts/cytodata --format h5ad \
    --verbose

if [ false ]; then
    # generate data from command line
    python -m cytodatagen \
        --add-switch-effect \
        --add-signal-effect \
        --add-batch-effect \
        --add-noise-effect \
        --add-sinh-effect \
        -o artifacts/cytodata --format fcs \
        --verbose
fi