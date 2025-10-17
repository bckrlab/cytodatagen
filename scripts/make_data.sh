#!/usr/bin/env bash

SEED=19
SUBJECTS="config/cli/data/subjects.json"
TRANSFORMS="config/cli/data/transforms.json"
N_CELLS_MIN=10000
N_CELLS_MAX=10000

python -m cytodatagen.cli.data -o artifacts/demo --format h5ad --verbose \
    --n-cells-min $N_CELLS_MIN \
    --n-cells-max $N_CELLS_MAX \
    --subjects $SUBJECTS \
    --transforms $TRANSFORMS \
    --seed $SEED \
