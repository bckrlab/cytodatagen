#!/usr/bin/env bash

N_CT=5
N_SIGNAL_CT=2
N_MARKER=5
N_SIGNAL_MARKER=2
SEED=19

python -m cytodatagen -o artifacts/demo --format h5ad --verbose \
    --ct-mean-loc 5 \
    --ct-mean-scale 2.0 \
    --ct-scale-min 0.5 \
    --ct-scale-max 1.0 \
    --n-ct $N_CT \
    --n-signal-ct $N_SIGNAL_CT \
    --n-marker $N_MARKER \
    --n-signal-marker $N_SIGNAL_MARKER \
    --seed $SEED
