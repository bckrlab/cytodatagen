#!/usr/bin/env bash

N_SAMPLE=5
N_CELLS=2000
N_MARKERS=5
N_BATCH=2
SEED=19

# shared arguments between different effect calls
ARGS="
    --n-samples-per-class ${N_SAMPLE}
    --n-cells-min ${N_CELLS}
    --n-cells-max ${N_CELLS}
    --n-markers ${N_MARKERS}
    --n-batch ${N_BATCH}
    --seed ${SEED}
"

python -m cytodatagen \
    ${ARGS} \
    -o artifacts/demo/effect_0/ --format h5ad \
    --verbose

python -m cytodatagen ${ARGS} \
    --add-switch-effect \
    -o artifacts/demo/effect_1/ --format h5ad \
    --verbose

python -m cytodatagen ${ARGS} \
    --add-switch-effect \
    --add-signal-effect \
    -o artifacts/demo/effect_2/ --format h5ad \
    --verbose

python -m cytodatagen ${ARGS} \
    --add-switch-effect \
    --add-signal-effect \
    --add-batch-effect \
    -o artifacts/demo/effect_3/ --format h5ad \
    --verbose

python -m cytodatagen ${ARGS} \
    --add-switch-effect \
    --add-signal-effect \
    --add-batch-effect \
    --add-noise-effect \
    -o artifacts/demo/effect_4/ --format h5ad \
    --verbose

python -m cytodatagen ${ARGS} \
    --add-switch-effect \
    --add-signal-effect \
    --add-batch-effect \
    --add-noise-effect \
    --add-sinh-effect \
    -o artifacts/demo/effect_5/ --format h5ad \
    --verbose
