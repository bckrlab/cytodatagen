#!/usr/bin/env bash

python -m cytodatagen.v2 -o artifacts/v2 --format h5ad --verbose \
    --ct-mean-loc 5
