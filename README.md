<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/cytodatagen.svg?branch=main)](https://cirrus-ci.com/github/<USER>/cytodatagen)
[![ReadTheDocs](https://readthedocs.org/projects/cytodatagen/badge/?version=latest)](https://cytodatagen.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/cytodatagen/main.svg)](https://coveralls.io/r/<USER>/cytodatagen)
[![PyPI-Server](https://img.shields.io/pypi/v/cytodatagen.svg)](https://pypi.org/project/cytodatagen/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/cytodatagen.svg)](https://anaconda.org/conda-forge/cytodatagen)
[![Monthly Downloads](https://pepy.tech/badge/cytodatagen/month)](https://pepy.tech/project/cytodatagen)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/cytodatagen)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

> :warning: cytodatagen takes a very naive approach to generate cytometry data.
> You might want to checkout [FlowCyPy](https://github.com/MartinPdeS/FlowCyPy) instead!
> FlowCyPy attempts to generate realistic SSC and FSC signals by regarding the fluiddynamics, optics and electronics of a flow cytometer.
> However, at the time of writing, FlowCyPy didn't yet support generating flourescence signals.

# cytodatagen

> Generate synthetic cytometry data for classification tasks


This package provides modular and highly configurable tools to generate synthetic flow cytometry/CyTOF data for classification tasks.
Supported formats are `fcs` and `h5ad`.

In general, for each subject, the generator samples cell type proportions from a Dirichlet distribution, and cells from cell type specific multivariate normals.
Then, the generator applies different *effects*, that affect either *cell type compositions* or *marker expression* within samples.

1. Composition Effects:
     - "switch": switches the proportion of some cell types within affected samples
2. Expression Effects:
     - "signal": changes the expression values of class specific markers within certain cell types
     - "batch": divides samples from each class in batches and applies a batch shift
     - "noise": applies Gaussian noise to each channel given by a SNR
     - "sinh": inverse to the popular arsinh transform
     - "exp": inverse to the logarithmic transform

## Installation

```sh
pip install -e .

# or install with extras for jupyter notebooks
pip install -e .[notebook]
```

## Usage

The package provides a CLI:

```sh
# display help message
python -m cytodatagen --help

# generate data from command line
python -m cytodatagen -o cytodata --format fcs
```

## Related

- [FlowCyPy](https://github.com/MartinPdeS/FlowCyPy)
- Cheung et al.: [*Systematic Design, Generation, and Application of Synthetic Datasets for Flow Cytometry*](https://doi.org/10.5731/pdajpst.2021.012659) (2022)

## Acknowledgements

Thanks to the FlowKit and anndata team.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
