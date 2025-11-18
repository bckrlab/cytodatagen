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

This package provides tools to generate synthetic flow cytometry/CyTOF data for classification tasks.
Supported formats are `fcs` and `h5ad`.

The synthetic data generated with `cytodatagen` assumes that subjects differ either in:

1. distribution shifts of populations
     - example: T-cells have a higher marker value in postive subjects
2. cell type composition
     - example: positive subjects have a higher T-cell count

![pairplot](https://raw.githubusercontent.com/bckrlab/cytodatagen/refs/heads/main/docs/figures/pairplot.png)

![t-SNE](https://raw.githubusercontent.com/bckrlab/cytodatagen/refs/heads/main/docs/figures/tsne.png)

## Installation

```sh
# install via pip
pip install cytodatagen

# or install from source
pip install -e .

# install with extras for jupyter notebooks
pip install -e .[notebook]
```

## Usage

The package provides a CLI:

```sh
# display help message
python -m cytodatagen.cli.subjects --help

# generate subjects.json configuration file and adjust it later
python -m cytodatagen.cli.subjects config/cli/subjects/config.json -o artifacts/subjects.json --seed 19

# display help message
python -m cytodatagen.cli.data --help

# generate data from command line
python -m cytodatagen.cli.data -o artifacts/cytodata --format fcs

python -m cytodatagen.cli.data -o artifacts/demo --format h5ad --subjects config/cli/data/subjects.json --transforms config/cli/data/transforms.json --seed 19
```

Look at the scripts `scripts/make_subjects.sh` and `scripts/make_data.sh` for more examples.

## Related

- [FlowCyPy](https://github.com/MartinPdeS/FlowCyPy)
- Yang et al.: [*Cytomulate: accurate and efficient simulation of CyTOF data*](https://doi.org/10.1186/s13059-023-03099-1) (2023)
- Cheung et al.: [*Systematic Design, Generation, and Application of Synthetic Datasets for Flow Cytometry*](https://doi.org/10.5731/pdajpst.2021.012659) (2022)

## Acknowledgements

Thanks to the FlowKit and anndata team.

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
