# The `selector` Library

[![This project supports Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub Actions CI Tox Status](https://github.com/theochem/Selector/actions/workflows/ci_tox.yml/badge.svg?branch=main)](https://github.com/theochem/Selector/actions/workflows/ci_tox.yml)
[![codecov](https://codecov.io/gh/theochem/Selector/graph/badge.svg?token=0UJixrJfNJ)](https://codecov.io/gh/theochem/Selector)

The `selector` library provides methods for selecting a diverse subset of a (molecular) dataset.

## Citation

Please use the following citation in any publication using the `selector` library:

```md
@article{
    TO BE ADDED LATER
}
```

## Installation

It is recommended to install `selector` within a virtual environment. To create a virtual
environment, we can use the `venv` module (Python 3.3+,
https://docs.python.org/3/tutorial/venv.html), `miniconda` (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), or
`pipenv` (https://pipenv.pypa.io/en/latest/). We use `miniconda` in the following example to create a virtual environment:

```bash
    # Create and activate qcdevs conda environment (optional, but recommended)
    conda create -n qcdevs python=3.11
    conda activate qcdevs

```

To install `selector` with `pip`, we can install the latest stable release from the Python Package Index (PyPI) as follows:

```bash
    # Install the stable release.
    pip install qc-selector
```

To download the prebuilt wheel files, visit the [PyPI page](https://pypi.org/project/qc-selector/)
and [GitHub releases](https://github.com/theochem/Selector/tags).

In addition, we can install the latest development version from the GitHub repository as follows:

```bash
    # install the latest development version
    pip install git+https://github.com/theochem/Selector.git

```

## More

See https://selector.qcdevs.org for full details.
