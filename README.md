<div style="text-align:center">
  <!-- <h1 style="margin-right: 20px;">The Selector Library</h1> -->
  <img src="https://github.com/theochem/Selector/blob/main/book/content/selector_logo.png?raw=true" alt="Logo" style="width: 50%">
</div>

[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![CI Tox](https://github.com/theochem/Selector/actions/workflows/ci_tox.yaml/badge.svg?branch=main)](https://github.com/theochem/Selector/actions/workflows/ci_tox.yaml)
[![codecov](https://codecov.io/gh/theochem/Selector/graph/badge.svg?token=0UJixrJfNJ)](https://codecov.io/gh/theochem/Selector)

The `Selector` library provides methods for selecting a diverse subset of a (molecular) dataset.

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
`pipenv` (https://pipenv.pypa.io/en/latest/).

### Installing from PyPI

To install `selector` with `pip`, we can install the latest stable release from the Python Package Index (PyPI) as follows:

```bash
    # install the stable release.
    pip install qc-selector
```

### Installing from The Prebuild Wheel Files

To download the prebuilt wheel files, visit the [PyPI page](https://pypi.org/project/qc-selector/)
and [GitHub releases](https://github.com/theochem/Selector/tags).

```bash
    # download the wheel file first to your local machine
    # then install the wheel file
    pip install file_path/qc_selector-0.0.2b12-py3-none-any.whl
```

### Installing from the Source Code

In addition, we can install the latest development version from the GitHub repository as follows:

```bash
    # install the latest development version
    pip install git+https://github.com/theochem/Selector.git
```

We can also clone the repository to access the latest development version, test it and install it as follows:

```bash
    # clone the repository
    git clone git@github.com:theochem/Selector.git

    # change into the working directory
    cd Selector
    # run the tests
    python -m pytest .

    # install the package
    pip install .

```

## More

See https://selector.qcdevs.org for full details.
