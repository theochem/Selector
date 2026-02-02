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
@article{selector_library,
author = {Meng, Fanwang and Martínez González, Marco and Chuiko, Valerii and Tehrani, Alireza and Al Nabulsi, Abdul Rahman and Broscius, Abigail and Khaleel, Hasan and López-P{\'e}rez, Kenneth and Miranda-Quintana, Ramón Alain and Ayers, Paul W. and Heidar-Zadeh, Farnaz},
title = {Selector: A General Python Library for Diverse Subset Selection},
journal = {Journal of Chemical Information and Modeling},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1021/acs.jcim.5c01499},
note ={PMID: 41591801},
URL = {https://doi.org/10.1021/acs.jcim.5c01499}
}
```

## Web Server

We have a web server for the `selector` library at https://huggingface.co/spaces/QCDevs/Selector.
For small and prototype datasets, you can use the web server to select a diverse subset of your
dataset and compute the diversity metrics, where you can download the selected subset and the
computed diversity metrics.

## Installation

It is recommended to install `selector` within a virtual environment. To create a virtual
environment, we can use the `venv` module (Python 3.9+,
https://docs.python.org/3/tutorial/venv.html), `miniconda` (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), or
`pipenv` (https://pipenv.pypa.io/en/latest/).

### Installing from PyPI

To install `selector` with `pip`, we can install the latest stable release from the Python Package Index (PyPI) as follows:

```bash
# install the stable release
pip install qc-selector
```

### Installing from The Prebuild Wheel Files

To download the prebuilt wheel files, visit the [PyPI page](https://pypi.org/project/qc-selector/)
and [GitHub releases](https://github.com/theochem/Selector/tags).

```bash
# download the wheel file first to your local machine
# then install the wheel file
pip install file_path/qc_selector-0.1.0-py3-none-any.whl

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
