<!-- #region -->
# Welcome to QC-Selector's Documentation!

[Selector](https://github.com/theochem/Selector) is a free, open-source, and cross-platform Python library designed to help you effortlessly identify the most diverse subset of molecules from your dataset. Please use the following citation in any publication using the `selector` library:

> **Selector: A General Python Library for Diverse Subset Selection.**
> Fanwang Meng, Marco Martínez González, Valerii Chuiko, Alireza Tehrani,
> Abdul Rahman Al Nabulsi, Abigail Broscius, Hasan Khaleel, Kenneth López-Pérez,
> Ramón Alain Miranda-Quintana, Paul W. Ayers, Farnaz Heidar-Zadeh.
> *J. Chem. Inf. Model. 66 (3), 1275–1285 (2026).*
> https://doi.org/10.1021/acs.jcim.5c01499

```bibtex
@article{selector_library,
    author = {Meng, Fanwang and Martínez González, Marco and Chuiko, Valerii and Tehrani, Alireza and Al Nabulsi, Abdul Rahman and Broscius, Abigail and Khaleel, Hasan and López-Pérez, Kenneth and Miranda-Quintana, Ramón Alain and Ayers, Paul W. and Heidar-Zadeh, Farnaz},
    title = {Selector: A General Python Library for Diverse Subset Selection},
    journal = {Journal of Chemical Information and Modeling},
    volume = {66},
    number = {3},
    pages = {1275--1285},
    year = {2026},
    doi = {10.1021/acs.jcim.5c01499},
    url = {https://doi.org/10.1021/acs.jcim.5c01499},
}
```

The Selector source code is hosted on [GitHub](https://github.com/theochem/Selector) and is released under the [GNU General Public License v3.0](https://github.com/theochem/Selector/blob/main/LICENSE). We welcome any contributions to the Selector library in accordance with our Code of Conduct; please see our [Contributing Guidelines](https://qcdevs.org/guidelines/QCDevsCodeOfConduct/). Please report any issues you encounter while using Selector library on [GitHub Issues](https://github.com/theochem/Selector/issues). For further information and inquiries please contact us at qcdevs@gmail.com.


## Why QC-Selector?

Selecting diverse and representative subsets is crucial for the data-driven models and machine
learning applications in many science and engineering disciplines, especially for molecular design
and drug discovery. Motivated by this, we develop the Selector package, a free and open-source Python library for selecting diverse subsets.

The Selector library implements a range of existing algorithms for subset sampling based on the
distance between and similarity of samples, as well as tools based on spatial partitioning. In
addition, it includes seven diversity measures for quantifying the diversity of a given set. We also
implemented various mathematical formulations to convert similarities into dissimilarities.


## Web Server

We have a web server for the `selector` library at https://huggingface.co/spaces/QCDevs/Selector.
For small and prototype datasets, you can use the web server to select a diverse subset of your
dataset and compute the diversity metrics, where you can download the selected subset and the
computed diversity metrics.

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
