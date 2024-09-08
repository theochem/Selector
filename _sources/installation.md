# Installation

## Stable Releases

<div class="alert alert-block alert-warning">
<b>Warning:
</b>
We are preparing a 1.0 release. Until then, we can install the beta relase with PyPI,
https://pypi.org/project/qc-selector/.
</div>

The following dependencies are required to run selector properly,

* Python >= 3.9: http://www.python.org/
* NumPy >= 1.21.2: http://www.numpy.org/
* SciPy >= 1.11.1: http://www.scipy.org/
* bitarray >= 2.5.1: https://pypi.org/project/bitarray/

Normally, you don’t need to install these dependencies manually. They will be installed automatically when you follow the instructions below.

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

## Prebuilt Wheels

To download the prebuilt wheel files, visit the [PyPI page](https://pypi.org/project/qc-selector/)
and [GitHub Releases](https://github.com/theochem/Selector/releases).

```bash
    # install the prebuilt wheel file
    pip install qc_selector-0.0.2b10-py3-none-any.whl
```

## Development Version

In addition, we can install the latest development version from the GitHub repository as follows:

```bash
    # install the latest development version
    pip install git+https://github.com/theochem/Selector.git

```
