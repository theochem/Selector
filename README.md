# DiverseSelector

[![This project supports Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub Actions CI Tox Status](https://github.com/theochem/DiverseSelector/actions/workflows/ci_tox.yml/badge.svg?branch=main)](https://github.com/theochem/DiverseSelector/actions/workflows/ci_tox.yml)
[![codecov](https://codecov.io/gh/theochem/DiverseSelector/branch/main/graph/badge.svg?token=0UJixrJfNJ)](https://codecov.io/gh/theochem/DiverseSelector)

The DiverseSelector library provides methods for selecting a diverse subset of a (molecular) dataset.


Citation
--------

Please use the following citation in any publication using DiverseSelector library:

```md
@article{
}
```

Dependencies
------------

The following dependencies are required to run DiverseSelector properly,

* Python >= 3.6: http://www.python.org/
* NumPy >= 1.21.5: http://www.numpy.org/
* SciPy >= 1.5.0: http://www.scipy.org/
* PyTest >= 5.3.4: https://docs.pytest.org/
* PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov/


Installation
------------

To install DiverseSelector using the conda package management system, install
[miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download)
first, and then:

```bash
    # Create and activate myenv conda environment (optional, but recommended)
    conda create -n myenv python=3.6
    conda activate myenv

    # Install the stable release.
    conda install -c theochem qc-selector
```

To install DiverseSelector with pip, you may want to create a
[virtual environment](https://docs.python.org/3/tutorial/venv.html), and then:


```bash
    # Install the stable release.
    pip install qc-selector
```

See https://selector.qcdevs.org for full details.
