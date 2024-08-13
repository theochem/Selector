# The `selector` Webapp

[![This project supports Python 3.7+](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org/downloads)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![GitHub Actions CI Tox Status](https://github.com/theochem/Selector/actions/workflows/ci_tox.yml/badge.svg?branch=main)](https://github.com/theochem/Selector/actions/workflows/ci_tox.yml)
[![codecov](https://codecov.io/gh/theochem/Selector/graph/badge.svg?token=0UJixrJfNJ)](https://codecov.io/gh/theochem/Selector)

The `selector` library provides methods for selecting a diverse subset of a (molecular) dataset.

🤗 [Selector on HuggingFace](https://huggingface.co/spaces/QCDevs/selector)

Citation
--------

Please use the following citation in any publication using the `selector` library:

```md
@article{
}
```


Installation
------------

To install `selector` using the conda package management system, install
[miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download)
first, and then:

```bash
    # Create and activate qcdevs conda environment (optional, but recommended)
    conda create -n qcdevs python=3.10
    conda activate qcdevs

    # Install the stable release
    # current conda release is not ready yet
    # conda install -c theochem qc-selector

    # install the development version
    pip install git+https://github.com/theochem/Selector.git
```

To install `selector` with `pip`, you may want to create a
[virtual environment](https://docs.python.org/3/tutorial/venv.html), and then:


```bash
    # Install the stable release.
    pip install qc-selector
```

See https://selector.qcdevs.org for full details.

Running Web Interface Locally
------------

After installing the package, you can run the interface locally by running the following command:

```bash
    streamlit run streamlit_app/app.py
```
