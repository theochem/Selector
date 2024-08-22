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

Running Web Interface Locally Using Docker
------------

### Install Docker
  - Visit this [site](https://docs.docker.com/engine/install/) to install Docker Engine
  - Open the installed Docker App once to complete the setup

### Build the Docker Image
- Clone the repository
```bash
git clone https://github.com/theochem/Selector.git
```

- Change to the Selector directory
```bash
cd Selector
```

- Change to the webapp branch
```bash
git checkout webapp
```

- Build the Docker image
```bash
docker build -t selector-webapp .
```

- Run the Webapp inside Docker Container
```bash
docker run -p 8501:8501 selector-webapp
```
