#!/usr/bin/env python
# coding: utf-8

# # Installation & Testing
# 
# 1. To install DiverseSelector using the conda package management system, install [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download) first, and then:
# 
# ```
#     # Create and activate myenv conda environment (optional, but recommended)
#     conda create -n myenv python=3.6
#     conda activate myenv
# 
#     # Install the stable release.
#     conda install -c theochem qc-
# ```
# 2. To install DiverseSelector with pip, you may want to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html), and then:
# 
# ```
#     # Install the stable release.
#     pip install qc-
# ```
# 
# # Dependencies
# 
# The following dependencies are required to run DiverseSelector properly,
# 
# * Python >= 3.6: http://www.python.org/ 
# * NumPy >= 1.21.5: http://www.numpy.org/
# * SciPy >= 1.5.0: http://www.scipy.org/
# * PyTest >= 5.3.4: https://docs.pytest.org/
# * PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov/
# 
# # Testing
# 
# The tests are automatically run when we build packages with conda, but you may try them again on your own machine after installation.
# 
# With Ana- or Miniconda:
# ```
# # Install pytest in your conda env.
# conda install pytest pytest-xdist
# # Then run the tests.
# pytest --pyargs diverseselector -n auto
# ```
# 
# With Pip:
# ```
# # Install pytest in your conda env ...
# pip install pytest pytest-xdist
# # .. and refresh the virtual environment.
# # This is a venv quirk. Without it, pytest may not find IOData.
# deactivate && source ~/diverseselector/activate
# 
# # Alternatively, install pytest in your home directory.
# pip install pytest pytest-xdist --user
# 
# # Finally, run the tests.
# pytest --pyargs diverseselector -n auto
# ```
