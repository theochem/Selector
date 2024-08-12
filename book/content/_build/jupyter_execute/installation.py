#!/usr/bin/env python

# # Installation
#
# ## Stable releases
#
# <div class="alert alert-block alert-warning">
# <b>Warning:
# </b>
#
# We are preparing a 1.0 release. Until then, these instructions for installing a stable release will not work yet. If you enjoy living on the edge, try the development release as explained in the “Latest git revision” section below.
# </div>
#
# The following dependencies are required to run selector properly,
#
# * Python >= 3.6: http://www.python.org/
# * NumPy >= 1.21.5: http://www.numpy.org/
# * SciPy >= 1.5.0: http://www.scipy.org/
# * PyTest >= 5.3.4: https://docs.pytest.org/
# * PyTest-Cov >= 2.8.0: https://pypi.org/project/pytest-cov
#
# Normally, you don’t need to install these dependencies manually. They will be installed automatically when you follow the instructions below.
#
#
# ## Installation with Ana- or Miniconda:
#
# 1. To install selector using the conda package management system, install [miniconda](https://conda.io/miniconda.html) or [anaconda](https://www.anaconda.com/download) first, and then:
#
# ```
# # Activate your main conda environment if it is not loaded in your .bashrc.
# # E.g. run the following if you have miniconda installed in e.g. ~/miniconda3
# source ~/miniconda3/bin/activate
#
# # Create a horton3 conda environment. (optional, recommended)
# conda create -n horton3
# source activate horton3
#
# # Install the stable release.
# conda install -c theochem qc-selector
#
# # Unstable releases
# # (Only do this if you understand the implications.)
# # Install the testing release. (beta)
# conda install -c theochem/label/test qc-selector
# # Install the development release. (alpha)
# conda install -c theochem/label/dev qc-selector
# ```
#
# ## Installation with Pip
#
# 1. You can work in a [virtual environment](https://docs.python.org/3/tutorial/venv.html):
#
# ```
# # Create a virtual environment in ~/horton3
# # Feel free to change the path.
# python3 -m venv ~/horton3
#
# # Activate the virtual environemnt.
# source ~/horton3/bin/activate
#
# # Install the stable release in the venv horton3.
# pip3 install qc-selector
# # alternative: python3 -m pip install qc-selector
#
# # For developers, install a pre-release (alpha or beta).
# # (Only do this if you understand the implications.)
# pip3 install --pre qc-selector
# # alternative: python3 -m pip install --pre qc-selector
# ```
#
#
# 2. You can install into your `{$HOME}` directory without creating a virtual environment
#
# ```
# # Install the stable release in your home directory.
# pip3 install qc-selector --user
# # alternative: python3 -m pip install qc-selector --user
#
# # For developers, install a pre-release (alpha or beta).
# # (Only do this if you understand the implications.)
# pip3 install --pre qc-selector --user
# # alternative: python3 -m pip install --pre qc-selector --user
# ```
#
# This is by far the simplest method, ideal to get started, but you have only one home directory. If the installation breaks due to some experimentation, it is harder to make a clean start in comparison to the other options.
#
# In case the `pip3` executable is not found, pip may be installed in a directory which is not included in your `${PATH}` variable. This seems to be a common issue on macOS. A simple workaround is to replace `pip3` by `python3 -m pip`.
#
# In case Python and your operating system are up to date, you may also use `pip` instead of `pip3` or `python` instead of `python3`. The `3` is only used to avoid potential confusion with Python 2. Note that the `3` is only present in names of executables, not names of
# Python modules.
#
# ## Latest git revision
#
# This section shows how one can install the latest revision of selector from the git repository. This kind of installation comes with some risks (sudden API changes, bugs, …) and so be prepared to accept them when using the following installation instructions.
#
# There are two installation methods:
#
# 1. **Quick and dirty.** Of this method, there are four variants, depending on the correctness of your `PATH` variable and the presence of a virtual or conda environment. These different scenarios are explained in more detail in the previous section.
# ```
# # with env, correct PATH
# pip install git+https://github.com/theochem/Selector.git
# # with env, broken PATH
# python -m pip install git+https://github.com/theochem/Selector.git
# # without env, correct PATH
# pip install git+https://github.com/theochem/Selector.git --user
# # without env, broken PATH
# python -m pip install git+https://github.com/theochem/Selector.git --user
# ```
#
# 2. **Slow and Smart.** In addition to the four variations in the quick and dirty method, the slow and smart can be used with `pip` or just with `setup.py`. You also have the options to use SSH or HTTPS protocols to clone the git repository. Pick whichever works best for you.
# ```
# # A) Clone git repo with https OR ssh:
# # The second one only works if you have ssh set up for Github
# #  A1) https
# git clone https://github.com/theochem/Selector.git
# #  A2) ssh
# git clone git@github.com:theochem/Selector.git
# # B) Optionally write the version string
# pip install roberto  # or any of the three other ways of running pip, see above.
# rob write-version
# # C) Actual install, 6 different methods.
# #  C1) setup.py, with env
# python setup.py install
# #  C2) pip, with env, correct PATH
# pip install .
# #  C3) pip, with env, broken PATH
# python -m pip install .
# #  C4) setup.py, without env
# python setup.py install --user
# #  C5) pip, without env, correct PATH
# pip install . --user
# #  C6) pip, without env, broken PATH
# python -m pip install . --user
# ```
#
#
# ## Testing
#
# The tests are automatically run when we build packages with conda, but you may try them again on your own machine after installation.
#
# With Ana- or Miniconda:
# ```
# # Install pytest in your conda env.
# conda install pytest pytest-xdist
# # Then run the tests.
# pytest --pyargs selector -n auto
# ```
#
# With Pip:
# ```
# # Install pytest in your conda env ...
# pip install pytest pytest-xdist
# # .. and refresh the virtual environment.
# # This is a venv quirk. Without it, pytest may not find IOData.
# deactivate && source ~/selector/activate
#
# # Alternatively, install pytest in your home directory.
# pip install pytest pytest-xdist --user
#
# # Finally, run the tests.
# pytest --pyargs selector -n auto
# ```
