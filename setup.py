# -*- coding: utf-8 -*-
# The DiverseSelector library provides a set of tools to select molecule
# subset with maximum molecular diversity.
#
# Copyright (C) 2022 The QC-Devs Community
#
# This file is part of DiverseSelector.
#
# DiverseSelector is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# DiverseSelector is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Setup and Install Script."""

import sys

from setuptools import setup
import versioneer

short_description = "Molecule selection with maximum diversity".split("\n")[0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except ValueError:
    long_description = short_description


setup(
    name="selector",
    author="QC-Devs Community",
    author_email="qcdevs@gmail.com",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="GNU (Version 3)",
    package_dir={"selector": "selector"},
    packages=["selector", "selector.methods", "selector.tests", "selector.methods.tests"],
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[
        "numpy>=1.21.2",
        "scipy==1.11.1",
        "pytest>=6.2.4",
        "scikit-learn",
        "bitarray",
    ]
    + pytest_runner,
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url="https://github.com/theochem/DiverseSelector",  # Website
    install_requires=[
        "numpy>=1.21.2",
        "scipy==1.11.1",
        "pytest>=6.2.4",
        "scikit-learn",
        "bitarray",
    ],
    # platforms=["Linux",
    #            "Mac OS-X",
    #            "Unix",
    #            "Windows"],
    # Python version restrictions
    python_requires=">=3.7",
    # Manual control if final package is compressible or not, set False to prevent the .egg
    # from being made
    # zip_safe=False,
    # todo: add classifiers
)
