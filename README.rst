.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/snnax.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/snnax
    .. image:: https://readthedocs.org/projects/snnax/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://snnax.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/snnax/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/snnax
    .. image:: https://img.shields.io/pypi/v/snnax.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/snnax/
    .. image:: https://img.shields.io/conda/vn/conda-forge/snnax.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/snnax
    .. image:: https://pepy.tech/badge/snnax/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/snnax
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/snnax

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====
SNNAX
=====


SNNAX is a lightweight library for implementing Spiking Neural Networks (SNN) is JAX. It leverages the excellent and intuitive [Equinox Library](https://docs.kidger.site/equinox/).
The core of SNNAX is a module that stacks layers of pre-defined or custom defined SNNs and Equinox neural network modules, and providing the functions to call them in a single *scan* loop. This mode of operation enables feedback loops across the layers of SNNs, while leveraging GPU acceleration as much as possible

A longer description of your project goes here...

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.3.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
