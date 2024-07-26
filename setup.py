"""
    Setup file for snnax.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.3.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup

if __name__ == "__main__":
    try:
        setup(
            name="snnax",
            version="0.0.1",
            description="A library for spiking neural networks in JAX.",
            author="Jamie Lohoff, Jan Finkbeiner, Emre Neftci",
            author_email="jamie.lohoff@gmail.com",
            packages=["snnax"],
            use_scm_version={"version_scheme": "no-guess-dev"},
            install_requires=["jax", "equinox", "chex"]
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
