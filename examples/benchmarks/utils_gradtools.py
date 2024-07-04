#!/bin/python
#-----------------------------------------------------------------------------
# File Name : eqx_grads.py
# Author: Emre Neftci
#
# Creation Date : Thu 09 Mar 2023 09:54:17 AM CET
# Last Modified : Wed 24 Apr 2024 10:17:28 AM CEST
#
# Copyright : (c) Emre Neftci, PGI-15 Forschungszentrum Juelich
# Licence : GPLv2
#----------------------------------------------------------------------------- 

import functools as ft
import types
import typing
import warnings
from typing import Any, Callable, Dict

import jax
import jax.interpreters.ad as ad
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from equinox._filters import (
    combine,
    is_array,
    is_inexact_array,
    partition,
)
from equinox._module import Module, module_update_wrapper, Static, static_field
from equinox._custom_types import sentinel

class _JacRevWrapper(Module):
    _fun: Callable
    _has_aux: bool
    _gradkwargs: Dict[str, Any]

    def __call__(__self, __x, *args, **kwargs):
        @ft.partial(jax.jacrev, argnums=0, has_aux=__self._has_aux, **__self._gradkwargs)
        def fun_jacrev(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            return __self._fun(_x, *_args, **_kwargs)

        diff_x, nondiff_x = partition(__x, is_inexact_array)
        return fun_jacrev(diff_x, nondiff_x, *args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return jtu.Partial(self, instance)

def filter_jacrev(
    fun: Callable = sentinel,
    *,
    has_aux: bool = False,
    **gradkwargs,
) -> Callable:
    """As [`equinox.filter_grad`][], except that it is `jax.jacrev` that is
    wrapped.
    """
    if fun is sentinel:
        return ft.partial(filter_jacrev, has_aux=has_aux, **gradkwargs)
    
    return module_update_wrapper(_JacRevWrapper(fun, has_aux, gradkwargs), fun)
    




