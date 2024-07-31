from typing import Any, Callable,Type
from functools import partial

from chex import PyTreeDef
from equinox import tree_at, is_inexact_array
from jax.tree_util import tree_leaves


NEURON_PARAMS = [
    "threshold", 
    "leak",
    "decay_constant", 
    "decay_constants", 
    "reset_val"
]


is_identifier = lambda name, leaf: hasattr(leaf, name)


has_identifier = lambda name, leaf: hasattr(leaf, name) \
                                and getattr(leaf, name) is not None


def is_threshold(element: Any) -> bool:
    """
    Check if the given leaf is a parameter named `threshold`.
    
    Arguments:
        `element`: The leaf we want to check.
    
    Returns:
        `bool`: True if the leaf is a threshold, False otherwise.
    """
    return is_identifier("threshold", element) and is_inexact_array(element)


def is_decay_constant(element: Any) -> bool:
    """
    Check if the given leaf is a parameter named `decay_constant`.
    
    Arguments:
        `element`: The leaf we want to check.
    
    Returns:
        `bool`: True if the leaf is a decay constant, False otherwise.
    """
    return is_identifier("decay_constant", element) and is_inexact_array(element)


def is_decay_constants(element: Any) -> bool:
    """
    Check if the given leaf is a parameter named `decay_constants`.
    
    Arguments:
        `element`: The leaf we want to check.
    
    Returns:
        `bool`: True if the leaf is a decay constants, False otherwise.
    """
    return is_identifier("decay_constants", element) and is_inexact_array(element)


def is_reset_val(element: Any) -> bool: 
    """
    Check if the given leaf is a parameter named `reset_val`.
    
    Arguments:
        `element`: The leaf we want to check.
    
    Returns:
        `bool`: True if the leaf is a reset value, False otherwise.
    """
    return is_identifier("reset_val", element) and is_inexact_array(element)


def is_neuron_param(element: Any) -> bool:
    return any(is_identifier(element, param_name) 
                for param_name in NEURON_PARAMS) and is_inexact_array(element)


def apply_to_tree_leaf(pytree: PyTreeDef, 
                        identifier: str, 
                        replace_fn: Callable) -> PyTreeDef:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`
    
    Arguments:
        `pytree`: The pytree where we want to modify the leaves.
        `identifier`: A string used to identify the name/field of the leaves.
        `replace_fn`: Callable which is applied to the leaf values.

    Returns:
        The modified PyTree.
    """
    _has_identifier = partial(has_identifier, identifier)
    def _identifier(pytree):
        return tuple(
            getattr(leaf, identifier)
            for leaf in tree_leaves(pytree, is_leaf=_has_identifier)
            if _has_identifier(leaf)
        )

    return tree_at(_identifier, pytree, replace_fn=replace_fn)


def apply_to_tree_leaf_bytype(pytree: PyTreeDef, 
                            typ: Type, 
                            identifier: str, 
                            replace_fn: Callable) -> PyTreeDef:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`. This function
    is more specific than `apply_to_tree_leaf` as it allows to specify the type
    of the leaf we want to modify.
    
    Arguments:
        `pytree`: The pytree where we want to modify the leaves.
        `identifier`: A string used to identify the name/field of the leaves.
        `replace_fn`: Callable which is applied to the leaf values.

    Returns:
        The modified PyTree.
    """
    _has_identifier = lambda leaf: (type(leaf) == typ) and has_identifier(identifier, leaf)
    def _identifier(pytree):
        return tuple(
            getattr(leaf, identifier) 
            for leaf in tree_leaves(pytree, is_leaf=_has_identifier) 
            if _has_identifier(leaf, identifier)
        )

    return tree_at(_identifier, pytree, replace_fn=replace_fn)
