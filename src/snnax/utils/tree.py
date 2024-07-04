from typing import Callable,Type

import jax

from jaxtyping import PyTree
from equinox import tree_at
from jax.tree_util import tree_leaves


def apply_to_tree_leaf(pytree: PyTree, 
                        identifier: str, 
                        replace_fn: Callable) -> PyTree:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`
    
    **Arguments**
    - `pytree`: The pytree where we want to modify the leaves.
    - `identifier`: A string used to identify the name/field of the leaves.
    - `replace_fn`: Callable which is applied to the leaf values.
    """
    _has_identifier = lambda leaf: hasattr(leaf, identifier) and getattr(leaf, identifier) is not None
    def _identifier(pytree):
        return tuple(
            getattr(leaf, identifier)
            for leaf in tree_leaves(pytree, is_leaf=_has_identifier)
            if _has_identifier(leaf)
        )

    return tree_at(_identifier, pytree, replace_fn=replace_fn)

def apply_to_tree_leaf_bytype(pytree: PyTree, 
                        typ: Type, 
                        identifier: str, 
                        replace_fn: Callable) -> PyTree:
    """
    Apply a function all leaves in the given pytree with the given identifier.
    To simply replace values, use `replace_fn=lambda _: value`
    
    **Arguments**
    - `pytree`: The pytree where we want to modify the leaves.
    - `identifier`: A string used to identify the name/field of the leaves.
    - `replace_fn`: Callable which is applied to the leaf values.
    """
    _has_identifier = lambda leaf: (type(leaf) == typ) and hasattr(leaf, identifier)
    def _identifier(pytree):
        return tuple( getattr(leaf,identifier) for leaf in jax.tree_util.tree_leaves(pytree, is_leaf=_has_identifier) if _has_identifier(leaf) and getattr(leaf, identifier) is not None )

    return eqx.tree.tree_at(_identifier, pytree, replace_fn=replace_fn)
