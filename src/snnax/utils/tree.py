from typing import Callable,Type

from chex import PyTreeDef
from equinox import tree_at
from jax.tree_util import tree_leaves


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
    _has_identifier = lambda leaf: hasattr(leaf, identifier) \
                                    and getattr(leaf, identifier) is not None
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
    _has_identifier = lambda leaf: (type(leaf) == typ) \
                                    and hasattr(leaf, identifier) \
                                    and getattr(leaf, identifier) is not None
    def _identifier(pytree):
        return tuple(
            getattr(leaf, identifier) 
            for leaf in tree_leaves(pytree, is_leaf=_has_identifier) 
            if _has_identifier(leaf) and getattr(leaf, identifier) is not None 
        )

    return tree_at(_identifier, pytree, replace_fn=replace_fn)
