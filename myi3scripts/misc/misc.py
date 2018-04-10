# coding: utf8

from __future__ import absolute_import

import numpy as np


def arr2str(arr, sep=", ", fmt="{}"):
    """
    Make a string from a list seperated by ``sep`` and each item formatted
    with ``fmt``.
    """
    return sep.join([fmt.format(v) for v in arr])


def serialize_ndarrays(d):
    """
    Recursively traverse through iterable object ``d`` and convert all occuring
    ndarrays to lists to make it JSON serializable.

    Note: Works for 1D dicts with ndarrays at first level. Certainly not tested
          and meant to work for all use cases.
          Made with code from: http://code.activestate.com/recipes/577504/

    Parameters
    ----------
    d : iterable
        Can be dict, list, set, tuple or frozenset.

    Returns
    -------
    d : iterable
        Same as input, but all ndarrays replaced by lists.
    """
    def dict_handler(d):
        return d.items()

    handlers = {list: enumerate, tuple: enumerate,
                set: enumerate, frozenset: enumerate,
                dict: dict_handler}

    def serialize(o):
        for typ, handler in handlers.items():
            if isinstance(o, typ):
                for key, val in handler(o):
                    if isinstance(val, np.ndarray):
                        o[key] = val.tolist()
                    else:
                        o[key] = serialize_ndarrays(o[key])
        return o

    return serialize(d)
