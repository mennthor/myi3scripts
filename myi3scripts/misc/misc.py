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


def fill_dict_defaults(d, required_keys=None, opt_keys=None, noleft=True):
    """
    Populate dictionary with data from a given dict ``d``, and check if ``d``
    has required and optional keys. Set optionals with default if not present.

    If input ``d`` is None and ``required_keys`` is empty, just return
    ``opt_keys``.

    Parameters
    ----------
    d : dict or None
        Input dictionary containing the data to be checked. If is ``None``, then
        a copy of ``opt_keys`` is returned. If ``opt_keys`` is ``None``, a
        ``TypeError`` is raised. If ``d``is ``None`` and ``required_keys`` is
        not, then a ``ValueError`` israised.
    required_keys : list or None, optional
        Keys that must be present  and set in ``d``. (default: None)
    opt_keys : dict or None, optional
        Keys that are optional. ``opt_keys`` provides optional keys and default
        values ``d`` is filled with if not present in ``d``. (default: None)
    noleft : bool, optional
        If True, raises a ``KeyError``, when ``d`` contains etxra keys, other
        than those given in ``required_keys`` and ``opt_keys``. (default: True)

    Returns
    -------
    out : dict
        Contains all required and optional keys, using default values, where
        optional keys were missing. If ``d`` was None, a copy of ``opt_keys`` is
        returned, if ``opt_keys`` was not ``None``.
    """
    if required_keys is None:
        required_keys = []
    if opt_keys is None:
        opt_keys = {}
    if d is None:
        if not required_keys:
            if opt_keys is None:
                raise TypeError("`d` and òpt_keys` are both None.")
            return opt_keys.copy()
        else:
            raise ValueError("`d` is None, but `required_keys` is not empty.")

    d = d.copy()
    out = {}
    # Set required keys
    for key in required_keys:
        if key in d:
            out[key] = d.pop(key)
        else:
            raise KeyError("Dict is missing required key '{}'.".format(key))
    # Set optional values, if key not given
    for key, val in opt_keys.items():
        out[key] = d.pop(key, val)
    # Complain when extra keys are left and noleft is True
    if d and noleft:
        raise KeyError("Leftover keys ['{}'].".format(
            "', '".join(list(d.keys()))))
    return out
