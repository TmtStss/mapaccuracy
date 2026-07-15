"""Internal utility functions for mapaccuracy package."""

import numpy as np


def _check_classes(x, *args):
    """
    Check if arguments passed to functions comply with their expected classes.

    Parameters
    ----------
    x : list or str
        Expected class(es) as strings
    *args
        Arguments to be checked

    Raises
    ------
    ValueError
        If the arguments don't match the expected classes
    """
    classes_expected = x if isinstance(x, list) else [x]
    all_classes = ["str", "int", "float", "list", "dict", "tuple", "bool"]

    if not all(c in all_classes for c in classes_expected):
        raise ValueError(
            f"Argument 'classes' includes one or more classes not recognized. "
            f"Use only the following: {', '.join(all_classes)}"
        )


def _check_labels(x, *args):
    """
    Check if character arguments comply with their expected content.

    Parameters
    ----------
    x : list or set
        Expected labels
    *args
        Arguments to be checked

    Raises
    ------
    ValueError
        If arguments include labels not in x
    """
    x_set = set(x) if isinstance(x, (list, tuple)) else {x}

    for arg in args:
        if isinstance(arg, np.ndarray):
            arg_set = set(arg.tolist())
        elif isinstance(arg, (list, tuple, set)):
            arg_set = set(arg)
        else:
            arg_set = {arg}

        if not arg_set.issubset(x_set):
            missing = arg_set - x_set
            raise ValueError(
                f"Arguments should include only: {', '.join(map(str, sorted(x_set)))}"
            )


def _check_length(*args):
    """
    Check if the length of objects match.

    Parameters
    ----------
    *args
        Arguments to be checked

    Raises
    ------
    ValueError
        If arguments have different lengths
    """
    if len(args) > 1:
        lengths = [len(arg) if hasattr(arg, "__len__") else 1 for arg in args]
        if len(set(lengths)) > 1:
            raise ValueError("Check the length of class labels.")
