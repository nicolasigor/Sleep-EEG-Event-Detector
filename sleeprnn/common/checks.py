"""Module that defines common errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def check_valid_range(value, name, valid_range):
    """Raises a ValueError exception if value not in valid_range"""
    if value > valid_range[1] or value < valid_range[0]:
        msg = "Expected range %s for %s, but %s was provided." % (
            valid_range,
            name,
            value,
        )
        raise ValueError(msg)


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list"""
    if value not in valid_list:
        msg = "Expected %s for %s, but %s was provided." % (valid_list, name, value)
        raise ValueError(msg)


def check_directory(path_dir):
    """Raises FileNotFoundError exception if directory doesn't exists"""
    if not os.path.isdir(path_dir):
        raise FileNotFoundError("Directory not found: %s" % path_dir)


def ensure_directory(path_dir):
    """If directory doesn't exists, is created."""
    os.makedirs(path_dir, exist_ok=True)
