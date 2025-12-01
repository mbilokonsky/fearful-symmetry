"""
Discrete Calculus for Cellular Automata

This module implements a discrete calculus framework for analyzing
elementary cellular automata, including operations for identifying
local pockets of nonlinearity.
"""

import numpy as np
from typing import Optional


class CACalculus:
    """
    Discrete calculus operations for cellular automata analysis.

    This class provides tools for analyzing the behavior of elementary
    cellular automata through a calculus-based framework.
    """

    def __init__(self):
        """Initialize the CA Calculus framework."""
        pass


# Utility functions for CA calculus operations

def rule_to_table(rule: int) -> dict[tuple[int, int, int], int]:
    """
    Convert a rule number (0-255) to a lookup table.

    Args:
        rule: Wolfram rule number

    Returns:
        Dictionary mapping (left, center, right) tuples to output values
    """
    table = {}
    for i in range(8):
        output = (rule >> i) & 1
        left = (i >> 2) & 1
        center = (i >> 1) & 1
        right = i & 1
        table[(left, center, right)] = output
    return table


def table_to_rule(table: dict[tuple[int, int, int], int]) -> int:
    """
    Convert a lookup table back to a rule number.

    Args:
        table: Dictionary mapping neighborhoods to outputs

    Returns:
        Wolfram rule number (0-255)
    """
    rule = 0
    for i in range(8):
        left = (i >> 2) & 1
        center = (i >> 1) & 1
        right = i & 1
        if table.get((left, center, right), 0):
            rule |= (1 << i)
    return rule
