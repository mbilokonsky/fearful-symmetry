"""
Discrete Calculus for Cellular Automata

This module implements a discrete calculus framework for analyzing
elementary cellular automata, including a commutator for identifying
local pockets of nonlinearity.

Definitions:
    G - Configuration space
    R - Ruleset (applied to G)
    N - Neighborhood (used by R)
    S - Vector of states over time, where S_n gives rise to S_{n+1}

Core Operations:
    D(S) - Derivative: bitmask where cells are 1 iff they change from S_n to S_{n+1}
    I(S, S') - Integral: XOR operation, flips cells in S wherever S' is 1
    E(S) - Evolve: I(S, D(S)) - the step function decomposed
    C(S) - Commutator: E(D(S)) XOR D(E(S)) - reveals nonlinearity
"""

import numpy as np
from typing import Optional


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


def apply_rule(state: np.ndarray, rule: int) -> np.ndarray:
    """
    Apply a Wolfram rule to a state to produce the next state.

    Args:
        state: Current state (1D binary array)
        rule: Wolfram rule number (0-255)

    Returns:
        Next state after applying the rule
    """
    table = rule_to_table(rule)
    width = len(state)
    next_state = np.zeros_like(state)

    for i in range(width):
        left = state[(i - 1) % width]
        center = state[i]
        right = state[(i + 1) % width]
        next_state[i] = table[(left, center, right)]

    return next_state


class CACalculus:
    """
    Discrete calculus operations for cellular automata analysis.

    This class encapsulates the derivative, integral, evolve, and commutator
    operations for analyzing elementary cellular automata.

    Attributes:
        rule: The Wolfram rule number (0-255)
        table: The rule lookup table
    """

    def __init__(self, rule: int):
        """
        Initialize the CA Calculus framework with a specific rule.

        Args:
            rule: Wolfram rule number (0-255)
        """
        if not 0 <= rule <= 255:
            raise ValueError("Rule must be between 0 and 255")
        self.rule = rule
        self.table = rule_to_table(rule)

    def _apply_rule(self, state: np.ndarray) -> np.ndarray:
        """Apply the rule to produce the next state."""
        width = len(state)
        next_state = np.zeros_like(state)

        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            next_state[i] = self.table[(left, center, right)]

        return next_state

    def derivative(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the derivative D(S).

        The derivative is a bitmask where cells are 1 wherever the state
        will change under evolution. Defined directly in terms of R:

        D(S)[i] = R(left, center, right) XOR center

        This determines if applying the rule to each neighborhood produces
        a different value than the current cell, without computing the
        full next state.

        Args:
            state: Current state (1D binary array)

        Returns:
            Derivative bitmask (same shape as state)
        """
        width = len(state)
        derivative = np.zeros_like(state)

        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]

            # D(S)[i] = R(neighborhood) XOR center
            rule_output = self.table[(left, center, right)]
            derivative[i] = rule_output ^ center

        return derivative

    def integral(self, state: np.ndarray, derivative: np.ndarray) -> np.ndarray:
        """
        Compute the integral I(S, S').

        The integral flips cells in S wherever S' (the derivative) is 1.
        This is equivalent to XOR.

        Args:
            state: Current state (1D binary array)
            derivative: Derivative bitmask (1D binary array)

        Returns:
            New state with cells flipped where derivative is 1
        """
        return np.bitwise_xor(state, derivative).astype(np.uint8)

    def evolve(self, state: np.ndarray) -> np.ndarray:
        """
        Compute E(S) = I(S, D(S)).

        The evolve function is the step function decomposed through
        the derivative and integral operations.

        Args:
            state: Current state (1D binary array)

        Returns:
            Next state (equivalent to apply_rule)
        """
        d = self.derivative(state)
        return self.integral(state, d)

    def commutator(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the commutator C(S) = E(D(S)) XOR D(E(S)).

        The commutator reveals nonlinearity - cells are 1 where the
        order of operations between E (evolve) and D (derivative) matters.

        Args:
            state: Current state (1D binary array)

        Returns:
            Commutator bitmask showing positions of nonlinearity
        """
        # E(D(S)) - evolve the derivative
        d_s = self.derivative(state)
        e_d_s = self.evolve(d_s)

        # D(E(S)) - derivative of the evolved state
        e_s = self.evolve(state)
        d_e_s = self.derivative(e_s)

        # Commutator: where these differ
        return np.bitwise_xor(e_d_s, d_e_s).astype(np.uint8)

    def run_analysis(
        self,
        initial_state: np.ndarray,
        generations: int
    ) -> dict[str, np.ndarray]:
        """
        Run a complete analysis over multiple generations.

        Args:
            initial_state: Starting state (1D binary array)
            generations: Number of generations to simulate

        Returns:
            Dictionary containing:
                - 'states': State evolution history
                - 'derivatives': Derivative at each step
                - 'commutators': Commutator at each step
        """
        states = [initial_state.copy()]
        derivatives = []
        commutators = []

        current = initial_state.copy()
        for _ in range(generations):
            d = self.derivative(current)
            c = self.commutator(current)

            derivatives.append(d)
            commutators.append(c)

            current = self.evolve(current)
            states.append(current)

        return {
            'states': np.array(states),
            'derivatives': np.array(derivatives),
            'commutators': np.array(commutators)
        }

    def __repr__(self) -> str:
        return f"CACalculus(rule={self.rule})"
