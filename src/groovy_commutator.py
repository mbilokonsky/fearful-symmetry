"""
Groovy Commutator - A Generalized Commutator for Structure Detection

The groovy commutator K is a full generalization of the commutator C from
the discrete CA calculus framework. It provides a powerful mechanism for
identifying underlying structure within non-noisy, non-predictable patterns.

Formula:
    K(psi) = Delta(psi + Delta(psi)) - (Delta(psi) + Delta(Delta(psi)))

Where:
    - Delta is the discrete derivative (difference operator)
    - + is addition (XOR for binary, regular addition for integers)
    - - is subtraction (XOR for binary, regular subtraction for integers)

This can be expanded as:
    K(psi) = Delta(psi + Delta(psi)) - Delta(psi) - Delta(Delta(psi))

The groovy commutator measures the difference between:
    - The derivative of the "evolved" signal (psi + Delta(psi))
    - The sum of the first and second derivatives

When K = 0, the system exhibits a form of linearity where these quantities
balance. Non-zero K reveals structural irregularities and hidden patterns.
"""

import numpy as np
from typing import Union, Optional, Callable


def delta_binary(psi: np.ndarray, rule: int = None) -> np.ndarray:
    """
    Compute the discrete derivative for binary (CA) states.

    For CA states, this uses XOR-based differencing. If a rule is provided,
    it uses the CA derivative (R(N) XOR center). Otherwise, it uses simple
    neighbor differencing.

    Args:
        psi: Binary state array
        rule: Optional Wolfram rule number for CA-based derivative

    Returns:
        Binary derivative array
    """
    if rule is not None:
        # Import here to avoid circular dependency
        from .ca_calculus import CACalculus
        calc = CACalculus(rule)
        return calc.derivative(psi)

    # Simple binary difference: XOR with shifted self
    width = len(psi)
    result = np.zeros_like(psi)
    for i in range(width):
        # Difference with right neighbor (circular)
        result[i] = psi[i] ^ psi[(i + 1) % width]
    return result


def delta_integer(psi: np.ndarray) -> np.ndarray:
    """
    Compute the discrete derivative for integer sequences.

    Uses the standard forward difference operator:
        Delta(f)[i] = f[i+1] - f[i]

    Args:
        psi: Integer sequence array

    Returns:
        Difference array (length n-1 for input of length n)
    """
    return np.diff(psi)


def delta_integer_padded(psi: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Compute the discrete derivative for integer sequences with padding.

    Returns an array of the same length as input by appending a pad value.

    Args:
        psi: Integer sequence array
        pad_value: Value to append (default 0)

    Returns:
        Difference array (same length as input)
    """
    diff = np.diff(psi)
    return np.append(diff, pad_value)


class GroovyCommutator:
    """
    Computes the groovy commutator K for structure detection.

    The groovy commutator generalizes the CA commutator C to work with
    any discrete sequence, providing a tool for detecting hidden structure
    in patterns that appear random or chaotic.

    Attributes:
        mode: 'binary' for CA/XOR operations, 'integer' for standard arithmetic
        rule: Optional Wolfram rule for binary mode CA derivative
    """

    def __init__(self, mode: str = 'integer', rule: Optional[int] = None):
        """
        Initialize the groovy commutator.

        Args:
            mode: 'binary' for CA states, 'integer' for numeric sequences
            rule: Wolfram rule number (only used in binary mode)
        """
        if mode not in ('binary', 'integer'):
            raise ValueError("Mode must be 'binary' or 'integer'")

        self.mode = mode
        self.rule = rule

    def delta(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute the discrete derivative Delta(psi).

        Args:
            psi: Input sequence

        Returns:
            Derivative sequence
        """
        if self.mode == 'binary':
            return delta_binary(psi, self.rule)
        else:
            return delta_integer(psi)

    def delta_padded(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute the discrete derivative with length preservation.

        For integer mode, pads with 0. For binary mode, uses circular boundary.

        Args:
            psi: Input sequence

        Returns:
            Derivative sequence (same length as input)
        """
        if self.mode == 'binary':
            return delta_binary(psi, self.rule)
        else:
            return delta_integer_padded(psi)

    def compute(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute the groovy commutator K(psi).

        K(psi) = Delta(psi + Delta(psi)) - (Delta(psi) + Delta(Delta(psi)))

        For binary mode, + and - are XOR operations.
        For integer mode, + and - are standard arithmetic.

        Args:
            psi: Input sequence

        Returns:
            Groovy commutator values
        """
        if self.mode == 'binary':
            return self._compute_binary(psi)
        else:
            return self._compute_integer(psi)

    def _compute_binary(self, psi: np.ndarray) -> np.ndarray:
        """Compute K for binary (CA) states using XOR operations."""
        # Delta(psi)
        d_psi = delta_binary(psi, self.rule)

        # Delta(Delta(psi))
        d_d_psi = delta_binary(d_psi, self.rule)

        # psi + Delta(psi) in binary is XOR
        psi_plus_d = np.bitwise_xor(psi, d_psi).astype(np.uint8)

        # Delta(psi + Delta(psi))
        d_psi_plus_d = delta_binary(psi_plus_d, self.rule)

        # Delta(psi) + Delta(Delta(psi)) in binary is XOR
        d_plus_dd = np.bitwise_xor(d_psi, d_d_psi).astype(np.uint8)

        # Final subtraction in binary is XOR
        k = np.bitwise_xor(d_psi_plus_d, d_plus_dd).astype(np.uint8)

        return k

    def _compute_integer(self, psi: np.ndarray) -> np.ndarray:
        """Compute K for integer sequences using standard arithmetic."""
        # Ensure we're working with a numeric array
        psi = np.asarray(psi, dtype=np.int64)

        # Delta(psi) - length n-1
        d_psi = delta_integer(psi)

        if len(d_psi) < 2:
            return np.array([], dtype=np.int64)

        # Delta(Delta(psi)) - length n-2
        d_d_psi = delta_integer(d_psi)

        # psi + Delta(psi) - need to align lengths
        # Use psi[:-1] to match length of d_psi
        psi_plus_d = psi[:-1] + d_psi  # This equals psi[1:]

        # Delta(psi + Delta(psi)) - length n-2
        d_psi_plus_d = delta_integer(psi_plus_d)

        # Delta(psi) + Delta(Delta(psi)) - need to align
        # Use d_psi[:-1] to match length of d_d_psi
        d_plus_dd = d_psi[:-1] + d_d_psi

        # K = Delta(psi + Delta(psi)) - (Delta(psi) + Delta(Delta(psi)))
        k = d_psi_plus_d - d_plus_dd

        return k

    def __repr__(self) -> str:
        if self.mode == 'binary':
            return f"GroovyCommutator(mode='binary', rule={self.rule})"
        return f"GroovyCommutator(mode='integer')"


def groovy_commutator(psi: np.ndarray, mode: str = 'integer',
                       rule: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to compute the groovy commutator K(psi).

    K(psi) = Delta(psi + Delta(psi)) - (Delta(psi) + Delta(Delta(psi)))

    Args:
        psi: Input sequence
        mode: 'binary' for CA states, 'integer' for numeric sequences
        rule: Wolfram rule number (only for binary mode)

    Returns:
        Groovy commutator values
    """
    gc = GroovyCommutator(mode=mode, rule=rule)
    return gc.compute(psi)


# Prime number utilities for structure exploration

def primes_up_to(n: int) -> np.ndarray:
    """
    Generate all prime numbers up to n using the Sieve of Eratosthenes.

    Args:
        n: Upper bound (inclusive)

    Returns:
        Array of prime numbers
    """
    if n < 2:
        return np.array([], dtype=np.int64)

    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False

    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    return np.where(sieve)[0].astype(np.int64)


def first_n_primes(n: int) -> np.ndarray:
    """
    Generate the first n prime numbers.

    Args:
        n: Number of primes to generate

    Returns:
        Array of the first n prime numbers
    """
    if n <= 0:
        return np.array([], dtype=np.int64)

    # Estimate upper bound using prime number theorem
    if n < 6:
        upper = 15
    else:
        upper = int(n * (np.log(n) + np.log(np.log(n)) + 2))

    primes = primes_up_to(upper)

    # If we didn't get enough, keep doubling
    while len(primes) < n:
        upper *= 2
        primes = primes_up_to(upper)

    return primes[:n]


def prime_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Compute the gaps between consecutive primes.

    Args:
        primes: Array of prime numbers

    Returns:
        Array of gaps (p_{n+1} - p_n)
    """
    return np.diff(primes)


def analyze_prime_structure(n_primes: int = 1000) -> dict:
    """
    Analyze the structure of prime numbers using the groovy commutator.

    Args:
        n_primes: Number of primes to analyze

    Returns:
        Dictionary containing:
            - 'primes': The prime numbers
            - 'gaps': Prime gaps (first differences)
            - 'gap_diffs': Second differences of gaps
            - 'K_primes': Groovy commutator of primes
            - 'K_gaps': Groovy commutator of prime gaps
            - 'K_nonzero_ratio': Ratio of non-zero K values
    """
    primes = first_n_primes(n_primes)
    gaps = prime_gaps(primes)

    gc = GroovyCommutator(mode='integer')

    # Analyze primes directly
    k_primes = gc.compute(primes)

    # Analyze prime gaps (often reveals more structure)
    k_gaps = gc.compute(gaps) if len(gaps) > 2 else np.array([])

    # Statistics
    k_nonzero = np.count_nonzero(k_primes)
    k_ratio = k_nonzero / len(k_primes) if len(k_primes) > 0 else 0

    return {
        'primes': primes,
        'gaps': gaps,
        'gap_diffs': np.diff(gaps) if len(gaps) > 1 else np.array([]),
        'K_primes': k_primes,
        'K_gaps': k_gaps,
        'K_nonzero_ratio': k_ratio,
        'K_mean': np.mean(np.abs(k_primes)) if len(k_primes) > 0 else 0,
        'K_std': np.std(k_primes) if len(k_primes) > 0 else 0,
    }


if __name__ == '__main__':
    # Demo: Binary mode with CA rule (this is where K becomes non-trivial!)
    print("Groovy Commutator for Binary CA States")
    print("=" * 50)

    # The groovy commutator is non-trivial when Delta uses a non-linear CA rule
    from .ca_calculus import CACalculus

    # Create a binary pattern
    width = 31
    binary_state = np.zeros(width, dtype=np.uint8)
    binary_state[width // 2] = 1  # Single cell in center
    print(f"\nInitial binary state (single cell):")
    print(f"  {''.join(str(x) for x in binary_state)}")

    # Test with different rules
    for rule in [30, 90, 110]:
        print(f"\n--- Rule {rule} ---")
        gc = GroovyCommutator(mode='binary', rule=rule)
        calc = CACalculus(rule=rule)

        # Evolve a few steps and compute K at each
        state = binary_state.copy()
        total_k = 0
        for step in range(5):
            k = gc.compute(state)
            k_sum = np.sum(k)
            total_k += k_sum
            c = calc.commutator(state)
            c_sum = np.sum(c)
            print(f"  Step {step}: K nonlinearity={k_sum:3d}, C nonlinearity={c_sum:3d}")
            state = calc.evolve(state)

        print(f"  Total K: {total_k}")

    # Demo: Analyze prime number structure
    print("\n" + "=" * 50)
    print("Groovy Commutator Analysis of Prime Numbers")
    print("=" * 50)
    print("\nNote: For integer sequences with standard +/-, K is identically 0")
    print("(this is a mathematical identity for linear operations).")
    print("The groovy commutator reveals structure when Delta is non-linear.")

    results = analyze_prime_structure(100)

    print(f"\nFirst 20 primes: {results['primes'][:20]}")
    print(f"First 20 gaps: {results['gaps'][:20]}")
    print(f"\nGroovy commutator K of primes (first 20): {results['K_primes'][:20]}")
    print(f"K is zero: {np.all(results['K_primes'] == 0)}")

    # For primes, the interesting analysis uses the CA framework
    # by encoding prime gaps as binary patterns
    print("\n" + "=" * 50)
    print("Encoding Prime Gaps as Binary Patterns")
    print("=" * 50)

    gaps = results['gaps'][:50]
    # Create binary encoding: gap mod 2 (even/odd structure)
    binary_gaps = (gaps % 2).astype(np.uint8)
    print(f"\nPrime gaps mod 2 (even=0, odd=1):")
    print(f"  {''.join(str(x) for x in binary_gaps)}")

    for rule in [30, 90]:
        gc = GroovyCommutator(mode='binary', rule=rule)
        k = gc.compute(binary_gaps)
        print(f"\nRule {rule} K on gap parity: sum={np.sum(k)}")
        print(f"  K: {''.join(str(x) for x in k)}")
