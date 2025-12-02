#!/usr/bin/env python3
"""
Groovy Commutator (K) Analysis on Twin Prime Gaps

Investigates whether the cumulative sum of K(n) across twin prime gaps
converges to -π or some other constant.

Definitions:
    - Arithmetic Derivative D(n):
        D(1) = 0
        D(p) = 1 for prime p
        D(ab) = a*D(b) + b*D(a) (product rule)

    - Groovy Commutator K(n):
        K(n) = D(n + D(n)) - (D(n) + D(D(n)))

For twin primes (p, p+2), we calculate K(p+1) - the commutator of the
composite number between them.

We track: sum(K(p+1)) / count_of_twin_pairs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
from functools import lru_cache
import sys

# ============================================================================
# PART 1: Prime Sieve and Factorization
# ============================================================================

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    """
    Generate boolean array where sieve[i] = True iff i is prime.
    Uses numpy for speed.
    """
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False

    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    return sieve


def get_primes_up_to(n: int) -> np.ndarray:
    """Return array of all primes up to n."""
    sieve = sieve_of_eratosthenes(n)
    return np.where(sieve)[0]


def get_twin_primes(n: int) -> List[Tuple[int, int]]:
    """Return list of twin prime pairs (p, p+2) where p+2 <= n."""
    sieve = sieve_of_eratosthenes(n)
    twins = []

    for p in range(3, n - 1):
        if sieve[p] and sieve[p + 2]:
            twins.append((p, p + 2))

    return twins


# ============================================================================
# PART 2: Arithmetic Derivative
# ============================================================================

class ArithmeticDerivative:
    """
    Computes the arithmetic derivative D(n) efficiently using cached
    factorization and the formula:

        D(n) = n * sum(e_i / p_i) for n = p_1^e_1 * p_2^e_2 * ...

    Properties:
        D(0) = 0 (by convention)
        D(1) = 0
        D(p) = 1 for prime p
        D(ab) = a*D(b) + b*D(a) (Leibniz rule)
    """

    def __init__(self, max_n: int):
        """
        Initialize with prime sieve up to max_n for efficient factorization.
        """
        self.max_n = max_n
        self.sieve = sieve_of_eratosthenes(max_n)
        self.primes = get_primes_up_to(int(max_n**0.5) + 1)

        # Cache for small values
        self._cache = {}

    def is_prime(self, n: int) -> bool:
        """Check if n is prime using pre-computed sieve."""
        if n <= self.max_n:
            return self.sieve[n]
        # Fallback for numbers beyond sieve
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for p in range(3, int(n**0.5) + 1, 2):
            if n % p == 0:
                return False
        return True

    def factorize(self, n: int) -> List[Tuple[int, int]]:
        """
        Return prime factorization as list of (prime, exponent) pairs.
        """
        if n <= 1:
            return []

        factors = []
        temp = n

        for p in self.primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((int(p), exp))

        if temp > 1:
            factors.append((temp, 1))

        return factors

    def derivative(self, n: int) -> int:
        """
        Compute D(n) using the formula:
            D(n) = n * sum(e_i / p_i)

        For n = p_1^e_1 * ... * p_k^e_k:
            D(n) = n * (e_1/p_1 + e_2/p_2 + ... + e_k/p_k)

        We compute this exactly using integer arithmetic to avoid
        floating point errors.
        """
        if n <= 1:
            return 0

        # Check cache
        if n in self._cache:
            return self._cache[n]

        # For prime, D(p) = 1
        if n <= self.max_n and self.sieve[n]:
            return 1

        # Factorize and use the formula
        factors = self.factorize(n)

        if not factors:
            return 0

        # D(n) = sum over i of (n / p_i) * e_i
        # This is integer arithmetic equivalent of n * sum(e_i/p_i)
        result = 0
        for p, e in factors:
            result += (n // p) * e

        # Cache small results
        if n <= 10000:
            self._cache[n] = result

        return result


# ============================================================================
# PART 3: Groovy Commutator K
# ============================================================================

def groovy_commutator_K(n: int, D: ArithmeticDerivative) -> int:
    """
    Compute the Groovy Commutator:
        K(n) = D(n + D(n)) - (D(n) + D(D(n)))

    This measures the "non-linearity" of the arithmetic derivative
    at point n.
    """
    Dn = D.derivative(n)
    DDn = D.derivative(Dn)
    D_n_plus_Dn = D.derivative(n + Dn)

    return D_n_plus_Dn - (Dn + DDn)


# ============================================================================
# PART 4: Main Experiment
# ============================================================================

def run_experiment(N: int = 10_000_000, checkpoint_interval: int = 10000):
    """
    Run the full experiment:
    1. Find all twin primes up to N
    2. For each twin pair (p, p+2), compute K(p+1)
    3. Track cumulative sum and ratio
    4. Return data for analysis
    """
    print(f"=" * 70)
    print(f"GROOVY COMMUTATOR TWIN PRIME EXPERIMENT")
    print(f"=" * 70)
    print(f"Target N = {N:,}")
    print()

    # Initialize arithmetic derivative with sieve
    print("Initializing arithmetic derivative engine...")
    t0 = time.time()

    # We need sieve up to at least N + D(N) which could be large
    # D(n) <= n * log(n) roughly, so let's be safe
    sieve_limit = N + 2 * int(N * np.log(N) / np.log(2)) if N > 2 else 100
    # Cap it to avoid memory issues
    sieve_limit = min(sieve_limit, 100_000_000)

    D = ArithmeticDerivative(sieve_limit)
    print(f"  Sieve initialized up to {sieve_limit:,} in {time.time() - t0:.2f}s")

    # Find twin primes
    print("Finding twin primes...")
    t0 = time.time()
    twin_primes = get_twin_primes(N)
    print(f"  Found {len(twin_primes):,} twin prime pairs in {time.time() - t0:.2f}s")
    print(f"  First few: {twin_primes[:5]}")
    print(f"  Last few: {twin_primes[-5:]}")
    print()

    # Calculate K values
    print("Computing K(p+1) for all twin pairs...")
    t0 = time.time()

    K_values = []
    cumulative_K = []
    ratios = []
    p_values = []  # Store p for each twin pair for plotting

    sum_K = 0

    for i, (p, p2) in enumerate(twin_primes):
        # The gap contains p+1 (the composite between twins)
        composite = p + 1
        K_val = groovy_commutator_K(composite, D)

        K_values.append(K_val)
        sum_K += K_val
        cumulative_K.append(sum_K)

        # Ratio: cumulative K / number of gaps
        ratio = sum_K / (i + 1)
        ratios.append(ratio)
        p_values.append(p)

        # Progress report
        if (i + 1) % checkpoint_interval == 0:
            print(f"  Processed {i+1:,} / {len(twin_primes):,} pairs | "
                  f"Sum(K) = {sum_K:,} | Ratio = {ratio:.6f}")

    print(f"  Completed in {time.time() - t0:.2f}s")
    print()

    return {
        'twin_primes': twin_primes,
        'K_values': np.array(K_values),
        'cumulative_K': np.array(cumulative_K),
        'ratios': np.array(ratios),
        'p_values': np.array(p_values),
        'N': N
    }


def analyze_results(data: Dict):
    """
    Analyze the results and check for convergence to -π or other constants.
    """
    print(f"=" * 70)
    print(f"ANALYSIS")
    print(f"=" * 70)

    K_values = data['K_values']
    cumulative_K = data['cumulative_K']
    ratios = data['ratios']
    p_values = data['p_values']

    n_pairs = len(K_values)

    # Basic statistics
    print(f"\nBasic Statistics on K(p+1):")
    print(f"  Total twin pairs analyzed: {n_pairs:,}")
    print(f"  Sum of K values: {cumulative_K[-1]:,}")
    print(f"  Mean K: {np.mean(K_values):.6f}")
    print(f"  Std K: {np.std(K_values):.6f}")
    print(f"  Min K: {np.min(K_values):,}")
    print(f"  Max K: {np.max(K_values):,}")

    # Final ratio
    final_ratio = ratios[-1]
    print(f"\nFinal Ratio (Sum K / count):")
    print(f"  Final ratio = {final_ratio:.10f}")
    print(f"  -π = {-np.pi:.10f}")
    print(f"  Difference from -π: {abs(final_ratio - (-np.pi)):.10f}")
    print(f"  Relative error: {abs(final_ratio + np.pi) / np.pi * 100:.4f}%")

    # Check other constants
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    print(f"\nComparison with other constants:")
    print(f"  -φ (golden ratio) = {-phi:.10f}, diff = {abs(final_ratio + phi):.10f}")
    print(f"  -e = {-np.e:.10f}, diff = {abs(final_ratio + np.e):.10f}")
    print(f"  -2 = -2.0, diff = {abs(final_ratio + 2):.10f}")
    print(f"  -3 = -3.0, diff = {abs(final_ratio + 3):.10f}")
    print(f"  -4 = -4.0, diff = {abs(final_ratio + 4):.10f}")

    # Convergence analysis at different scales
    print(f"\nConvergence at different scales:")
    checkpoints = [100, 1000, 10000, 50000, 100000, n_pairs - 1]
    checkpoints = [c for c in checkpoints if c < n_pairs]

    for idx in checkpoints:
        r = ratios[idx]
        p = p_values[idx]
        print(f"  After {idx+1:,} pairs (p ~ {p:,}): ratio = {r:.8f}, "
              f"diff from -π = {abs(r + np.pi):.8f}")

    # Check if ratio is drifting or converging
    if n_pairs > 1000:
        early_ratio = np.mean(ratios[100:1000])
        mid_ratio = np.mean(ratios[n_pairs//2 - 500:n_pairs//2 + 500])
        late_ratio = np.mean(ratios[-1000:])

        print(f"\nDrift Analysis:")
        print(f"  Early mean (100-1000): {early_ratio:.8f}")
        print(f"  Mid mean (middle 1000): {mid_ratio:.8f}")
        print(f"  Late mean (last 1000): {late_ratio:.8f}")

        drift = late_ratio - early_ratio
        print(f"  Total drift: {drift:.8f}")

        if abs(drift) < 0.1:
            print(f"  Status: Appears to be CONVERGING")
        else:
            print(f"  Status: Still DRIFTING")

    return final_ratio


def create_plot(data: Dict, output_path: str):
    """
    Create a comprehensive plot showing the convergence analysis.
    """
    ratios = data['ratios']
    p_values = data['p_values']
    K_values = data['K_values']
    cumulative_K = data['cumulative_K']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Groovy Commutator K(n) Analysis on Twin Prime Gaps\n'
                 f'N = {data["N"]:,}, Twin pairs = {len(ratios):,}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Ratio convergence
    ax1 = axes[0, 0]
    ax1.plot(p_values, ratios, 'b-', linewidth=0.5, alpha=0.7, label='Ratio = ΣK / count')
    ax1.axhline(y=-np.pi, color='r', linestyle='--', linewidth=2, label=f'-π ≈ {-np.pi:.6f}')
    ax1.axhline(y=-(1 + np.sqrt(5))/2, color='g', linestyle=':', linewidth=2, label=f'-φ ≈ {-(1+np.sqrt(5))/2:.6f}')
    ax1.set_xlabel('p (lower twin prime)')
    ax1.set_ylabel('Ratio = Σ K(p+1) / count')
    ax1.set_title('Ratio Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Cumulative K vs twin pair index
    ax2 = axes[0, 1]
    ax2.plot(range(len(cumulative_K)), cumulative_K, 'b-', linewidth=0.5)
    # Add reference line for -π * n
    n_range = np.arange(1, len(cumulative_K) + 1)
    ax2.plot(n_range, -np.pi * n_range, 'r--', linewidth=2, alpha=0.7, label=f'-π × n')
    ax2.set_xlabel('Twin pair index')
    ax2.set_ylabel('Cumulative Σ K(p+1)')
    ax2.set_title('Cumulative K Sum')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of K values
    ax3 = axes[1, 0]
    # Clip extreme values for visualization
    K_clipped = np.clip(K_values, -500, 500)
    ax3.hist(K_clipped, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax3.axvline(x=np.mean(K_values), color='r', linestyle='--', linewidth=2,
                label=f'Mean = {np.mean(K_values):.2f}')
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax3.set_xlabel('K(p+1) value')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of K Values (clipped to [-500, 500])')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Deviation from -π over log scale
    ax4 = axes[1, 1]
    deviation = np.array(ratios) + np.pi
    ax4.plot(p_values, deviation, 'b-', linewidth=0.5, alpha=0.7)
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero (exact -π)')
    ax4.set_xlabel('p (lower twin prime)')
    ax4.set_ylabel('Deviation from -π')
    ax4.set_title('Deviation from -π Convergence')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig


def main():
    """Main entry point."""
    # Run experiment up to N = 10,000,000
    N = 10_000_000

    print(f"\nStarting experiment with N = {N:,}")
    print(f"This may take several minutes...\n")

    data = run_experiment(N, checkpoint_interval=25000)

    final_ratio = analyze_results(data)

    # Create output directory if needed
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(output_dir), 'images')
    os.makedirs(images_dir, exist_ok=True)

    output_path = os.path.join(images_dir, 'groovy_commutator_convergence.png')
    create_plot(data, output_path)

    # Final verdict
    print(f"\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    diff_from_pi = abs(final_ratio + np.pi)

    if diff_from_pi < 0.01:
        print(f"✓ The ratio appears to converge to -π!")
        print(f"  Final ratio: {final_ratio:.10f}")
        print(f"  -π:          {-np.pi:.10f}")
        print(f"  Error:       {diff_from_pi:.10f} ({diff_from_pi/np.pi*100:.4f}%)")
    elif diff_from_pi < 0.1:
        print(f"? The ratio is close to -π but not converged")
        print(f"  Final ratio: {final_ratio:.10f}")
        print(f"  -π:          {-np.pi:.10f}")
        print(f"  Error:       {diff_from_pi:.10f}")
        print(f"  Verdict: INCONCLUSIVE - may need larger N")
    else:
        print(f"✗ The ratio does NOT appear to converge to -π")
        print(f"  Final ratio: {final_ratio:.10f}")
        print(f"  -π:          {-np.pi:.10f}")
        print(f"  Error:       {diff_from_pi:.10f}")

        # Check what it might converge to
        if abs(final_ratio + (1 + np.sqrt(5))/2) < diff_from_pi:
            print(f"  Closer to -φ (golden ratio): {-(1+np.sqrt(5))/2:.10f}")
        if abs(final_ratio + np.e) < diff_from_pi:
            print(f"  Closer to -e: {-np.e:.10f}")

    return data


if __name__ == '__main__':
    data = main()
