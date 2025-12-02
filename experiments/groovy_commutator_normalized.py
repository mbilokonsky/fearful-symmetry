#!/usr/bin/env python3
"""
Groovy Commutator (K) Analysis - Normalized Ratios

The raw ratio ΣK/count diverges because K(n) grows with n.
This script explores multiple normalizations to find the -π signature.

Candidate Ratios:
    R1 = ΣK / Σ(p+1)              - normalize by sum of gap composites
    R2 = Σ(K/(p+1)) / count       - mean of normalized K
    R3 = ΣK / Σ(D(p+1))           - normalize by sum of derivatives
    R4 = Σ(K/D(p+1)) / count      - mean of K/D ratio
    R5 = ΣK / Σ(p·log(p))         - prime counting normalization
    R6 = ΣK / (Σp)^(3/2)          - power scaling
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

# ============================================================================
# PART 1: Core Functions (from previous script)
# ============================================================================

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve


def get_twin_primes(n: int) -> List[Tuple[int, int]]:
    sieve = sieve_of_eratosthenes(n)
    twins = []
    for p in range(3, n - 1):
        if sieve[p] and sieve[p + 2]:
            twins.append((p, p + 2))
    return twins


class ArithmeticDerivative:
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.sieve = sieve_of_eratosthenes(max_n)
        self.primes = np.where(sieve_of_eratosthenes(int(max_n**0.5) + 1))[0]
        self._cache = {}

    def factorize(self, n: int) -> List[Tuple[int, int]]:
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
        if n <= 1:
            return 0
        if n in self._cache:
            return self._cache[n]
        if n <= self.max_n and self.sieve[n]:
            return 1
        factors = self.factorize(n)
        if not factors:
            return 0
        result = 0
        for p, e in factors:
            result += (n // p) * e
        if n <= 10000:
            self._cache[n] = result
        return result


def groovy_commutator_K(n: int, D: ArithmeticDerivative) -> int:
    Dn = D.derivative(n)
    DDn = D.derivative(Dn)
    D_n_plus_Dn = D.derivative(n + Dn)
    return D_n_plus_Dn - (Dn + DDn)


# ============================================================================
# PART 2: Multi-Normalization Experiment
# ============================================================================

def run_normalized_experiment(N: int = 10_000_000):
    """
    Test multiple normalizations to find -π signature.
    """
    print("=" * 70)
    print("GROOVY COMMUTATOR - NORMALIZED RATIO SEARCH")
    print("=" * 70)
    print(f"Target N = {N:,}\n")

    # Initialize
    sieve_limit = min(N + 2 * int(N * np.log(N) / np.log(2)), 100_000_000)
    D = ArithmeticDerivative(sieve_limit)
    print(f"Sieve initialized to {sieve_limit:,}")

    twin_primes = get_twin_primes(N)
    print(f"Found {len(twin_primes):,} twin prime pairs\n")

    # Track multiple quantities
    n_pairs = len(twin_primes)

    # Running sums
    sum_K = 0
    sum_p_plus_1 = 0
    sum_K_normalized = 0.0  # K/(p+1)
    sum_D_p_plus_1 = 0
    sum_K_over_D = 0.0
    sum_p = 0
    sum_p_log_p = 0.0
    sum_K_over_p = 0.0

    # Track ratios over time
    R1_history = []  # ΣK / Σ(p+1)
    R2_history = []  # mean(K/(p+1))
    R3_history = []  # ΣK / Σ(D(p+1))
    R4_history = []  # mean(K/D(p+1))
    R5_history = []  # ΣK / Σ(p·log(p))
    R6_history = []  # ΣK / (count)^(3/2) — different scaling
    R7_history = []  # K(p+1)/(p+1) individual values
    R8_history = []  # ΣK / count² — quadratic scaling
    p_history = []

    print("Computing K values and tracking ratios...")
    t0 = time.time()

    for i, (p, p2) in enumerate(twin_primes):
        composite = p + 1
        K_val = groovy_commutator_K(composite, D)
        D_composite = D.derivative(composite)

        sum_K += K_val
        sum_p_plus_1 += composite
        sum_K_normalized += K_val / composite if composite > 0 else 0
        sum_D_p_plus_1 += D_composite
        sum_K_over_D += K_val / D_composite if D_composite > 0 else 0
        sum_p += p
        sum_p_log_p += p * np.log(p)
        sum_K_over_p += K_val / p if p > 0 else 0

        count = i + 1

        # Record ratios
        R1_history.append(sum_K / sum_p_plus_1 if sum_p_plus_1 > 0 else 0)
        R2_history.append(sum_K_normalized / count)
        R3_history.append(sum_K / sum_D_p_plus_1 if sum_D_p_plus_1 > 0 else 0)
        R4_history.append(sum_K_over_D / count)
        R5_history.append(sum_K / sum_p_log_p if sum_p_log_p > 0 else 0)
        R6_history.append(sum_K / (count ** 1.5) if count > 0 else 0)
        R7_history.append(K_val / composite)  # Individual K/(p+1)
        R8_history.append(sum_K / (count ** 2) if count > 0 else 0)
        p_history.append(p)

        if (i + 1) % 25000 == 0:
            print(f"  {i+1:,} pairs: R1={R1_history[-1]:.6f}, R2={R2_history[-1]:.6f}")

    print(f"Completed in {time.time() - t0:.2f}s\n")

    # Convert to arrays
    results = {
        'R1': np.array(R1_history),  # ΣK / Σ(p+1)
        'R2': np.array(R2_history),  # mean(K/(p+1))
        'R3': np.array(R3_history),  # ΣK / Σ(D(p+1))
        'R4': np.array(R4_history),  # mean(K/D)
        'R5': np.array(R5_history),  # ΣK / Σ(p·log p)
        'R6': np.array(R6_history),  # ΣK / count^1.5
        'R7': np.array(R7_history),  # Individual K/(p+1)
        'R8': np.array(R8_history),  # ΣK / count²
        'p': np.array(p_history),
        'n_pairs': n_pairs,
        'N': N
    }

    return results


def analyze_ratios(results: Dict):
    """
    Analyze all ratios and check for -π convergence.
    """
    print("=" * 70)
    print("RATIO ANALYSIS - Checking for -π convergence")
    print("=" * 70)

    ratio_names = {
        'R1': 'ΣK / Σ(p+1)',
        'R2': 'mean(K/(p+1))',
        'R3': 'ΣK / Σ(D(p+1))',
        'R4': 'mean(K/D(p+1))',
        'R5': 'ΣK / Σ(p·log p)',
        'R6': 'ΣK / count^1.5',
        'R7': 'Individual K/(p+1)',
        'R8': 'ΣK / count²'
    }

    print(f"\nFinal values and distance from -π ({-np.pi:.8f}):\n")
    print(f"{'Ratio':<20} {'Final Value':>18} {'|Diff from -π|':>16} {'Status':>20}")
    print("-" * 76)

    best_ratio = None
    best_diff = float('inf')

    for key in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R8']:
        final_val = results[key][-1]
        diff = abs(final_val + np.pi)

        # Check if converging
        if len(results[key]) > 1000:
            early = np.mean(results[key][100:500])
            late = np.mean(results[key][-500:])
            drift = abs(late - early)
            status = "CONVERGING" if drift < abs(final_val) * 0.1 else "DRIFTING"
        else:
            status = "INSUFFICIENT DATA"

        print(f"{ratio_names[key]:<20} {final_val:>18.8f} {diff:>16.8f} {status:>20}")

        if diff < best_diff:
            best_diff = diff
            best_ratio = key

    # Also check running mean of R7
    R7_running_mean = np.cumsum(results['R7']) / np.arange(1, len(results['R7']) + 1)
    final_R7_mean = R7_running_mean[-1]
    diff_R7 = abs(final_R7_mean + np.pi)
    print(f"{'Running mean R7':<20} {final_R7_mean:>18.8f} {diff_R7:>16.8f}")

    print(f"\nBest match: {ratio_names.get(best_ratio, best_ratio)} with diff = {best_diff:.8f}")

    # Detailed analysis of most promising ratios
    print("\n" + "=" * 70)
    print("DETAILED CONVERGENCE ANALYSIS")
    print("=" * 70)

    for key in ['R1', 'R2', 'R3', 'R4']:
        r = results[key]
        print(f"\n{ratio_names[key]}:")

        checkpoints = [100, 1000, 5000, 10000, 30000, len(r)-1]
        checkpoints = [c for c in checkpoints if c < len(r)]

        for idx in checkpoints:
            val = r[idx]
            diff = abs(val + np.pi)
            print(f"  n={idx+1:>6,}: ratio = {val:>12.8f}, |diff from -π| = {diff:.8f}")

    return best_ratio, best_diff


def create_comprehensive_plot(results: Dict, output_path: str):
    """
    Create a comprehensive plot showing multiple ratio convergences.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Groovy Commutator K(n): Searching for -π Signature\n'
                 f'Twin primes up to N={results["N"]:,}, {results["n_pairs"]:,} pairs',
                 fontsize=14, fontweight='bold')

    p = results['p']

    ratio_configs = [
        ('R1', 'ΣK / Σ(p+1)', axes[0, 0]),
        ('R2', 'mean(K/(p+1))', axes[0, 1]),
        ('R3', 'ΣK / Σ D(p+1)', axes[0, 2]),
        ('R4', 'mean(K/D(p+1))', axes[1, 0]),
        ('R5', 'ΣK / Σ(p·log p)', axes[1, 1]),
    ]

    for key, label, ax in ratio_configs:
        r = results[key]
        ax.plot(p, r, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(y=-np.pi, color='r', linestyle='--', linewidth=2,
                   label=f'-π = {-np.pi:.4f}')

        # Add final value annotation
        final_val = r[-1]
        ax.axhline(y=final_val, color='g', linestyle=':', linewidth=1,
                   label=f'Final = {final_val:.4f}')

        ax.set_xlabel('p (lower twin prime)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    # Last subplot: Distribution of K/(p+1)
    ax = axes[1, 2]
    R7_clipped = np.clip(results['R7'], -50, 50)
    ax.hist(R7_clipped, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(x=-np.pi, color='r', linestyle='--', linewidth=2, label=f'-π')
    ax.axvline(x=np.mean(results['R7']), color='g', linestyle=':', linewidth=2,
               label=f'Mean = {np.mean(results["R7"]):.4f}')
    ax.set_xlabel('K(p+1)/(p+1)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of K/(p+1)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig


def investigate_scaling(results: Dict):
    """
    Investigate how K(p+1) scales with p to understand the right normalization.
    """
    print("\n" + "=" * 70)
    print("SCALING INVESTIGATION")
    print("=" * 70)

    R7 = results['R7']  # K/(p+1) values
    p = results['p']

    # Check if K ~ p, K ~ p*log(p), K ~ p², etc.
    # by looking at how K/(p+1) behaves

    # Look at mean K/(p+1) in windows
    window_size = 5000
    n_windows = len(R7) // window_size

    print(f"\nMean K/(p+1) in windows of {window_size} twin pairs:")
    print(f"{'Window':<10} {'p_center':>15} {'Mean K/(p+1)':>15} {'Std':>15}")
    print("-" * 58)

    window_means = []
    p_centers = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_mean = np.mean(R7[start:end])
        window_std = np.std(R7[start:end])
        p_center = np.mean(p[start:end])

        window_means.append(window_mean)
        p_centers.append(p_center)

        print(f"{i+1:<10} {p_center:>15,.0f} {window_mean:>15.6f} {window_std:>15.6f}")

    # Check if window means are converging
    if len(window_means) >= 2:
        trend = window_means[-1] - window_means[0]
        print(f"\nTrend from first to last window: {trend:+.6f}")

        if abs(trend) < abs(window_means[-1]) * 0.2:
            print("→ K/(p+1) appears roughly CONSTANT as p → ∞")
        else:
            print("→ K/(p+1) is still DRIFTING")


def main():
    N = 10_000_000

    print(f"\nStarting normalized ratio experiment with N = {N:,}\n")

    results = run_normalized_experiment(N)

    best_ratio, best_diff = analyze_ratios(results)

    investigate_scaling(results)

    # Create plot
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(output_dir), 'images')
    os.makedirs(images_dir, exist_ok=True)

    output_path = os.path.join(images_dir, 'groovy_commutator_normalized.png')
    create_comprehensive_plot(results, output_path)

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if best_diff < 0.1:
        print(f"✓ Found potential -π signature in {best_ratio}!")
        print(f"  Closest approach: {best_diff:.8f} from -π")
    elif best_diff < 1.0:
        print(f"? Weak signal for -π in {best_ratio}")
        print(f"  Distance: {best_diff:.8f}")
        print("  May need larger N or different normalization")
    else:
        print(f"✗ No clear -π signature found in standard normalizations")
        print(f"  Best match ({best_ratio}): {best_diff:.8f} from -π")
        print("\n  The Groovy Commutator K(n) on twin prime gaps does NOT")
        print("  appear to yield a -π ratio with tested normalizations.")

    return results


if __name__ == '__main__':
    results = main()
