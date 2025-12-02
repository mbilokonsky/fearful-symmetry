#!/usr/bin/env python3
"""
OPEN QUESTIONS INVESTIGATION
============================

This experiment addresses the remaining open questions from CONCLUSION.md:

1. What happens beyond 10M? Does the convergence tighten?
2. Does this generalize to cousin primes (p, p+4) or sexy primes (p, p+6)?
3. Phase space topology - Why is K-evolution memoryless? Hidden structure at different step sizes?
4. Can the K-operator detect other prime patterns? (Sophie Germain, k-tuples)
5. Does the spectrum have physical meaning?
6. Closed-form proof for K(n)/n → -1/2

Each section generates analysis and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fractions import Fraction
from typing import List, Tuple, Dict, Optional
import os
import time

# ============================================================================
# Core Functions (from existing codebase)
# ============================================================================

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    """Generate prime sieve up to n."""
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve


class ArithmeticDerivative:
    """Compute the arithmetic derivative D(n)."""

    def __init__(self, max_n: int):
        self.max_n = max_n
        self.sieve = sieve_of_eratosthenes(max_n)
        self.primes = np.where(sieve_of_eratosthenes(int(max_n**0.5) + 1))[0]

    def derivative(self, n: int) -> int:
        if n <= 1:
            return 0
        if n <= self.max_n and self.sieve[n]:
            return 1

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

        if not factors:
            return 0

        result = 0
        for p, e in factors:
            result += (n // p) * e
        return result


def groovy_commutator_K(n: int, D: ArithmeticDerivative) -> int:
    """Compute K(n) = D(n + D(n)) - (D(n) + D(D(n)))"""
    Dn = D.derivative(n)
    DDn = D.derivative(Dn)
    D_n_plus_Dn = D.derivative(n + Dn)
    return D_n_plus_Dn - (Dn + DDn)


# ============================================================================
# Prime Pattern Finders
# ============================================================================

def get_twin_primes(sieve: np.ndarray, max_n: int) -> List[Tuple[int, int]]:
    """Find twin primes (p, p+2) up to max_n."""
    twins = []
    for p in range(3, max_n):
        if sieve[p] and p + 2 < len(sieve) and sieve[p + 2]:
            twins.append((p, p + 2))
    return twins


def get_cousin_primes(sieve: np.ndarray, max_n: int) -> List[Tuple[int, int]]:
    """Find cousin primes (p, p+4) up to max_n."""
    cousins = []
    for p in range(3, max_n):
        if sieve[p] and p + 4 < len(sieve) and sieve[p + 4]:
            cousins.append((p, p + 4))
    return cousins


def get_sexy_primes(sieve: np.ndarray, max_n: int) -> List[Tuple[int, int]]:
    """Find sexy primes (p, p+6) up to max_n."""
    sexies = []
    for p in range(3, max_n):
        if sieve[p] and p + 6 < len(sieve) and sieve[p + 6]:
            sexies.append((p, p + 6))
    return sexies


def get_sophie_germain_primes(sieve: np.ndarray, max_n: int) -> List[int]:
    """Find Sophie Germain primes p where 2p+1 is also prime."""
    sg_primes = []
    for p in range(2, max_n):
        if sieve[p]:
            safe_prime = 2 * p + 1
            if safe_prime < len(sieve) and sieve[safe_prime]:
                sg_primes.append(p)
    return sg_primes


def get_prime_triplets(sieve: np.ndarray, max_n: int) -> List[Tuple[int, int, int]]:
    """Find prime triplets (p, p+2, p+6) or (p, p+4, p+6)."""
    triplets = []
    for p in range(3, max_n):
        if sieve[p]:
            # Type 1: (p, p+2, p+6)
            if (p + 2 < len(sieve) and sieve[p + 2] and
                p + 6 < len(sieve) and sieve[p + 6]):
                triplets.append((p, p + 2, p + 6))
            # Type 2: (p, p+4, p+6)
            elif (p + 4 < len(sieve) and sieve[p + 4] and
                  p + 6 < len(sieve) and sieve[p + 6]):
                triplets.append((p, p + 4, p + 6))
    return triplets


# ============================================================================
# INVESTIGATION 1: Beyond 10M - Convergence Analysis
# ============================================================================

def investigate_beyond_10M(N_values: List[int] = None):
    """
    Investigate convergence at larger N values.
    Does the -π signature tighten beyond 10M?
    """
    if N_values is None:
        N_values = [1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000]

    print("\n" + "=" * 70)
    print("INVESTIGATION 1: Convergence Beyond 10M")
    print("=" * 70)

    results = []

    for N in N_values:
        print(f"\n[N = {N:,}]")

        sieve_limit = min(N * 3, 200_000_000)
        sieve = sieve_of_eratosthenes(N + 10)
        D = ArithmeticDerivative(sieve_limit)

        twins = get_twin_primes(sieve, N)
        print(f"  Twin pairs found: {len(twins):,}")

        if len(twins) == 0:
            continue

        # Compute K(p+1)/(p+1) for all twins
        K_ratios = []
        for p, p2 in twins:
            composite = p + 1
            K_val = groovy_commutator_K(composite, D)
            K_ratios.append(K_val / composite)

        K_ratios = np.array(K_ratios)

        mean_K = np.mean(K_ratios)
        pi_signal = mean_K * 2 * np.pi
        error = abs(pi_signal - (-np.pi)) / np.pi * 100

        results.append({
            'N': N,
            'n_twins': len(twins),
            'mean_K': mean_K,
            'pi_signal': pi_signal,
            'error_pct': error,
            'std': np.std(K_ratios)
        })

        print(f"  Mean K/(p+1): {mean_K:.6f}")
        print(f"  × 2π = {pi_signal:.6f} (target: -π = {-np.pi:.6f})")
        print(f"  Error: {error:.4f}%")

    # Create convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis: Beyond 10 Million\n'
                 'Does the -π signature tighten at larger N?',
                 fontsize=14, fontweight='bold')

    Ns = [r['N'] for r in results]
    errors = [r['error_pct'] for r in results]
    means = [r['mean_K'] for r in results]
    stds = [r['std'] for r in results]

    # Panel 1: Error vs N
    ax1 = axes[0, 0]
    ax1.semilogx(Ns, errors, 'bo-', linewidth=2, markersize=10)
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('N (upper limit)', fontsize=11)
    ax1.set_ylabel('Relative Error (%)', fontsize=11)
    ax1.set_title('Convergence of -π Signal', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Mean K/(p+1) vs N
    ax2 = axes[0, 1]
    ax2.semilogx(Ns, means, 'ro-', linewidth=2, markersize=10)
    ax2.axhline(y=-0.5, color='gold', linestyle='--', linewidth=2, label='-1/2')
    ax2.set_xlabel('N', fontsize=11)
    ax2.set_ylabel('Mean K/(p+1)', fontsize=11)
    ax2.set_title('Mean K-ratio Convergence', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Standard deviation vs N
    ax3 = axes[1, 0]
    ax3.loglog(Ns, stds, 'go-', linewidth=2, markersize=10)
    ax3.set_xlabel('N', fontsize=11)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('Spread of K-ratios', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [[f"{r['N']:,}", f"{r['n_twins']:,}",
                   f"{r['mean_K']:.6f}", f"{r['error_pct']:.4f}%"]
                  for r in results]
    table = ax4.table(cellText=table_data,
                      colLabels=['N', 'Twin Pairs', 'Mean K/n', 'Error'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Convergence Summary', fontsize=12, pad=20)

    plt.tight_layout()

    return fig, results


# ============================================================================
# INVESTIGATION 2: Cousin and Sexy Primes Generalization
# ============================================================================

def investigate_prime_gaps(N: int = 10_000_000):
    """
    Does the K-structure generalize to other prime gaps?
    - Twin primes: (p, p+2) - gap 2
    - Cousin primes: (p, p+4) - gap 4
    - Sexy primes: (p, p+6) - gap 6
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 2: Generalization to Other Prime Gaps")
    print("=" * 70)

    print(f"\nAnalyzing primes up to N = {N:,}")

    sieve = sieve_of_eratosthenes(N + 10)
    sieve_limit = min(N * 3, 100_000_000)
    D = ArithmeticDerivative(sieve_limit)

    # Find all prime pair types
    twins = get_twin_primes(sieve, N)
    cousins = get_cousin_primes(sieve, N)
    sexies = get_sexy_primes(sieve, N)

    print(f"\nFound:")
    print(f"  Twin primes (gap 2):   {len(twins):,}")
    print(f"  Cousin primes (gap 4): {len(cousins):,}")
    print(f"  Sexy primes (gap 6):   {len(sexies):,}")

    def analyze_pairs(pairs, gap, name):
        """Analyze K(center)/center for prime pairs."""
        if len(pairs) == 0:
            return None

        K_ratios = []
        centers = []

        for p, p2 in pairs:
            # For twins (gap 2): center is p+1
            # For cousins (gap 4): center is p+2
            # For sexies (gap 6): center is p+3
            center = p + gap // 2
            K_val = groovy_commutator_K(center, D)
            K_ratios.append(K_val / center)
            centers.append(center)

        K_ratios = np.array(K_ratios)

        return {
            'name': name,
            'gap': gap,
            'count': len(pairs),
            'mean': np.mean(K_ratios),
            'std': np.std(K_ratios),
            'median': np.median(K_ratios),
            'pi_signal': np.mean(K_ratios) * 2 * np.pi,
            'K_ratios': K_ratios,
            'centers': np.array(centers)
        }

    results = {
        'twins': analyze_pairs(twins, 2, 'Twin (gap 2)'),
        'cousins': analyze_pairs(cousins, 4, 'Cousin (gap 4)'),
        'sexies': analyze_pairs(sexies, 6, 'Sexy (gap 6)')
    }

    # Print comparison
    print("\n" + "-" * 60)
    print("K-STRUCTURE COMPARISON")
    print("-" * 60)
    print(f"{'Type':<20}{'Mean K/n':<15}{'× 2π':<15}{'Std':<12}")
    print("-" * 60)

    for key, r in results.items():
        if r:
            print(f"{r['name']:<20}{r['mean']:<15.6f}{r['pi_signal']:<15.6f}{r['std']:<12.6f}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Prime Gap Generalization: Twin vs Cousin vs Sexy Primes\n'
                 'Does the K-structure depend on gap size?',
                 fontsize=14, fontweight='bold')

    colors = {'twins': 'blue', 'cousins': 'green', 'sexies': 'red'}

    for idx, (key, r) in enumerate(results.items()):
        if r is None:
            continue

        # Top row: Histograms
        ax = axes[0, idx]
        ax.hist(r['K_ratios'], bins=100, color=colors[key], alpha=0.7,
                range=(-3, 2), edgecolor='none')
        ax.axvline(x=-0.5, color='gold', linestyle='--', linewidth=2)
        ax.axvline(x=-5/6, color='orange', linestyle=':', linewidth=2)
        ax.axvline(x=r['mean'], color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('K(center)/center', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f"{r['name']}\nMean={r['mean']:.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Bottom row: Scatter plots
        ax = axes[1, idx]
        sample = min(5000, len(r['centers']))
        idx_sample = np.random.choice(len(r['centers']), sample, replace=False)

        ax.scatter(np.log10(r['centers'][idx_sample]),
                   r['K_ratios'][idx_sample],
                   s=1, alpha=0.5, c=colors[key])
        ax.axhline(y=-0.5, color='gold', linestyle='--', linewidth=2)
        ax.axhline(y=-5/6, color='orange', linestyle=':', linewidth=2)
        ax.set_xlabel('log₁₀(center)', fontsize=10)
        ax.set_ylabel('K/n', fontsize=10)
        ax.set_ylim(-3, 2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, results


# ============================================================================
# INVESTIGATION 3: Phase Space at Different Step Sizes
# ============================================================================

def investigate_phase_space_steps(N: int = 10_000_000):
    """
    Is K-evolution memoryless? Check correlations at different step sizes.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 3: Phase Space Topology at Different Step Sizes")
    print("=" * 70)

    sieve = sieve_of_eratosthenes(N + 100)
    sieve_limit = min(N * 3, 100_000_000)
    D = ArithmeticDerivative(sieve_limit)

    twins = get_twin_primes(sieve, N)
    print(f"\nAnalyzing {len(twins):,} twin prime centers")

    # Compute K(n)/n for all twin centers
    centers = []
    K_ratios = []

    for p, p2 in twins:
        center = p + 1
        K_val = groovy_commutator_K(center, D)
        centers.append(center)
        K_ratios.append(K_val / center)

    centers = np.array(centers)
    K_ratios = np.array(K_ratios)

    # Test different step sizes (in terms of indices)
    step_sizes = [1, 2, 5, 10, 20, 50, 100]
    correlations = []

    print("\nCorrelation Analysis:")
    print("-" * 40)
    print(f"{'Step':<10}{'Correlation':<15}{'P-value Proxy':<15}")
    print("-" * 40)

    for step in step_sizes:
        if step >= len(K_ratios):
            continue

        x = K_ratios[:-step]
        y = K_ratios[step:]

        # Pearson correlation
        corr = np.corrcoef(x, y)[0, 1]
        correlations.append((step, corr))

        # Rough p-value estimate
        n = len(x)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0

        print(f"{step:<10}{corr:<15.6f}{abs(t_stat):<15.4f}")

    # Create phase space visualizations
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Phase Space Topology: K(n)/n vs K(n+k)/(n+k)\n'
                 'Is there hidden structure at different step sizes?',
                 fontsize=14, fontweight='bold')

    steps_to_plot = [1, 2, 5, 10, 20, 50, 100]

    for idx, step in enumerate(steps_to_plot[:4]):
        if step >= len(K_ratios):
            continue

        ax = axes[0, idx]
        x = K_ratios[:-step]
        y = K_ratios[step:]

        # 2D histogram
        h = ax.hist2d(x, y, bins=50, range=[[-2, 1], [-2, 1]],
                      cmap='viridis', norm=LogNorm())
        ax.plot([-2, 1], [-2, 1], 'r--', alpha=0.5)  # Diagonal
        ax.axhline(y=-5/6, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=-5/6, color='orange', linestyle=':', alpha=0.5)

        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'Step={step}\nCorr={corr:.4f}', fontsize=10)
        ax.set_xlabel('K(n)/n')
        ax.set_ylabel(f'K(n+{step})/(n+{step})')

    for idx, step in enumerate(steps_to_plot[4:7]):
        if step >= len(K_ratios):
            continue

        ax = axes[1, idx]
        x = K_ratios[:-step]
        y = K_ratios[step:]

        h = ax.hist2d(x, y, bins=50, range=[[-2, 1], [-2, 1]],
                      cmap='viridis', norm=LogNorm())
        ax.plot([-2, 1], [-2, 1], 'r--', alpha=0.5)
        ax.axhline(y=-5/6, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=-5/6, color='orange', linestyle=':', alpha=0.5)

        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'Step={step}\nCorr={corr:.4f}', fontsize=10)
        ax.set_xlabel('K(n)/n')
        ax.set_ylabel(f'K(n+{step})/(n+{step})')

    # Correlation summary plot
    ax = axes[1, 3]
    steps_plot = [c[0] for c in correlations]
    corrs_plot = [c[1] for c in correlations]
    ax.plot(steps_plot, corrs_plot, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Step Size')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation Decay', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, correlations


# ============================================================================
# INVESTIGATION 4: Sophie Germain and Other Prime Patterns
# ============================================================================

def investigate_other_patterns(N: int = 10_000_000):
    """
    Can K-operator detect Sophie Germain primes, prime triplets, etc.?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 4: K-Operator on Other Prime Patterns")
    print("=" * 70)

    sieve = sieve_of_eratosthenes(2 * N + 10)
    sieve_limit = min(N * 3, 100_000_000)
    D = ArithmeticDerivative(sieve_limit)

    # Find patterns
    sg_primes = get_sophie_germain_primes(sieve, N)
    triplets = get_prime_triplets(sieve, N)
    twins = get_twin_primes(sieve, N)

    print(f"\nFound:")
    print(f"  Sophie Germain primes: {len(sg_primes):,}")
    print(f"  Prime triplets: {len(triplets):,}")
    print(f"  Twin primes: {len(twins):,}")

    results = {}

    # Analyze Sophie Germain primes
    # For SG prime p, both p and 2p+1 are prime
    # Analyze K(p)/p
    if len(sg_primes) > 0:
        sg_K = []
        for p in sg_primes:
            K_val = groovy_commutator_K(p, D)
            sg_K.append(K_val / p)
        sg_K = np.array(sg_K)

        results['sophie_germain'] = {
            'name': 'Sophie Germain',
            'count': len(sg_primes),
            'mean': np.mean(sg_K),
            'std': np.std(sg_K),
            'K_ratios': sg_K
        }
        print(f"\nSophie Germain K(p)/p: mean={np.mean(sg_K):.6f}, std={np.std(sg_K):.6f}")

    # Analyze prime triplets (first element)
    if len(triplets) > 0:
        triplet_K = []
        for t in triplets:
            # Analyze the center composite
            center = t[0] + 3  # Midpoint of triplet span
            K_val = groovy_commutator_K(center, D)
            triplet_K.append(K_val / center)
        triplet_K = np.array(triplet_K)

        results['triplets'] = {
            'name': 'Prime Triplets',
            'count': len(triplets),
            'mean': np.mean(triplet_K),
            'std': np.std(triplet_K),
            'K_ratios': triplet_K
        }
        print(f"Prime Triplet centers K/n: mean={np.mean(triplet_K):.6f}, std={np.std(triplet_K):.6f}")

    # Analyze twin prime centers for comparison
    if len(twins) > 0:
        twin_K = []
        for p, p2 in twins:
            center = p + 1
            K_val = groovy_commutator_K(center, D)
            twin_K.append(K_val / center)
        twin_K = np.array(twin_K)

        results['twins'] = {
            'name': 'Twin Prime Centers',
            'count': len(twins),
            'mean': np.mean(twin_K),
            'std': np.std(twin_K),
            'K_ratios': twin_K
        }

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Operator on Various Prime Patterns\n'
                 'Sophie Germain, Prime Triplets, and Twin Primes',
                 fontsize=14, fontweight='bold')

    # Histograms
    ax1 = axes[0, 0]
    if 'sophie_germain' in results:
        ax1.hist(results['sophie_germain']['K_ratios'], bins=80,
                 color='purple', alpha=0.7, label='Sophie Germain K(p)/p',
                 range=(-3, 2), density=True)
    ax1.axvline(x=-0.5, color='gold', linestyle='--', linewidth=2)
    ax1.set_xlabel('K(p)/p')
    ax1.set_ylabel('Density')
    ax1.set_title('Sophie Germain Primes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if 'triplets' in results:
        ax2.hist(results['triplets']['K_ratios'], bins=80,
                 color='green', alpha=0.7, label='Triplet centers',
                 range=(-3, 2), density=True)
    ax2.axvline(x=-0.5, color='gold', linestyle='--', linewidth=2)
    ax2.set_xlabel('K(center)/center')
    ax2.set_ylabel('Density')
    ax2.set_title('Prime Triplets')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Comparison overlay
    ax3 = axes[1, 0]
    if 'twins' in results:
        ax3.hist(results['twins']['K_ratios'], bins=80, color='blue',
                 alpha=0.5, label=f"Twins (n={len(twins):,})",
                 range=(-3, 2), density=True)
    if 'sophie_germain' in results:
        ax3.hist(results['sophie_germain']['K_ratios'], bins=80, color='purple',
                 alpha=0.5, label=f"SG (n={len(sg_primes):,})",
                 range=(-3, 2), density=True)
    ax3.axvline(x=-0.5, color='gold', linestyle='--', linewidth=2)
    ax3.set_xlabel('K/n')
    ax3.set_ylabel('Density')
    ax3.set_title('Comparison: Twins vs Sophie Germain')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    for key in ['twins', 'sophie_germain', 'triplets']:
        if key in results:
            r = results[key]
            table_data.append([r['name'], f"{r['count']:,}",
                               f"{r['mean']:.6f}", f"{r['std']:.6f}"])

    if table_data:
        table = ax4.table(cellText=table_data,
                          colLabels=['Pattern', 'Count', 'Mean K/n', 'Std'],
                          loc='center',
                          cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
    ax4.set_title('Pattern Comparison Summary', fontsize=12, pad=20)

    plt.tight_layout()

    return fig, results


# ============================================================================
# INVESTIGATION 5: Physical Meaning of Spectrum
# ============================================================================

def investigate_spectral_physics(N: int = 10_000_000):
    """
    Investigate physical interpretations of the K-spectrum.
    Compare to known physical spectra (hydrogen, harmonic oscillator, etc.)
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 5: Physical Meaning of the Spectrum")
    print("=" * 70)

    sieve = sieve_of_eratosthenes(N + 10)
    sieve_limit = min(N * 3, 100_000_000)
    D = ArithmeticDerivative(sieve_limit)

    twins = get_twin_primes(sieve, N)

    K_ratios = []
    for p, p2 in twins:
        center = p + 1
        K_val = groovy_commutator_K(center, D)
        K_ratios.append(K_val / center)

    K_ratios = np.array(K_ratios)

    # Find peak locations
    print("\nSpectral Peak Analysis:")
    print("-" * 50)

    # High-resolution histogram
    counts, bin_edges = np.histogram(K_ratios, bins=500, range=(-3, 2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find local maxima
    peaks = []
    for i in range(2, len(counts) - 2):
        if (counts[i] > counts[i-1] and counts[i] > counts[i+1] and
            counts[i] > counts[i-2] and counts[i] > counts[i+2]):
            peaks.append((bin_centers[i], counts[i]))

    # Sort by count
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Compare to theoretical spectra
    print("\nTop 10 peaks vs theoretical models:")
    print(f"{'Rank':<6}{'Peak':<12}{'Hydrogen':<15}{'Harmonic':<15}{'Match?':<12}")
    print("-" * 60)

    # Hydrogen-like: E_n = -1/n²
    hydrogen_levels = [-1/n**2 for n in range(1, 11)]

    # Harmonic oscillator: E_n = (n + 1/2), shifted
    harmonic_levels = [-(n + 0.5)/6 for n in range(10)]  # Scaled to match range

    peak_analysis = []
    for i, (center, count) in enumerate(peaks[:10], 1):
        # Find nearest hydrogen level
        h_dist = min(abs(center - h) for h in hydrogen_levels)
        h_match = min(hydrogen_levels, key=lambda h: abs(center - h))

        # Find nearest harmonic level
        harm_dist = min(abs(center - h) for h in harmonic_levels)
        harm_match = min(harmonic_levels, key=lambda h: abs(center - h))

        match_type = ""
        if h_dist < 0.05:
            match_type = "HYDROGEN"
        elif harm_dist < 0.05:
            match_type = "HARMONIC"

        print(f"{i:<6}{center:<12.4f}{h_match:<15.4f}{harm_match:<15.4f}{match_type:<12}")

        peak_analysis.append({
            'peak': center,
            'count': count,
            'nearest_hydrogen': h_match,
            'h_distance': h_dist,
            'nearest_harmonic': harm_match,
            'harm_distance': harm_dist
        })

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Physical Interpretation of K-Spectrum\n'
                 'Comparison with Hydrogen and Harmonic Oscillator Energy Levels',
                 fontsize=14, fontweight='bold')

    # Panel 1: K-spectrum with hydrogen levels overlaid
    ax1 = axes[0, 0]
    ax1.hist(K_ratios, bins=200, range=(-2, 0.5), color='steelblue',
             alpha=0.7, density=True)

    for n in range(1, 6):
        E = -1/n**2
        ax1.axvline(x=E, color='red', linestyle='--', alpha=0.7,
                    label=f'H n={n}' if n <= 3 else None)

    ax1.set_xlabel('K(n)/n')
    ax1.set_ylabel('Density')
    ax1.set_title('K-Spectrum vs Hydrogen Levels\n(E_n = -1/n²)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: With harmonic oscillator levels
    ax2 = axes[0, 1]
    ax2.hist(K_ratios, bins=200, range=(-2, 0.5), color='steelblue',
             alpha=0.7, density=True)

    for n in range(6):
        E = -(n + 0.5)/6
        ax2.axvline(x=E, color='green', linestyle=':', alpha=0.7,
                    label=f'HO n={n}' if n <= 3 else None)

    ax2.set_xlabel('K(n)/n')
    ax2.set_ylabel('Density')
    ax2.set_title('K-Spectrum vs Harmonic Oscillator\n(E_n = -(n+1/2)/6)', fontsize=11)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: With simple fractions
    ax3 = axes[1, 0]
    ax3.hist(K_ratios, bins=200, range=(-1.5, 0), color='steelblue',
             alpha=0.7, density=True)

    fractions = [(-5/6, '-5/6', 'gold'), (-1, '-1', 'red'),
                 (-3/4, '-3/4', 'orange'), (-2/3, '-2/3', 'yellow'),
                 (-1/2, '-1/2', 'lime'), (-1/3, '-1/3', 'cyan')]

    for val, label, color in fractions:
        ax3.axvline(x=val, color=color, linestyle='--', linewidth=2, alpha=0.8)

    ax3.set_xlabel('K(n)/n')
    ax3.set_ylabel('Density')
    ax3.set_title('K-Spectrum vs Simple Fractions', fontsize=11)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Level spacing distribution
    ax4 = axes[1, 1]

    # Compute level spacings (normalized)
    sorted_K = np.sort(K_ratios)
    spacings = np.diff(sorted_K)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing

    # Histogram of spacings
    ax4.hist(normalized_spacings, bins=100, range=(0, 3),
             color='purple', alpha=0.7, density=True, label='Observed')

    # Theoretical distributions
    s = np.linspace(0.001, 3, 100)
    poisson = np.exp(-s)  # Poisson (random)
    goe = (np.pi/2) * s * np.exp(-np.pi * s**2 / 4)  # GOE (fermion)

    ax4.plot(s, poisson, 'g--', linewidth=2, label='Poisson (random)')
    ax4.plot(s, goe, 'r--', linewidth=2, label='GOE (fermion)')

    ax4.set_xlabel('Normalized Spacing s')
    ax4.set_ylabel('P(s)')
    ax4.set_title('Level Spacing Distribution\n(Bosonic if neither Poisson nor GOE)', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Compute specific statistics
    P_small = np.mean(normalized_spacings < 0.1)
    print(f"\n\nLevel Spacing Statistics:")
    print(f"  P(s < 0.1) = {P_small:.4f}")
    print(f"  Poisson expected: 0.095")
    print(f"  GOE expected: 0.004")
    print(f"  {'BOSONIC clustering detected!' if P_small > 0.3 else 'Distribution type unclear'}")

    return fig, peak_analysis


# ============================================================================
# INVESTIGATION 6: Closed-Form Analysis
# ============================================================================

def investigate_closed_form(N: int = 1_000_000):
    """
    Develop understanding toward closed-form proof for K(n)/n → -1/2.
    Analyze the algebraic structure of K for 6-multiples.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION 6: Toward a Closed-Form Proof")
    print("=" * 70)

    sieve = sieve_of_eratosthenes(N + 10)
    sieve_limit = min(N * 3, 20_000_000)
    D = ArithmeticDerivative(sieve_limit)

    print("\nAnalyzing the structure of K for multiples of 6...")

    # For n = 6k, analyze the components of K
    # K(n) = D(n + D(n)) - D(n) - D(D(n))

    analysis_data = []

    for k in range(1, min(10000, N // 6)):
        n = 6 * k

        Dn = D.derivative(n)
        DDn = D.derivative(Dn)
        D_sum = D.derivative(n + Dn)
        K = D_sum - Dn - DDn

        # Theoretical: D(6k) = 6k × (1/2 + 1/3 + Σ e_i/p_i for k's factors)
        # Base contribution: 6k × (1/2 + 1/3) = 6k × 5/6 = 5k
        theoretical_base = 5 * k
        epsilon = Dn - theoretical_base

        analysis_data.append({
            'k': k,
            'n': n,
            'Dn': Dn,
            'DDn': DDn,
            'D_sum': D_sum,
            'K': K,
            'K_ratio': K / n,
            'epsilon': epsilon,
            'Dn_over_n': Dn / n
        })

    df = analysis_data

    # Statistical analysis
    K_ratios = np.array([d['K_ratio'] for d in df])
    Dn_over_n = np.array([d['Dn_over_n'] for d in df])
    epsilons = np.array([d['epsilon'] for d in df])

    print(f"\nFor n = 6k (k from 1 to {len(df)}):")
    print(f"  Mean K(n)/n: {np.mean(K_ratios):.6f}")
    print(f"  Mean D(n)/n: {np.mean(Dn_over_n):.6f} (theoretical 5/6 = {5/6:.6f})")
    print(f"  Mean ε(k) = D(6k) - 5k: {np.mean(epsilons):.4f}")

    # Check when K/n = exactly -5/6
    exact_count = np.sum(np.abs(K_ratios + 5/6) < 0.0001)
    print(f"\n  Exact -5/6 cases: {exact_count} ({exact_count/len(df)*100:.2f}%)")

    # Analyze what makes K/n = -5/6
    exact_indices = np.where(np.abs(K_ratios + 5/6) < 0.0001)[0]

    print("\n  Sample of exact -5/6 cases:")
    print(f"  {'k':<8}{'n':<10}{'D(n)':<12}{'D(D(n))':<12}{'D(n+D(n))':<12}")
    print("-" * 54)

    for idx in exact_indices[:10]:
        d = df[idx]
        print(f"  {d['k']:<8}{d['n']:<10}{d['Dn']:<12}{d['DDn']:<12}{d['D_sum']:<12}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algebraic Structure of K for Multiples of 6\n'
                 'Toward Understanding K(n)/n → -5/6 (and mean → -1/2)',
                 fontsize=14, fontweight='bold')

    ks = np.array([d['k'] for d in df])

    # Panel 1: K(n)/n vs k
    ax1 = axes[0, 0]
    ax1.scatter(ks[:2000], K_ratios[:2000], s=1, alpha=0.5, c='blue')
    ax1.axhline(y=-5/6, color='gold', linestyle='--', linewidth=2, label='-5/6 (mode)')
    ax1.axhline(y=-1/2, color='red', linestyle=':', linewidth=2, label='-1/2 (mean)')
    ax1.axhline(y=np.mean(K_ratios), color='green', linestyle='-', linewidth=1, label=f'Observed mean')
    ax1.set_xlabel('k (where n = 6k)')
    ax1.set_ylabel('K(n)/n')
    ax1.set_title('K(n)/n for n = 6k')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-3, 2)

    # Panel 2: D(n)/n vs k
    ax2 = axes[0, 1]
    ax2.scatter(ks[:2000], Dn_over_n[:2000], s=1, alpha=0.5, c='green')
    ax2.axhline(y=5/6, color='gold', linestyle='--', linewidth=2, label='5/6 (base)')
    ax2.set_xlabel('k')
    ax2.set_ylabel('D(n)/n')
    ax2.set_title(f'D(6k)/(6k) — converges to 5/6 + O(1/k)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: ε(k) = D(6k) - 5k
    ax3 = axes[1, 0]
    ax3.scatter(ks[:2000], epsilons[:2000], s=1, alpha=0.5, c='purple')
    ax3.axhline(y=0, color='gray', linestyle='--')
    ax3.set_xlabel('k')
    ax3.set_ylabel('ε(k) = D(6k) - 5k')
    ax3.set_title('Deviation from 5k formula')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Histogram of K/n
    ax4 = axes[1, 1]
    ax4.hist(K_ratios, bins=100, range=(-3, 2), color='steelblue', alpha=0.7)
    ax4.axvline(x=-5/6, color='gold', linestyle='--', linewidth=2, label='-5/6')
    ax4.axvline(x=-1/2, color='red', linestyle=':', linewidth=2, label='-1/2')
    ax4.axvline(x=np.mean(K_ratios), color='green', linestyle='-', linewidth=2,
                label=f'mean={np.mean(K_ratios):.4f}')
    ax4.set_xlabel('K(n)/n')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of K(n)/n for n = 6k')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Derive the asymptotic formula
    print("\n" + "=" * 50)
    print("ASYMPTOTIC ANALYSIS")
    print("=" * 50)

    print("""
For n = 6k:

1. D(6k) = 6k × Σ(e_p / p) over prime factorization of 6k
         = 6k × (1/2 + 1/3 + contributions from k)
         = 6k × (5/6 + O(log(k)/k))
         = 5k + O(log k)

2. Let D(n) = 5k + ε where ε = O(log k)

   n + D(n) = 6k + 5k + ε = 11k + ε

3. D(11k + ε) ≈ (11k) × (sum of 1/p for primes of 11k)
              ≈ 11k × O(log log k / log k) for "random" 11k

4. D(D(n)) = D(5k + ε) ≈ (5k) × (sum of 1/p for primes of 5k)
           ≈ 5k × O(log log k / log k)

5. K(n) = D(n + D(n)) - D(n) - D(D(n))
        ≈ D(11k) - 5k - D(5k)

   For large k with 11k and 5k having "typical" factorizations:
   K(n) ≈ -(5k) + O(log k)

   K(n)/n = K(6k)/(6k) ≈ -5k/(6k) = -5/6

The -5/6 ground state arises because:
- D(n) contributes +5k
- The K formula subtracts D(n), giving -5k
- Other terms are lower order

The -1/2 MEAN arises because the distribution is asymmetric:
- Ground state at -5/6 (most frequent)
- Some values above -5/6 (when D(n+D(n)) is large)
- Fewer values below -5/6
- Weighted average ≈ -1/2
""")

    return fig, df


# ============================================================================
# MAIN: Run All Investigations
# ============================================================================

def run_all_investigations(N: int = 10_000_000, output_dir: str = None):
    """Run all open question investigations and save results."""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    print("\n" + "=" * 70)
    print("FEARFUL SYMMETRY: OPEN QUESTIONS INVESTIGATION")
    print("=" * 70)
    print(f"\nMax N: {N:,}")
    print(f"Output directory: {output_dir}")

    # Investigation 1: Beyond 10M
    start = time.time()
    N_values = [1_000_000, 5_000_000, 10_000_000]
    if N >= 25_000_000:
        N_values.append(25_000_000)
    if N >= 50_000_000:
        N_values.append(50_000_000)

    fig1, results['beyond_10M'] = investigate_beyond_10M(N_values)
    fig1.savefig(os.path.join(output_dir, 'convergence_beyond_10M.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: convergence_beyond_10M.png] ({time.time()-start:.1f}s)")
    plt.close(fig1)

    # Investigation 2: Prime gaps
    start = time.time()
    fig2, results['prime_gaps'] = investigate_prime_gaps(N)
    fig2.savefig(os.path.join(output_dir, 'prime_gap_generalization.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: prime_gap_generalization.png] ({time.time()-start:.1f}s)")
    plt.close(fig2)

    # Investigation 3: Phase space
    start = time.time()
    fig3, results['phase_space'] = investigate_phase_space_steps(N)
    fig3.savefig(os.path.join(output_dir, 'phase_space_steps.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: phase_space_steps.png] ({time.time()-start:.1f}s)")
    plt.close(fig3)

    # Investigation 4: Other patterns
    start = time.time()
    fig4, results['other_patterns'] = investigate_other_patterns(N)
    fig4.savefig(os.path.join(output_dir, 'other_prime_patterns.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: other_prime_patterns.png] ({time.time()-start:.1f}s)")
    plt.close(fig4)

    # Investigation 5: Physical meaning
    start = time.time()
    fig5, results['spectral_physics'] = investigate_spectral_physics(N)
    fig5.savefig(os.path.join(output_dir, 'spectral_physics.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: spectral_physics.png] ({time.time()-start:.1f}s)")
    plt.close(fig5)

    # Investigation 6: Closed form
    start = time.time()
    fig6, results['closed_form'] = investigate_closed_form(min(N, 1_000_000))
    fig6.savefig(os.path.join(output_dir, 'closed_form_analysis.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[Saved: closed_form_analysis.png] ({time.time()-start:.1f}s)")
    plt.close(fig6)

    print("\n" + "=" * 70)
    print("ALL INVESTIGATIONS COMPLETE")
    print("=" * 70)

    return results


if __name__ == '__main__':
    # Run with configurable N
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    results = run_all_investigations(N=N)
