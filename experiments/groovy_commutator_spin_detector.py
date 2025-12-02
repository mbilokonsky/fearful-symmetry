#!/usr/bin/env python3
"""
SPIN DETECTOR: Is -0.5 unique to Twin Primes?

Control Experiment:
    Group A: Twin Prime Centers (p+1) where (p, p+2) are twin primes
    Group B: Random multiples of 6 that are NOT twin prime centers
    Group C: Random integers (uniform)

Question: Does K(n)/n → -0.5 for all multiples of 6, or is it special to twins?

Bonus: Level Spacing Analysis
    - GUE Statistics (Gaussian Unitary Ensemble) → Level repulsion (Fermion-like)
    - Poisson Statistics → Random clustering (Boson-like)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Tuple, Set
import time

# ============================================================================
# Core Functions (from previous experiments)
# ============================================================================

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve


def get_twin_prime_centers(n: int) -> Set[int]:
    """Return set of p+1 for all twin primes (p, p+2) up to n."""
    sieve = sieve_of_eratosthenes(n + 2)
    centers = set()
    for p in range(3, n):
        if sieve[p] and sieve[p + 2]:
            centers.add(p + 1)
    return centers


class ArithmeticDerivative:
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
    """K(n) = D(n + D(n)) - (D(n) + D(D(n)))"""
    Dn = D.derivative(n)
    DDn = D.derivative(Dn)
    D_n_plus_Dn = D.derivative(n + Dn)
    return D_n_plus_Dn - (Dn + DDn)


# ============================================================================
# Control Group Generation
# ============================================================================

def generate_groups(N: int = 10_000_000, sample_size: int = 50_000):
    """
    Generate three groups:
        A: Twin prime centers (p+1)
        B: Random multiples of 6 NOT in twin prime centers
        C: Random integers
    """
    print("=" * 70)
    print("SPIN DETECTOR: Generating Control Groups")
    print("=" * 70)

    # Get twin prime centers
    print("\n[1/4] Finding twin prime centers...")
    twin_centers = get_twin_prime_centers(N)
    print(f"       Found {len(twin_centers):,} twin prime centers")

    # Group A: Sample from twin centers
    group_A = np.array(sorted(twin_centers))
    if len(group_A) > sample_size:
        # Sample uniformly across the range
        indices = np.linspace(0, len(group_A) - 1, sample_size, dtype=int)
        group_A = group_A[indices]
    print(f"       Group A: {len(group_A):,} twin prime centers")

    # Group B: Random multiples of 6 NOT in twin centers
    print("\n[2/4] Generating random multiples of 6 (not twin centers)...")

    # All multiples of 6 in range
    all_6_multiples = set(range(6, N, 6))
    # Remove twin centers
    non_twin_6_multiples = list(all_6_multiples - twin_centers)

    print(f"       Total multiples of 6: {len(all_6_multiples):,}")
    print(f"       Non-twin multiples of 6: {len(non_twin_6_multiples):,}")

    # Random sample
    np.random.seed(42)  # Reproducibility
    group_B = np.random.choice(non_twin_6_multiples, size=min(sample_size, len(non_twin_6_multiples)), replace=False)
    group_B = np.sort(group_B)
    print(f"       Group B: {len(group_B):,} random 6-multiples (non-twin)")

    # Group C: Random integers (not multiples of 6, not primes - just random composites)
    print("\n[3/4] Generating random integers...")

    # Generate random integers in range, excluding very small numbers
    candidates = np.random.randint(100, N, size=sample_size * 3)
    # Remove duplicates
    candidates = np.unique(candidates)
    # Sample
    group_C = np.random.choice(candidates, size=min(sample_size, len(candidates)), replace=False)
    group_C = np.sort(group_C)
    print(f"       Group C: {len(group_C):,} random integers")

    # Initialize derivative calculator
    print("\n[4/4] Initializing arithmetic derivative engine...")
    sieve_limit = min(N * 3, 50_000_000)
    D = ArithmeticDerivative(sieve_limit)

    return group_A, group_B, group_C, D


def compute_K_ratios(group: np.ndarray, D: ArithmeticDerivative, name: str) -> np.ndarray:
    """Compute K(n)/n for all n in group."""
    print(f"       Computing K(n)/n for {name}...")
    ratios = []
    for i, n in enumerate(group):
        if n > 0:
            K_val = groovy_commutator_K(int(n), D)
            ratios.append(K_val / n)
        if (i + 1) % 10000 == 0:
            print(f"         {i+1:,} / {len(group):,}")
    return np.array(ratios)


# ============================================================================
# Level Spacing Analysis (GUE vs Poisson)
# ============================================================================

def analyze_level_spacing(K_ratios: np.ndarray, name: str):
    """
    Analyze level spacing statistics.

    GUE (Wigner-Dyson): P(s) ~ s * exp(-π*s²/4)  - Level repulsion at s=0
    Poisson: P(s) ~ exp(-s)                       - No repulsion, P(0) > 0

    We look at the distribution of spacings between consecutive K/n values.
    """
    # Sort the K/n values
    sorted_vals = np.sort(K_ratios)

    # Compute spacings
    spacings = np.diff(sorted_vals)

    # Remove zero spacings (exact duplicates)
    spacings = spacings[spacings > 0]

    # Normalize by mean spacing (unfold the spectrum)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing

    # Statistics
    results = {
        'name': name,
        'n_spacings': len(normalized_spacings),
        'mean_spacing': mean_spacing,
        'spacings': normalized_spacings,
    }

    # Compute P(s < 0.1) - probability of very small spacings
    # For GUE this should be near 0 (repulsion), for Poisson ~0.1
    small_spacing_prob = np.mean(normalized_spacings < 0.1)
    results['P_small'] = small_spacing_prob

    # Fit to Wigner surmise (GUE) and Poisson
    # GUE: P(s) = (π/2) * s * exp(-π*s²/4)
    # Poisson: P(s) = exp(-s)

    s_vals = normalized_spacings[normalized_spacings < 3]  # Truncate for fitting

    # Wigner surmise (GUE) PDF
    def wigner_pdf(s):
        return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    # Poisson PDF
    def poisson_pdf(s):
        return np.exp(-s)

    # Compute KS test statistics
    # (Note: these are approximate since we're comparing to theoretical distributions)
    results['wigner_ks'], _ = stats.kstest(s_vals, lambda x: 1 - np.exp(-np.pi * x**2 / 4))
    results['poisson_ks'], _ = stats.kstest(s_vals, 'expon')

    return results


# ============================================================================
# Main Analysis
# ============================================================================

def run_spin_detector(N: int = 10_000_000, sample_size: int = 50_000):
    """Run the full spin detector experiment."""

    print("\n" + "=" * 70)
    print("SPIN DETECTOR EXPERIMENT")
    print("=" * 70)
    print(f"N = {N:,}, Sample size = {sample_size:,}\n")

    # Generate groups
    group_A, group_B, group_C, D = generate_groups(N, sample_size)

    # Compute K(n)/n for each group
    print("\n" + "-" * 50)
    print("Computing K(n)/n ratios...")
    print("-" * 50)

    t0 = time.time()
    K_ratio_A = compute_K_ratios(group_A, D, "Group A (Twin Centers)")
    K_ratio_B = compute_K_ratios(group_B, D, "Group B (Random 6-multiples)")
    K_ratio_C = compute_K_ratios(group_C, D, "Group C (Random integers)")
    print(f"Completed in {time.time() - t0:.1f}s")

    # ========================================================================
    # Statistical Comparison
    # ========================================================================

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    groups = [
        ("A: Twin Prime Centers", K_ratio_A),
        ("B: Random 6-multiples", K_ratio_B),
        ("C: Random Integers", K_ratio_C),
    ]

    print(f"\n{'Group':<25} {'Mean':>12} {'Median':>12} {'Std':>12} {'|Mean+0.5|':>12}")
    print("-" * 75)

    for name, ratios in groups:
        mean_val = np.mean(ratios)
        median_val = np.median(ratios)
        std_val = np.std(ratios)
        diff_from_half = abs(mean_val + 0.5)
        print(f"{name:<25} {mean_val:>12.6f} {median_val:>12.6f} {std_val:>12.6f} {diff_from_half:>12.6f}")

    # Two-sample t-tests
    print("\n" + "-" * 50)
    print("Two-Sample T-Tests (is the mean different?)")
    print("-" * 50)

    t_AB, p_AB = stats.ttest_ind(K_ratio_A, K_ratio_B)
    t_AC, p_AC = stats.ttest_ind(K_ratio_A, K_ratio_C)
    t_BC, p_BC = stats.ttest_ind(K_ratio_B, K_ratio_C)

    print(f"A vs B: t = {t_AB:.4f}, p = {p_AB:.2e}")
    print(f"A vs C: t = {t_AC:.4f}, p = {p_AC:.2e}")
    print(f"B vs C: t = {t_BC:.4f}, p = {p_BC:.2e}")

    # Check for 1/137 (fine structure constant)
    print("\n" + "-" * 50)
    print("Checking for 1/137 (Fine Structure Constant α ≈ 0.00730)")
    print("-" * 50)

    alpha = 1/137  # ~0.007299

    for name, ratios in groups:
        mean_val = np.mean(ratios)
        # Various transformations that might yield 1/137
        checks = [
            ("|mean|", abs(mean_val)),
            ("|mean + 0.5|", abs(mean_val + 0.5)),
            ("|mean| / 68.5", abs(mean_val) / 68.5),  # ~0.5/68.5
            ("std / 274", np.std(ratios) / 274),
        ]
        for check_name, val in checks:
            if abs(val - alpha) < 0.001:
                print(f"  POSSIBLE MATCH in {name}: {check_name} = {val:.6f} (α = {alpha:.6f})")

    # ========================================================================
    # Level Spacing Analysis
    # ========================================================================

    print("\n" + "=" * 70)
    print("LEVEL SPACING ANALYSIS (GUE vs Poisson)")
    print("=" * 70)

    spacing_results = []
    for name, ratios in groups:
        result = analyze_level_spacing(ratios, name)
        spacing_results.append(result)

        print(f"\n{name}:")
        print(f"  P(s < 0.1) = {result['P_small']:.4f}")
        print(f"  (GUE predicts ~0.004, Poisson predicts ~0.095)")
        print(f"  Wigner KS stat: {result['wigner_ks']:.4f}")
        print(f"  Poisson KS stat: {result['poisson_ks']:.4f}")

    # ========================================================================
    # Create Plots
    # ========================================================================

    print("\n" + "-" * 50)
    print("Generating plots...")
    print("-" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Spin Detector: Is -0.5 Unique to Twin Primes?', fontsize=14, fontweight='bold')

    # Plot 1: Histogram comparison
    ax1 = axes[0, 0]
    bins = np.linspace(-5, 5, 80)

    ax1.hist(np.clip(K_ratio_A, -5, 5), bins=bins, alpha=0.5, density=True,
             label=f'A: Twin Centers (μ={np.mean(K_ratio_A):.3f})', color='blue')
    ax1.hist(np.clip(K_ratio_B, -5, 5), bins=bins, alpha=0.5, density=True,
             label=f'B: Random 6-mult (μ={np.mean(K_ratio_B):.3f})', color='red')
    ax1.hist(np.clip(K_ratio_C, -5, 5), bins=bins, alpha=0.5, density=True,
             label=f'C: Random int (μ={np.mean(K_ratio_C):.3f})', color='green')

    ax1.axvline(x=-0.5, color='black', linestyle='--', linewidth=2, label='-0.5')
    ax1.set_xlabel('K(n)/n')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    ax2 = axes[0, 1]
    # Clip for visualization
    data_for_box = [
        np.clip(K_ratio_A, -10, 10),
        np.clip(K_ratio_B, -10, 10),
        np.clip(K_ratio_C, -10, 10)
    ]
    bp = ax2.boxplot(data_for_box, labels=['A: Twin\nCenters', 'B: Random\n6-mult', 'C: Random\nIntegers'],
                     patch_artist=True)
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.axhline(y=-0.5, color='black', linestyle='--', linewidth=2)
    ax2.set_ylabel('K(n)/n')
    ax2.set_title('Box Plot Comparison')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Level spacing for Group A (Twin Centers)
    ax3 = axes[1, 0]
    spacings_A = spacing_results[0]['spacings']
    spacings_A_clipped = spacings_A[spacings_A < 4]

    ax3.hist(spacings_A_clipped, bins=50, density=True, alpha=0.7, label='Twin Centers')

    # Overlay theoretical distributions
    s_theory = np.linspace(0.001, 4, 200)
    wigner = (np.pi / 2) * s_theory * np.exp(-np.pi * s_theory**2 / 4)
    poisson = np.exp(-s_theory)

    ax3.plot(s_theory, wigner, 'r-', linewidth=2, label='GUE (Wigner)')
    ax3.plot(s_theory, poisson, 'g--', linewidth=2, label='Poisson')

    ax3.set_xlabel('Normalized spacing s')
    ax3.set_ylabel('P(s)')
    ax3.set_title('Level Spacing: Twin Prime Centers\n(GUE = Fermion-like, Poisson = Random)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 4)

    # Plot 4: Mean convergence over sample size
    ax4 = axes[1, 1]

    # Running mean for each group
    for name, ratios, color in [("A: Twin Centers", K_ratio_A, 'blue'),
                                  ("B: 6-multiples", K_ratio_B, 'red'),
                                  ("C: Random", K_ratio_C, 'green')]:
        running_mean = np.cumsum(ratios) / np.arange(1, len(ratios) + 1)
        ax4.plot(running_mean, label=name, color=color, alpha=0.7)

    ax4.axhline(y=-0.5, color='black', linestyle='--', linewidth=2, label='-0.5')
    ax4.set_xlabel('Sample index')
    ax4.set_ylabel('Running mean of K(n)/n')
    ax4.set_title('Convergence of Mean')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-1.5, 0.5)

    plt.tight_layout()

    # Save plot
    import os
    output_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(output_dir), 'images')
    os.makedirs(images_dir, exist_ok=True)

    output_path = os.path.join(images_dir, 'spin_detector_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # ========================================================================
    # Final Verdict
    # ========================================================================

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    mean_A = np.mean(K_ratio_A)
    mean_B = np.mean(K_ratio_B)
    mean_C = np.mean(K_ratio_C)

    print(f"""
GROUP MEANS:
    A (Twin Centers):     {mean_A:.6f}  (diff from -0.5: {abs(mean_A + 0.5):.6f})
    B (Random 6-mult):    {mean_B:.6f}  (diff from -0.5: {abs(mean_B + 0.5):.6f})
    C (Random integers):  {mean_C:.6f}  (diff from -0.5: {abs(mean_C + 0.5):.6f})
""")

    # Determine if -0.5 is unique to twins
    threshold = 0.05  # 5% of 0.5

    if abs(mean_A + 0.5) < threshold and abs(mean_B + 0.5) > threshold:
        print("RESULT: The -0.5 signal is UNIQUE to Twin Prime Centers!")
        print("        Random multiples of 6 show a DIFFERENT mean.")
        signal_type = "UNIQUE"
    elif abs(mean_A + 0.5) < threshold and abs(mean_B + 0.5) < threshold:
        print("RESULT: The -0.5 signal appears in ALL multiples of 6.")
        print("        It's a property of 6-divisibility, not twin primes specifically.")
        signal_type = "6-DIVISIBILITY"
    else:
        print("RESULT: The -0.5 signal is WEAK or ABSENT in both groups.")
        signal_type = "INCONCLUSIVE"

    # Level spacing verdict
    P_small_A = spacing_results[0]['P_small']

    if P_small_A < 0.03:
        level_type = "GUE-like (FERMION)"
    elif P_small_A > 0.07:
        level_type = "POISSON-like (RANDOM)"
    else:
        level_type = "INTERMEDIATE"

    print(f"""
LEVEL SPACING:
    P(s < 0.1) for Twin Centers = {P_small_A:.4f}
    → Classification: {level_type}

    (GUE/Fermion: P(s<0.1) ≈ 0.004, levels repel)
    (Poisson/Random: P(s<0.1) ≈ 0.095, levels cluster)
""")

    return {
        'K_ratio_A': K_ratio_A,
        'K_ratio_B': K_ratio_B,
        'K_ratio_C': K_ratio_C,
        'spacing_results': spacing_results,
        'signal_type': signal_type,
        'level_type': level_type
    }


if __name__ == '__main__':
    results = run_spin_detector(N=10_000_000, sample_size=50_000)
