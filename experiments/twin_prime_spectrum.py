#!/usr/bin/env python3
"""
TWIN PRIME SPECTRUM: Visualizing the Bosonic Condensate

Hypothesis: K(n)/n for twin prime centers doesn't form a smooth distribution.
Instead, it should show DISCRETE SPECTRAL LINES where values "stack up"
like energy levels in a quantum system.

Visualizations:
1. High-resolution density plot (500 bins) to reveal spectral structure
2. Peak identification with rational number matching
3. 2D "Golden Cloud" (n vs K(n)/n with density coloring)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fractions import Fraction
from typing import List, Tuple, Set
import time

# ============================================================================
# Core Functions
# ============================================================================

def sieve_of_eratosthenes(n: int) -> np.ndarray:
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve


def get_twin_primes(n: int) -> List[Tuple[int, int]]:
    sieve = sieve_of_eratosthenes(n + 2)
    twins = []
    for p in range(3, n):
        if sieve[p] and sieve[p + 2]:
            twins.append((p, p + 2))
    return twins


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
    Dn = D.derivative(n)
    DDn = D.derivative(Dn)
    D_n_plus_Dn = D.derivative(n + Dn)
    return D_n_plus_Dn - (Dn + DDn)


# ============================================================================
# Rational Number Matching
# ============================================================================

def find_nearest_rational(x: float, max_denom: int = 12) -> Tuple[Fraction, float]:
    """Find the nearest simple rational number to x."""
    best_frac = None
    best_diff = float('inf')

    for denom in range(1, max_denom + 1):
        numer = round(x * denom)
        frac = Fraction(numer, denom)
        diff = abs(float(frac) - x)
        if diff < best_diff:
            best_diff = diff
            best_frac = frac

    return best_frac, best_diff


def identify_peaks(values: np.ndarray, n_bins: int = 500, top_n: int = 10):
    """
    Identify peaks in the distribution.
    Returns list of (bin_center, count, nearest_rational, error)
    """
    # Focus on the main region
    v_min, v_max = -3, 2
    values_clipped = values[(values >= v_min) & (values <= v_max)]

    counts, bin_edges = np.histogram(values_clipped, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Find local maxima (peaks)
    peaks = []
    for i in range(2, len(counts) - 2):
        if (counts[i] > counts[i-1] and counts[i] > counts[i+1] and
            counts[i] > counts[i-2] and counts[i] > counts[i+2]):
            center = bin_centers[i]
            frac, err = find_nearest_rational(center, max_denom=12)
            peaks.append((center, counts[i], frac, err))

    # Sort by count (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)

    return peaks[:top_n], bin_centers, counts


# ============================================================================
# Main Visualization
# ============================================================================

def create_twin_prime_spectrum(N: int = 10_000_000):
    """Generate twin prime data and create spectral visualizations."""

    print("=" * 70)
    print("TWIN PRIME SPECTRUM: Visualizing the Bosonic Condensate")
    print("=" * 70)

    # Initialize
    print("\n[1/4] Initializing...")
    sieve_limit = min(N * 3, 50_000_000)
    D = ArithmeticDerivative(sieve_limit)

    # Get twin primes
    print("[2/4] Finding twin primes...")
    twin_primes = get_twin_primes(N)
    print(f"       Found {len(twin_primes):,} twin prime pairs")

    # Compute K(n)/n values
    print("[3/4] Computing K(p+1)/(p+1) for all twins...")

    n_values = []  # The composite p+1
    K_ratios = []  # K(p+1)/(p+1)
    K_raw = []     # Raw K(p+1) values

    for i, (p, p2) in enumerate(twin_primes):
        composite = p + 1
        K_val = groovy_commutator_K(composite, D)
        ratio = K_val / composite

        n_values.append(composite)
        K_ratios.append(ratio)
        K_raw.append(K_val)

        if (i + 1) % 20000 == 0:
            print(f"       {i+1:,} / {len(twin_primes):,}")

    n_values = np.array(n_values)
    K_ratios = np.array(K_ratios)
    K_raw = np.array(K_raw)

    print("[4/4] Generating visualizations...")

    # ========================================================================
    # FIGURE 1: High-Resolution Spectral Density
    # ========================================================================

    fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
    fig1.suptitle('Twin Prime Spectrum: K(p+1)/(p+1) Distribution\n'
                  'Searching for Discrete Quantum States',
                  fontsize=14, fontweight='bold')

    # Top panel: Full 500-bin histogram
    ax1 = axes1[0]
    n_bins = 500

    # Focus on main region
    v_min, v_max = -3, 2
    K_main = K_ratios[(K_ratios >= v_min) & (K_ratios <= v_max)]

    counts, bin_edges, patches = ax1.hist(K_main, bins=n_bins,
                                           color='steelblue', alpha=0.8,
                                           edgecolor='none')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find and label peaks
    peaks, _, _ = identify_peaks(K_ratios, n_bins=n_bins, top_n=10)

    print("\n" + "=" * 50)
    print("TOP 10 SPECTRAL PEAKS")
    print("=" * 50)
    print(f"{'Rank':<6}{'Value':>12}{'Count':>10}{'Rational':>12}{'Error':>12}")
    print("-" * 52)

    for rank, (center, count, frac, err) in enumerate(peaks[:10], 1):
        print(f"{rank:<6}{center:>12.6f}{count:>10}{str(frac):>12}{err:>12.6f}")

        # Mark top 3 on plot
        if rank <= 3:
            ax1.axvline(x=center, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax1.annotate(f'#{rank}: {frac}\n({center:.4f})',
                        xy=(center, count),
                        xytext=(center + 0.15, count * 1.1),
                        fontsize=9, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    ax1.axvline(x=-0.5, color='gold', linestyle='-', linewidth=2, label='-1/2', alpha=0.8)
    ax1.set_xlabel('K(p+1)/(p+1)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'High-Resolution Histogram ({n_bins} bins)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Zoomed view around -1 to 0
    ax2 = axes1[1]
    K_zoom = K_ratios[(K_ratios >= -1.5) & (K_ratios <= 0.5)]

    counts_zoom, bin_edges_zoom, _ = ax2.hist(K_zoom, bins=300,
                                               color='darkblue', alpha=0.8,
                                               edgecolor='none')

    # Mark key rational values
    rationals = [(-1, '-1'), (-5/6, '-5/6'), (-4/5, '-4/5'), (-3/4, '-3/4'),
                 (-2/3, '-2/3'), (-1/2, '-1/2'), (-1/3, '-1/3'), (0, '0')]

    for val, label in rationals:
        ax2.axvline(x=val, color='orange', linestyle=':', alpha=0.6, linewidth=1)
        ax2.text(val, ax2.get_ylim()[1] * 0.95, label, fontsize=8,
                ha='center', color='orange')

    ax2.axvline(x=-0.5, color='gold', linestyle='-', linewidth=2.5, alpha=0.9)
    ax2.set_xlabel('K(p+1)/(p+1)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Zoomed View: Searching for Spectral Lines', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    import os
    images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    os.makedirs(images_dir, exist_ok=True)

    fig1.savefig(os.path.join(images_dir, 'twin_prime_spectrum.png'),
                 dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSpectral plot saved to: images/twin_prime_spectrum.png")

    # ========================================================================
    # FIGURE 2: The Golden Cloud (2D Density)
    # ========================================================================

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle("The Golden Cloud: Twin Prime K-Structure\n"
                  "Visualizing the Undulating Pattern",
                  fontsize=14, fontweight='bold')

    # Panel 1: 2D Histogram (n vs K/n)
    ax3 = axes2[0, 0]

    # Use log scale for n
    log_n = np.log10(n_values)

    h = ax3.hist2d(log_n, K_ratios, bins=[200, 200],
                   range=[[np.log10(10), np.log10(N)], [-3, 2]],
                   cmap='magma', norm=LogNorm())
    plt.colorbar(h[3], ax=ax3, label='Count (log scale)')

    ax3.axhline(y=-0.5, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('log₁₀(p+1)', fontsize=11)
    ax3.set_ylabel('K(p+1)/(p+1)', fontsize=11)
    ax3.set_title('2D Density: The Structure Emerges', fontsize=12)

    # Panel 2: Golden colormap version
    ax4 = axes2[0, 1]

    # Create custom golden colormap
    from matplotlib.colors import LinearSegmentedColormap
    golden_colors = ['#1a0a00', '#4a2000', '#8b4513', '#cd853f', '#daa520', '#ffd700', '#ffec8b', '#fffacd']
    golden_cmap = LinearSegmentedColormap.from_list('golden', golden_colors)

    h2 = ax4.hist2d(log_n, K_ratios, bins=[200, 200],
                    range=[[np.log10(10), np.log10(N)], [-3, 2]],
                    cmap=golden_cmap, norm=LogNorm())
    plt.colorbar(h2[3], ax=ax4, label='Density')

    ax4.axhline(y=-0.5, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    ax4.set_xlabel('log₁₀(p+1)', fontsize=11)
    ax4.set_ylabel('K(p+1)/(p+1)', fontsize=11)
    ax4.set_title('The Golden Cloud', fontsize=12)

    # Panel 3: Scatter plot (sampled for visibility)
    ax5 = axes2[1, 0]

    # Sample for scatter
    sample_size = min(10000, len(n_values))
    idx = np.random.choice(len(n_values), sample_size, replace=False)

    scatter = ax5.scatter(log_n[idx], K_ratios[idx],
                          c=K_ratios[idx], cmap='coolwarm',
                          s=1, alpha=0.5, vmin=-2, vmax=1)
    plt.colorbar(scatter, ax=ax5, label='K/n value')

    ax5.axhline(y=-0.5, color='black', linestyle='--', linewidth=2)
    ax5.axhline(y=-1, color='gray', linestyle=':', linewidth=1)
    ax5.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax5.set_xlabel('log₁₀(p+1)', fontsize=11)
    ax5.set_ylabel('K(p+1)/(p+1)', fontsize=11)
    ax5.set_title('Scatter View (10K sample)', fontsize=12)
    ax5.grid(True, alpha=0.3)

    # Panel 4: K(n)/n vs n (linear scale, small range)
    ax6 = axes2[1, 1]

    # Take a slice
    mask = (n_values >= 1_000_000) & (n_values <= 2_000_000)
    n_slice = n_values[mask]
    K_slice = K_ratios[mask]

    ax6.scatter(n_slice, K_slice, s=2, alpha=0.6, c='darkgoldenrod')
    ax6.axhline(y=-0.5, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('n = p+1', fontsize=11)
    ax6.set_ylabel('K(n)/n', fontsize=11)
    ax6.set_title('Detail View: n ∈ [1M, 2M]', fontsize=12)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-4, 3)

    plt.tight_layout()

    fig2.savefig(os.path.join(images_dir, 'golden_cloud.png'),
                 dpi=200, bbox_inches='tight', facecolor='black')
    print(f"Golden Cloud saved to: images/golden_cloud.png")

    # ========================================================================
    # FIGURE 3: Ultra-High Resolution Peak Analysis
    # ========================================================================

    fig3, ax7 = plt.subplots(figsize=(16, 6))

    # Super fine bins around -1
    K_fine = K_ratios[(K_ratios >= -1.2) & (K_ratios <= -0.3)]

    counts_fine, edges_fine, _ = ax7.hist(K_fine, bins=400,
                                           color='navy', alpha=0.8,
                                           edgecolor='none')
    centers_fine = (edges_fine[:-1] + edges_fine[1:]) / 2

    # Mark theoretical rational values
    theoretical = [
        (-1, '-1', 'red'),
        (-11/12, '-11/12', 'orange'),
        (-5/6, '-5/6', 'yellow'),
        (-4/5, '-4/5', 'lime'),
        (-3/4, '-3/4', 'cyan'),
        (-7/10, '-7/10', 'blue'),
        (-2/3, '-2/3', 'purple'),
        (-3/5, '-3/5', 'magenta'),
        (-1/2, '-1/2', 'gold'),
        (-2/5, '-2/5', 'pink'),
        (-1/3, '-1/3', 'white'),
    ]

    for val, label, color in theoretical:
        ax7.axvline(x=val, color=color, linestyle='--', alpha=0.7, linewidth=1.5)

    ax7.set_xlabel('K(p+1)/(p+1)', fontsize=12)
    ax7.set_ylabel('Count', fontsize=12)
    ax7.set_title('Ultra-High Resolution: Searching for Quantized Levels\n'
                  'Vertical lines mark simple rational fractions', fontsize=12)
    ax7.set_facecolor('#0a0a1a')
    ax7.tick_params(colors='white')
    ax7.xaxis.label.set_color('white')
    ax7.yaxis.label.set_color('white')
    ax7.title.set_color('white')
    ax7.spines['bottom'].set_color('white')
    ax7.spines['left'].set_color('white')
    fig3.patch.set_facecolor('#0a0a1a')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=f'{label}={val:.4f}')
                       for val, label, color in theoretical]
    ax7.legend(handles=legend_elements, loc='upper right', fontsize=8,
               facecolor='#1a1a2e', labelcolor='white', ncol=2)

    plt.tight_layout()
    fig3.savefig(os.path.join(images_dir, 'spectral_lines.png'),
                 dpi=200, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"Spectral lines saved to: images/spectral_lines.png")

    # ========================================================================
    # Analysis Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal twin prime centers analyzed: {len(K_ratios):,}")
    print(f"Mean K/n: {np.mean(K_ratios):.6f}")
    print(f"Median K/n: {np.median(K_ratios):.6f}")

    # Check clustering around specific rationals
    print("\n" + "-" * 50)
    print("CLUSTERING AROUND RATIONAL VALUES")
    print("-" * 50)

    test_rationals = [(-1, 1), (-5/6, 6), (-4/5, 5), (-3/4, 4),
                      (-2/3, 3), (-1/2, 2), (-1/3, 3), (0, 1)]

    for val, denom in test_rationals:
        # Count within ±0.02 of the rational
        tolerance = 0.02
        count_near = np.sum(np.abs(K_ratios - val) < tolerance)
        pct = count_near / len(K_ratios) * 100
        print(f"  Near {val:>6.3f} (±{tolerance}): {count_near:>6,} ({pct:>5.2f}%)")

    return {
        'n_values': n_values,
        'K_ratios': K_ratios,
        'peaks': peaks
    }


if __name__ == '__main__':
    results = create_twin_prime_spectrum(N=10_000_000)
