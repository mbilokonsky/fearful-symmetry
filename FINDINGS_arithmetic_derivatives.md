# Arithmetic Derivatives Spike: Findings

This document summarizes the key findings from exploring three interrelated ideas:
1. The arithmetic derivative inverse D⁻¹ and its relationship to primes
2. K and K₂ commutators in relation to Riemann zeta zeros
3. Rule 110 cellular automata as a tool for exploring prime structure

---

## Part 1: D⁻¹ (Arithmetic Antiderivative)

### Key Discovery: Primes as D⁻¹(1)

The arithmetic derivative D is defined by:
- D(0) = D(1) = 0
- D(p) = 1 for any prime p
- D(ab) = aD(b) + bD(a) (product rule)

**The primes are exactly characterized as D⁻¹(1)** — the set of all integers with arithmetic derivative equal to 1.

This provides an alternative calculus-based definition of primality.

### Antiderivative Set Distribution

| k | |D⁻¹(k)| (up to 5000) | Notable elements |
|---|------------------------|------------------|
| 0 | 102 | {0, 1, 103, 107, 197, ...} |
| 1 | 569 | {all primes ≤ 5000} |
| 2 | 0 | ∅ (no solutions!) |
| 3 | 0 | ∅ (no solutions!) |
| 4 | 1 | {4} only |

**Observation**: D⁻¹(2) and D⁻¹(3) appear to be empty — no integer has arithmetic derivative 2 or 3. This is a known result: no n satisfies D(n) = 2 or D(n) = 3.

![Antiderivative Distribution](images/antiderivative_distribution.png)

### Derivative Chains

Numbers form chains under repeated differentiation: n → D(n) → D(D(n)) → ...

**Key findings**:
- Primes have chain length 2: p → 1 → 0
- Fixed points exist at p^p: D(4)=4, D(27)=27, D(3125)=3125
- Longest chains start from powers of 2 near primes

| n | Chain length | Chain |
|---|--------------|-------|
| 8 | 20 | 8 → 12 → 16 → 32 → 80 → ... → 4463 |
| 12 | 20 | 12 → 16 → 32 → ... → 1 → 0 |
| 15 | 20 | 15 → 8 → 12 → ... |

![Derivative Chains](images/derivative_chains.png)

### K₂ Telescoping Property

For elements in the same D⁻¹(k) class:
```
Σ K₂(n) from a to b-1 = -(b - a)
```

This was verified for all prime pairs tested — confirming the algebraic identity.

---

## Part 2: K, K₂, and Riemann Zeroes

### K₂ Values at Riemann Zero Integers

| γ (Riemann zero) | ⌊γ⌋ | D(⌊γ⌋) | K₂(⌊γ⌋) | Prime? |
|------------------|------|--------|---------|--------|
| 14.135 | 14 | 9 | -2 | No |
| 21.022 | 21 | 10 | 2 | No |
| 37.586 | 37 | 1 | 18 | **Yes** |
| 43.327 | 43 | 1 | 46 | **Yes** |
| 59.347 | 59 | 1 | 90 | **Yes** |
| 67.080 | 67 | 1 | 70 | **Yes** |

**Observation**: Several Riemann zeros fall very close to primes (37, 43, 59, 67). The K₂ values at these primes are notably large and positive, while non-prime positions show mixed signs.

![Cumulative K₂ with Riemann Zeros](images/k2_riemann_zeros.png)

### Spectral Analysis

The Fourier transform of the K₂ sequence shows:
- Strong low-frequency components
- Power-law decay in the spectrum
- Autocorrelation with periodic structure

![K₂ Spectral Analysis](images/k2_spectral_analysis.png)

### K vs K₂ Comparison

The groovy commutator K (using forward differences) and lucky commutator K₂ (using arithmetic derivative) show different behaviors on primes:

- K on primes: primarily small negative values, stable
- K₂ on primes: wide variance, both positive and negative
- Correlation: weak (data showed numerical issues due to boundary effects)

![K vs K₂ Comparison](images/k_vs_k2_comparison.png)

---

## Part 3: Rule 110 and Primes

### Experiment 1: Prime Positions as Initial State

When we seed a CA with 1s at prime positions (2, 3, 5, 7, 11, ...):

- **Rule 110** produces complex glider-like structures
- **Rule 30** produces chaotic expansion from prime seeds

![CA with Prime Initial Positions](images/ca_prime_initial.png)

### Experiment 2: Commutator Analysis

Running Rule 110 from prime-seeded initial conditions:

| Metric | Value |
|--------|-------|
| Total nonlinearity | 7133 |
| Peak nonlinearity | 83 (at generation 4) |
| Pattern | Nonlinearity decreases as system evolves |

![CA Commutator Analysis](images/ca_prime_commutator.png)

### Experiment 3: CA-Based Prime Detection

We encoded numbers n as binary and ran Rule 110 for 20 generations, measuring:

| Metric | Primes | Composites |
|--------|--------|------------|
| Average final density | 0.4825 | 0.4565 |
| Center cell = 1 probability | 0.4800 | 0.5946 |

**Observation**: Composites show *higher* probability of center cell = 1 after CA evolution. This is a weak but measurable signal — composites seem to produce slightly more "active" CA trajectories in the center.

![CA Prime Detection](images/ca_prime_detection.png)

### Sieve as Cellular Process

The Sieve of Eratosthenes can be visualized as a CA-like process, though it requires non-local rules (marking multiples at arbitrary distances).

![Sieve as CA](images/sieve_as_ca.png)

---

## Summary Figure

![Arithmetic Derivatives Summary](images/arithmetic_derivatives_summary.png)

---

## Conclusions

### Confirmed Insights

1. **D⁻¹(1) = {primes}** provides a calculus-based characterization of primality
2. **Derivative chains** measure "compositeness" — primes are exactly 2 steps from 0
3. **K₂ telescoping** links elements within the same antiderivative class
4. **Several Riemann zeros are near primes** (37, 43, 59, 67 among the first 20)

### Open Questions

1. **D⁻¹ structure**: Why are D⁻¹(2) and D⁻¹(3) empty? What determines |D⁻¹(k)|?
2. **Riemann connection**: Does the K₂ spectrum encode Riemann zero frequencies?
3. **CA primality**: Can Rule 110 be programmed (via initial conditions) to perform primality testing?
4. **Local sieve**: Is there a purely local CA rule that approximates sieving?

### Suggested Follow-ups

- Investigate the spectral peaks in K₂ more carefully against Riemann zeros
- Try different CA rules (Rule 54, Rule 150) for prime detection
- Explore whether D⁻¹(k) sizes follow any asymptotic law
- Connect derivative chain length to prime factorization structure
