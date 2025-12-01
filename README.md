# fearful-symmetry

A discrete calculus for cellular automata, a commutator for identifying local pockets of nonlinearity, and some fascinating implications for number theory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbilokonsky/fearful-symmetry/blob/main/wolfram_ca.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbilokonsky/fearful-symmetry/main?labpath=wolfram_ca.ipynb)

## Why Discrete Calculus for Cellular Automata?

Cellular automata (CAs) are typically understood as monolithic step functions: apply a rule, get the next state. But this view obscures the internal structure of how change propagates through the system.

By decomposing CA evolution into **derivative** and **integral** operations, we gain:

1. **Separation of concerns**: The derivative tells us *what will change*; the integral *applies* those changes
2. **A language for analyzing dynamics**: We can now ask questions like "how does the pattern of change itself change?"
3. **Detection of nonlinearity**: The commutator `C(S) = E(D(S)) XOR D(E(S))` reveals exactly where and when the CA exhibits nonlinear behavior

This framework transforms cellular automata from opaque computational objects into systems we can reason about using calculus-like operations.

## The Discrete Calculus Framework

### Definitions

| Symbol | Name | Description |
|--------|------|-------------|
| **G** | Configuration space | The space of all possible CA states |
| **R** | Ruleset | Maps neighborhoods to outputs (Wolfram rules 0-255) |
| **N** | Neighborhood | The cells that influence each cell (left, center, right) |
| **S** | State | A vector of cell values at a given time |

### Core Operations

#### Derivative: D(S)
```
D(S)[i] = R(N[i]) XOR S[i]
```

The derivative is a **bitmask indicating which cells will change** in the next generation. A cell is marked `1` if applying the rule produces a different value than its current state.

Key insight: We compute what will change *without* computing the full next state.

#### Integral: I(S, D)
```
I(S, D) = S XOR D
```

The integral **applies changes** by flipping cells wherever the derivative is `1`. This is the XOR operation—reversible and its own inverse.

#### Evolve: E(S)
```
E(S) = I(S, D(S)) = S XOR D(S)
```

Evolution is now expressed as a **composition** of derivative and integral. This is mathematically equivalent to applying the rule directly, but decomposed into meaningful parts.

#### Commutator: C(S)
```
C(S) = E(D(S)) XOR D(E(S))
```

The commutator asks: **does the order of operations matter?**

- Compute `E(D(S))`: first differentiate, then evolve
- Compute `D(E(S))`: first evolve, then differentiate
- XOR them together

Where these differ (commutator = 1), the system exhibits **nonlinearity**. Linear rules like Rule 90 have zero commutator everywhere. Chaotic rules like Rule 30 show complex commutator patterns.

## What This Reveals

### Linear vs Nonlinear Rules

The commutator provides a precise, local measure of nonlinearity:

- **Rule 90** (Sierpiński triangle): Purely linear (XOR-based). Commutator is zero everywhere—differentiation and evolution commute.
- **Rule 30** (chaotic): High nonlinearity. Complex commutator patterns reveal where the rule's behavior cannot be decomposed linearly.

### Patterns of Change

Visualizing the derivative alongside state evolution shows how "activity" propagates differently than the state itself. The derivative often reveals structure that isn't obvious from the state alone.

## Findings

The following visualizations show state evolution (S) alongside the commutator (C) for representative rules. Red regions in the commutator indicate nonlinearity—where the order of differentiation and evolution matters.

### Rule 30 (Chaotic) — Total nonlinearity: 2405

![Rule 30](images/rule_30.png)

Rule 30 produces pseudo-random, chaotic patterns from a single seed cell. The commutator shows intense nonlinearity throughout the active region, with complex patterns that mirror—but don't duplicate—the chaotic structure. This high nonlinearity is characteristic of Class III (chaotic) rules.

### Rule 90 (Linear) — Total nonlinearity: 0

![Rule 90](images/rule_90.png)

Rule 90 generates the Sierpiński triangle through pure XOR operations. The commutator is **completely zero**—differentiation and evolution commute perfectly. This confirms Rule 90 is purely linear: it can be fully decomposed without any order-dependent interactions.

### Rule 110 (Turing Complete) — Total nonlinearity: 1057

![Rule 110](images/rule_110.png)

Rule 110 is proven Turing complete—capable of universal computation. Its commutator shows moderate, structured nonlinearity. The nonlinear regions appear at the boundaries of localized structures, suggesting these are precisely the sites where computational interactions occur.

### Rule 184 (Traffic Flow) — Total nonlinearity: 150

![Rule 184](images/rule_184.png)

Rule 184 models traffic flow (particles moving right unless blocked). It shows minimal nonlinearity, concentrated at particle collision sites. This makes physical sense: the rule is mostly linear (particles flow freely) with nonlinearity only where particles interact.

### Summary

| Rule | Class | Total Nonlinearity | Behavior |
|------|-------|-------------------|----------|
| 30 | III (Chaotic) | 2405 | High, distributed nonlinearity |
| 90 | II (Periodic) | 0 | Perfectly linear (XOR-based) |
| 110 | IV (Complex) | 1057 | Moderate, localized at structures |
| 184 | II (Periodic) | 150 | Minimal, only at interactions |

The commutator successfully distinguishes rule classes: chaotic rules show high nonlinearity, linear rules show zero, and complex/computational rules show intermediate, structured patterns.

## Using the Notebook

Click either badge above to run the notebook interactively:

- **Google Colab**: Fast startup, requires Google account
- **Binder**: No account needed, may take longer to start

### Notebook Contents

1. **Core Implementation**: The `WolframCA` class with all discrete calculus operations
2. **Visualization Tools**: Functions to plot CA evolution, rule tables, and calculus analysis
3. **Worked Examples**:
   - Rule 30 (chaotic, high nonlinearity)
   - Rule 90 (linear, zero commutator)
   - Rule 110 (Turing complete)
4. **Interactive Exploration**: Customize rules, width, and initial conditions
5. **Rule Classification**: Compare Wolfram's four classes of CA behavior

### Quick Start

```python
from wolfram_ca import WolframCA

# Create a CA with Rule 30
ca = WolframCA(rule=30, width=101)
ca.initialize()

# Run for 50 generations
ca.run(50)

# Examine the calculus
state = ca.history[-1]
print("State:", state)
print("Derivative:", ca.differentiate(state))
print("Commutator:", ca.commutator(state))
```

### Using the Library

The `src/ca_calculus.py` module provides a standalone `CACalculus` class:

```python
from src.ca_calculus import CACalculus

calc = CACalculus(rule=30)

# Run complete analysis
results = calc.run_analysis(initial_state, generations=100)

# Access results
states = results['states']         # State evolution
derivatives = results['derivatives']  # D(S) at each step
commutators = results['commutators']  # C(S) at each step
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install with:
```bash
pip install -r requirements.txt
```

## Further Exploration

Questions this framework opens up:

- Can we classify rules by their commutator statistics?
- What is the relationship between commutator density and computational universality?
- How do these operations behave in higher-dimensional CAs?
- Are there rules where the commutator exhibits its own emergent patterns?
