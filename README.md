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
