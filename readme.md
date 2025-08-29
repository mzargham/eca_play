# Elementary Cellular Automata as Boolean Linear/Tensor Algebra

This repo implements **Elementary Cellular Automata (ECA)** as a **discrete-time network dynamical system** using only Boolean linear/tensor algebra—every operation is an OR/AND/NOT contraction over fixed tensors.

Mathematically, for a ring of $n$ cells with state $x(t)\in\{0,1\}^n$,

```math
x(t+1)=\underbrace{\text{reduce}}_{\; \{0,1\}^{n\times 8}\to\{0,1\}^n}
\big(
\underbrace{\text{lift}}_{\; \{0,1\}^{n\times 3}\to\{0,1\}^{n\times 8}}
(
\underbrace{\text{map}}_{\; \{0,1\}^n\to\{0,1\}^{n\times 3}}
(x(t))
\big)\big).
```

> **Code correspondence:**
> `ECA.map → ECA.lift → ECA.reduce` and the composition is wired in `ECA.iterate` (see `eca.py`).

---

## Boolean semiring and contraction

We work over the Boolean semiring $(\mathbb{B},\vee,\wedge,\neg)$ where

* elements: $\mathbb{B}=\{0,1\}$,
* “addition”: $a\oplus b:=a\vee b$ (OR),
* “multiplication”: $a\otimes b:=a\wedge b$ (AND).

Whenever we write a matrix/tensor “product,” we mean a standard index contraction with sums replaced by ORs and products by ANDs.

---

## Stage A — Neighborhood **map** $\mathbb{B}^n\to\mathbb{B}^{n\times 3}$

We encode the radius-1 ring topology with a fixed 3-mode tensor

$$
C\in\mathbb{B}^{n\times 3\times n},\quad
C[i,0,(i-1)\!\!\!\!\pmod n]=1,\;C[i,1,i]=1,\;C[i,2,(i+1)\!\!\!\!\pmod n]=1.
$$

The neighborhood gather is the Boolean contraction

$$
y[i,r] \;=\; \bigvee_{j=0}^{n-1}\big(C[i,r,j]\wedge x[j]\big),
\quad\Rightarrow\quad y[i,:]=(x_{i-1},x_i,x_{i+1}).
$$

**Implements:**

* `ECA.get_neighborhoods(n)` builds $C$.
* `ECA.map(x, C)` computes $y$ via `np.any(C & x[None,None,:], axis=2)`.

---

## Stage B — Pattern **lift** $\mathbb{B}^{n\times 3}\to\mathbb{B}^{n\times 8}$

Let the 8 Wolfram-ordered patterns be
$(111,110,101,100,011,010,001,000)$.
Build a fixed selector tensor $S\in\mathbb{B}^{8\times 3\times 2}$ with

$$
S[k,r,b]=1 \iff \text{the } r\text{-th bit of pattern }k \text{ equals } b\in\{0,1\}.
$$

For $y$, stack negatives and positives $Y_{\pm}=[\neg y,\;y]\in\mathbb{B}^{n\times 3\times 2}$ and define

$$
z[i,k]
= \bigwedge_{r=0}^{2}\Big( (\neg y[i,r]\wedge S[k,r,0]) \;\vee\; (y[i,r]\wedge S[k,r,1]) \Big),
$$

so each row $z[i,:]$ is one-hot for which of the 8 patterns occurs at site $i$.

**Implements:**

* `selector_tensor()` constructs $S$ once.
* `ECA.lift(y)` performs the OR/AND broadcast‐contractions to produce $z$.

---

## Stage C — Rule **reduce** $\mathbb{B}^{n\times 8}\to\mathbb{B}^n$

For rule $R\in\{0,\dots,255\}$, its Wolfram-ordered bits
$\mathbf{b}=(b_0,\dots,b_7)^\top\in\mathbb{B}^8$ are

```math
\mathbf b=\texttt{wolfram\_bits(rule)}.
```

The local update is a Boolean dot product of the one-hot $z[i,:]$ with $\mathbf b$:

$$
x_i^+ \;=\; \bigvee_{k=0}^{7}\big(z[i,k]\wedge b_k\big).
$$

**Implements:**

* `wolfram_bits(rule)` returns $\mathbf b$.
* `ECA.reduce(z)` computes $x^+$ by `np.any(z & bits[None,:], axis=1)`.

> For pedagogy, `ECA.get_mask(n)` also builds the diagonal mask $B'\in\mathbb{B}^{n\times n\times 8}$ with $B'[i,i,k]=b_k$, making the same reduction explicit as a three-index contraction.

---

## Putting it together

Type signatures (enforced by shapes in the code):

```
map   : 𝔹^n      → 𝔹^{n×3}       (ECA.map)
lift  : 𝔹^{n×3}  → 𝔹^{n×8}       (ECA.lift)
reduce: 𝔹^{n×8}  → 𝔹^n           (ECA.reduce)

iterate(x) = reduce(lift(map(x)))  (ECA.iterate)
```

**Implements:**

* `ECA.iterate(x, C)` wires the composition exactly as above (no hidden lifting).
* `ECA.reduce_from_neighborhoods(y)` is a convenience wrapper: `reduce(lift(y))`.

---

## Minimal example (Rule 30)

```python
import numpy as np
from eca import ECA

eca = ECA(rule=30)
n = 64
x0 = np.zeros(n, dtype=bool); x0[n//2] = True

# One explicit step, showing the math flow:
C = eca.get_neighborhoods(n)     # Stage A tensor
y = eca.map(x0, C)               # 𝔹^{n×3}
z = eca.lift(y)                  # 𝔹^{n×8}
x1 = eca.reduce(z)               # 𝔹^n

# Or simply:
x1_alt = eca.iterate(x0, C)
assert np.array_equal(x1, x1_alt)

# Visualize a space-time diagram
eca.space_time(x0, steps=64)
```

---

## Correctness properties (all hold in `eca.py`)

1. **One-hot lift:** `z = eca.lift(y)` satisfies `z.sum(axis=1) == 1` (Boolean sum) for all rows.
2. **Equivalence:** `eca.iterate(x) == eca.reduce(eca.lift(eca.map(x)))`.
3. **Locality:** `ECA.map` depends only on $x_{i-1},x_i,x_{i+1}$ via the fixed $C$.
4. **Rule linearity in the lifted basis:** `ECA.reduce` is a linear map over the Boolean semiring with respect to `z`.

You can quickly sanity-check (2) with randomized tests:

```python
rng = np.random.default_rng(0)
eca = ECA(110)
for n in (5, 17, 64):
    C = eca.get_neighborhoods(n)
    for _ in range(50):
        x = rng.integers(0, 2, size=n, dtype=bool)
        assert np.array_equal(eca.iterate(x, C),
                              eca.reduce(eca.lift(eca.map(x, C))))
```

---

## API (with shapes)

* `ECA(rule: int)` — construct from Wolfram code. *(Uses `wolfram_bits`.)*
* `map(x: Bool[n], C=None) -> Bool[n,3]` *(uses `get_neighborhoods` if `C` missing)*
* `lift(y: Bool[n,3]) -> Bool[n,8]` *(uses class tensor `S` from `selector_tensor`)*
* `reduce(z: Bool[n,8]) -> Bool[n]`
* `reduce_from_neighborhoods(y: Bool[n,3]) -> Bool[n]`
* `iterate(x: Bool[n], C=None) -> Bool[n]` *(per the composition above)*
* `run(x0: Bool[n], steps: int, C=None) -> Bool[steps,n]`
* Viz/aux: `space_time`, `plot_truth_table`, `get_random_state`, `get_mask`, `get_neighborhoods`.

---

## How this matches classic ECA

Classic ECA: “Look up the output bit for each 3-bit neighborhood.”
Here that is factorized into fixed Boolean contractions:

1. **Gather** $(x_{i-1},x_i,x_{i+1})$ — `map` with $C$.
2. **One-hot match** against 8 patterns — `lift` with $S$.
3. **Pick the rule bit** — `reduce` with $\mathbf b$.

This works uniformly for **all 256 rules**. For the linear subset (e.g., 90, 150), you can additionally write $x(t+1)=A\,x(t)$ over $\mathbb{F}_2$; the code keeps the more general lifted-basis formulation.

---

## Extending and experimenting

* **Radius $k>1$:** generalize $C$ to $n\times(2k+1)\times n$ and $S$ to $2^{2k+1}\times(2k+1)\times 2$; `lift` and `reduce` generalize verbatim.
* **Multiple ICs:** batch along a new axis; all contractions vectorize naturally.
* **Mask view:** inspect `get_mask(n)` to see the diagonal $B'$ implementing the rule as a three-index Boolean linear map.

---

## License

MIT
