# LaSalle-style Detection of Invariant Subspaces & Limit Cycles in ECAs

This note explains how we combine:

1. a **Boolean linear/tensor algebra** factorization of ECA updates, and
2. a **phase-space augmentation** built from higher-order XOR differences,

to construct LaSalle-type potentials, carve out **invariant subspaces**, and detect **limit cycles** (fixed points, 2-cycles, and beyond).

The approach is fully implemented by the code in `eca.py` (map–lift–reduce) and `lasalle.py` (order-$k$ phase space).

---

## 0) Notation (Boolean semiring and XOR differences)

* We work over the Boolean semiring $(\mathbb B,\vee,\wedge,\neg)$.
* The ECA global state at time $t$ is $x(t)\in\mathbb B^n$.
* Define the forward **XOR difference** operator $\Delta$ by

  $$
  \Delta s(t) = s(t{+}1)\oplus s(t),
  $$

  and recursively $\Delta^{m+1}s(t)=\Delta(\Delta^m s)(t)$.
  Our **order-$k$ phase state** is the layered tuple

  $$
  X^{(k)}(t)=\big(\Delta^0 x(t),\Delta^1 x(t),\ldots,\Delta^k x(t)\big)\in(\mathbb B^n)^{k+1}.
  $$

---

## 1) ECA as Boolean tensor algebra (map → lift → reduce)

For radius-1 ECAs, one update is the composition

$$
x(t{+}1)=\underbrace{\text{reduce}}_{\mathbb{B}^{n\times 8}\to\mathbb{B}^n}\big(\underbrace{\text{lift}}_{\mathbb{B}^{n\times3}\to\mathbb{B}^{n\times 8}}(\underbrace{\text{map}}_{\mathbb{B}^n\to\mathbb{B}^{n\times3}}(x(t)))\big),
$$

and the code wires it as `iterate(x) = reduce(lift(map(x)))` in `eca.py`. &#x20;

* **map** uses a fixed neighborhood tensor $C\in\mathbb B^{n\times3\times n}$ to gather $(x_{i-1},x_i,x_{i+1})$.&#x20;
* **lift** converts each triple to an 8-way one-hot vector using a fixed selector tensor $S$.&#x20;
* **reduce** is a Boolean dot product with the rule’s 8 Wolfram-ordered bits.&#x20;

These same objects and type signatures are summarized in the README.&#x20;

---

## 2) Order-$k$ phase space (layers $[x,\delta,\Delta^2,\ldots,\Delta^k]$)

The class `PhaseSpace` in `lasalle.py` stores the order-$k$ layers as a Boolean array of shape $(k{+}1,n)$ with row $j$ equal to $\Delta^j x(t)$.&#x20;

It provides a **one-step phase update** $\Phi_k$ with the exact layerwise rules:

* For $j=0,\dots,k-1$: $\Delta^j x(t{+}1)=\Delta^j x(t)\oplus\Delta^{j+1}x(t)$.
* For the top layer: $\Delta^k x(t{+}1)$ is formed from $x(t{+}1),\dots,x(t{+}k{+}1)$ via binomial parity (see §3).

These rules and the implementation strategy are documented in the class docstring.&#x20;

---

## 3) Efficient stepping: binomial parity + one ECA call

We use binomial coefficients **mod 2** (Lucas’ theorem) to reconstruct short trajectories and update the top layer:

* `_binom_parity(n,k)` returns $\binom{n}{k}\bmod2$.&#x20;
* `_build_parity_table(kmax)` builds the parity triangle up to order $k$.&#x20;

`PhaseSpace.step` works as follows (see docstring bullets):

1. Reconstruct $x(t),x(t{+}1),\dots,x(t{+}k)$ from the current layers using the parity table.
2. Compute $x(t{+}k{+}1)$ with **one** call to `eca.iterate(x(t{+}k))`.
3. Update lower layers by XOR-chaining; update the top layer from the window $x(t{+}1..t{+}k{+}1)$.&#x20;

The base updater used in step (2) is the Boolean tensor composition `ECA.iterate`.&#x20;

---

## 4) LaSalle potentials from lifted pattern counts

Given a rule’s 8 output bits and the per-site one-hot $z[i,:]$ (from **lift**), define counts $c_k(x)=\sum_i z[i,k]$ and an additive potential

$$
V_w(x)=\sum_{k=0}^{7} w_k\,c_k(x).
$$

`PhaseSpace` exposes helpers to evaluate such potentials:

* `V_triples(x, w8)`: counts triple occurrences via `eca.map` + `eca.lift`.&#x20;
  (Recall `ECA.lift` returns a strict one-hot per site. )
* `V_layers(layers, w8_per_layer)`: sums layerwise potentials to score $[x,\delta,\Delta^2,\dots]$.&#x20;

Design $w$ so that **every** local 5-block (radius-1 context) satisfies $V_w$ non-increase; the set where equality holds (no “energy-dropping” contexts) is a **LaSalle equality set** $E$. The **largest invariant subset** of $E$ is then a dynamically stable subspace under the ECA dynamics.

---

## 5) Invariant subspaces & cycle detection in phase space

Working directly in the order-$k$ phase space gives algebraic predicates that are easy to test:

* **Fixed points:** $\delta=\Delta^1 x=\mathbf 0$.
* **2-cycles:** $\delta\neq \mathbf 0$ and $\delta$ is **fixed** by one phase step: $\Delta^1(t{+}1)=\Delta^1(t)$.
* **Periods dividing $2^r$:** if $\Delta^{2^r}x\equiv 0$ then $x(t{+}2^r)=x(t)$. Raising $k$ increases the range of detectable cycle lengths (e.g., $k{=}3$ covers $\{1,2,4,8\}$).

`PhaseSpace.run(layers0, steps)` evolves the augmented state for analysis or visualization.&#x20;

---

## 6) Minimal workflow

1. **Choose rule/size.**

   ```python
   from eca import ECA
   from lasalle import PhaseSpace
   eca = ECA(rule)   # map→lift→reduce wired in iterate
   ps  = PhaseSpace(eca, n, k)
   ```

   (See the iterate composition and type signatures in `eca.py` & README.) &#x20;

2. **Initialize layers from a base state** $x$:
   `layers0 = ps.init_from_x(x)` (builds $[x,\delta,\ldots,\Delta^k]$ from a short trajectory using binomial parity).

3. **Step the phase map**:
   `layers1 = ps.step(layers0)` (reconstructs $x(t..t{+}k)$, calls `eca.iterate` once, updates layers).&#x20;

4. **Lyapunov scoring & invariance tests**:
   Evaluate `ps.V_triples` on selected layers or `ps.V_layers` across layers, filter states that keep the score constant (LaSalle equality set), and shrink to the **largest invariant subset** by closure under `ps.step`.&#x20;

5. **Cycle classification**:

   * project to base space and check minimal period (e.g., $1,2,4,8$ for $k{=}3$),
   * or use phase predicates (e.g., “$\Delta^j$ fixed” with constraints on lower layers) to target specific periods.

---

## 7) Relation to linear rules and extensions

* For GF(2)-**linear** rules (e.g., 90, 150), $\delta$ dynamics is linear and can be analyzed by standard algebra on subspaces; the same phase framework applies but with extra structure.
* **Larger neighborhoods:** the same map–lift–reduce factorization extends to radius $>1$ by enlarging the tensors $C$ and $S$; see the README’s “Extending and experimenting.”&#x20;

---

## 8) Files & where things live

* **`eca.py`** — Boolean tensor factorization of the update:

  * Type signatures, and `iterate(x)=reduce(lift(map(x)))`.&#x20;
  * Neighborhood tensor $C$ (**map**), selector tensor $S$ (**lift**), rule reduction (**reduce**).  &#x20;
* **`lasalle.py`** — Order-$k$ phase space and utilities:

  * Layer convention, phase step equations, parity table construction. &#x20;
  * Exact, efficient step using a single call to `eca.iterate` per phase step.&#x20;
  * Potentials on $x$ and on layers (`V_triples`, `V_layers`).&#x20;
* **`readme.md`** — High-level math summary and API signatures (helpful for orientation). &#x20;

---

## 9) Takeaways

* The **lifted one-hot basis** makes per-site rule application linear (in the Boolean semiring), enabling simple, local **LaSalle potentials** built from pattern counts.&#x20;
* The **order-$k$ phase space** turns cycle detection into algebra on layers $[x,\delta,\Delta^2,\ldots]$ and provides clean **invariance predicates** to prune the state space efficiently.&#x20;

---

*For examples and experiments, see the companion notebook you provided (`lasalle_play.ipynb`) alongside this repository. The code paths cited above are sufficient to reproduce the analyses within that notebook.*

---

*For larger scale experiments, we can migrate to a CuPy based implementation and run experiments on GPUs.*