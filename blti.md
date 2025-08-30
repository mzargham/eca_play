# Boolean Phase-Space Dynamics

## **System Overview**

We consider a **discrete-time dynamical system** over the Boolean hypercube $\{0,1\}^n$, defined by:

```math
z_{k+1} = f(z_k)
```

where $z_k \in \{0,1\}^n$, and $f: \{0,1\}^n \to \{0,1\}^n$ is a **Boolean linear operator** (e.g., matrix multiplication over GF(2)).

### Elementary Cellular Automata as Boolean Linear Dynamical Systems

Each **Elementary Cellular Automaton (ECA)** can be described as a **Boolean discrete-time dynamical system** over a ring of $n$ binary cells. The state vector $z \in \{0,1\}^n$ evolves by applying a local rule to each cell’s 3-bit neighborhood.

We model this using a **linear operator over the Boolean semiring** $(\{0,1\}, \vee, \wedge)$, decomposed into three stages:

1. **Map**: Gather each site’s neighborhood $(x_{i-1}, x_i, x_{i+1})$ using a fixed tensor $C$, producing a shape $\mathbb{B}^{n \times 3}$.
2. **Lift**: Convert each neighborhood to a one-hot encoding over 8 possible patterns, using a Boolean selector tensor $S$, yielding $\mathbb{B}^{n \times 8}$.
3. **Reduce**: Apply the Wolfram rule bits $b \in \{0,1\}^8$ as a Boolean dot product to compute the next state.

This composition defines a **linear operator** in the lifted basis:

```math
f(z) = \text{reduce}(\text{lift}(\text{map}(z)))
```

and the **Boolean derivative** used in the phase space is:

```math
df(z) = f(z) \oplus z
```

This formulation works uniformly for all 256 ECAs and captures both linear (e.g., Rule 90, 150) and nonlinear cases within a unifying tensor-algebraic framework.

## **Boolean Derivative Operator**

Define a discrete-time Boolean difference (analogous to a derivative):

```
df(z) = z \oplus f(z)
```

* where $f(z)$ is the next state and $\oplus$ is XOR.
* This captures **bit-wise change** in state: `1` where a bit flips between $z_k$ and $z_{k+1}$.
* The operator is recursive: $d^2f(z) = d(df(z))$, $d^3(z) = d(d^2f(z))$, etc.

### **k-Order Phase Space**

We construct an extended state vector:

```math
X_k = [z, df(z), d^2f(z), \ldots, d^kf(z)] \in \{0,1\}^{(k+1)n}
```

This captures the state and its **Boolean derivatives up to order $k$**. It acts like a **Boolean jet space**, analogous to higher-order ODE phase spaces.

### Jet Spaces in Classical Differential Geometry

In classical settings, a **jet space** encodes not just the value of a function (or state), but also its derivatives up to order $k$.

* A $k$-jet of a function at a point encodes:

  ```math
  (x, f(x), f'(x), f''(x), \dots, f^{(k)}(x))
  ```
* Used in control theory, PDEs, and variational calculus to describe the local behavior of trajectories or fields.

### Boolean Jet Space – Discrete Analog

In the **Boolean setting**, the state space is finite: $z \in \{0,1\}^n$ and our Boolean derivitive operator d.

A **Boolean jet of order $k$** is:

```math
J^k(z) = [z, df(z), d^2f(z), \dots, d^kf(z)] \in \{0,1\}^{(k+1)n}
```

This captures:

* **State**: current configuration of bits
* **First-order change**: which bits will flip next
* **Second-order change**: which bits will flip-flop or stabilize
* ...
* **Higher-order evolution patterns**

### Analogy to Classical Jet Spaces

| Classical Jet Space                                 | Boolean Jet Space                                    |
| --------------------------------------------------- | ---------------------------------------------------- |
| Continuous state $x(t)$                             | Discrete state $z_k$                                 |
| Derivatives $\frac{d^k x}{dt^k}$                    | Boolean differences $d^kf(z)$                         |
| Encodes local behavior of smooth trajectories       | Encodes flip patterns over multiple time steps       |
| Used in nonlinear control and differential geometry | Used in symbolic dynamics and discrete logic systems |

### Why It Matters

A **Boolean jet space** allows us to:

* Lift a low-dimensional binary system into a **richer phase space**
* Analyze not just *where* the system is, but *how* it’s evolving at multiple temporal resolutions
* Identify **invariants**, **stable patterns**, **transients**, and **limit cycles**
* Apply techniques like **Boolean Lyapunov analysis**, or symbolic pattern recognition, to understand long-term behavior


## Boolean Lyapunov Analysis

We analyze system evolution using a **Boolean energy function** such as:

```math
V(X_k) = \sum_{i=0}^k \text{wt}(d^if(z))
```

where `wt(·)` is the Hamming weight (number of 1s).

* $V$ measures the "activity" across derivative layers.
* If $V$ is **non-increasing**, the system is evolving toward a **stable invariant set**.
* Given the finite state space ($2^{(k+1)n}$), the system is **guaranteed to settle into a limit cycle or fixed point**.

### Invariance and Stability Interpretation

To perform Boolean Lyapunov analysis in this context, we treat $V(X_k)$ as a **discrete Lyapunov function** and analyze its behavior over time:

* If $V(X_{k+1}) < V(X_k)$ for all $X_k \notin M$, where $M \subseteq \{0,1\}^{(k+1)n}$ is the set of minimal-energy configurations, then the system **asymptotically evolves toward $M$**.
* If $V(X_{k+1}) \leq V(X_k)$, we apply a **LaSalle-style argument**: all trajectories enter and remain in the **largest invariant set contained in**

  ```math
  E = \{ X \in \{0,1\}^{(k+1)n} \mid V(X_{k+1}) = V(X_k) \}
  ```

  and the system's long-term behavior is constrained to $E$.

This provides a symbolic method for reasoning about **stability and convergence** even when traditional calculus-based tools are unavailable. It also enables the identification of:

* **Fixed points**: configurations where all Boolean derivatives vanish.
* **Limit cycles**: repeating or oscillating sequences of derivative vectors.
* **Basins of attraction**: sets of states that flow into a shared invariant subset.

In this way, Boolean Lyapunov analysis extends classical techniques to **fully discrete systems**, enabling rigorous reasoning about dynamics in automata, logic circuits, and digital models.

### Computing $k$-Limit Cycles in Boolean Jet Space

In the Boolean jet space framework, a **$k$-limit cycle** is a sequence of $k$ distinct extended states $\{X_0, X_1, \dots, X_{k-1}\}$ such that:

```math
X_{j+1} = \Phi(X_j) \quad \text{for } j = 0, \dots, k-2, \quad \text{and} \quad X_0 = \Phi(X_{k-1})
```

where $\Phi$ is the system's full phase-space update map (i.e., the lifted dynamics of $z \to f(z)$, including the derivatives).


### Fixed Points as 1-Cycles

A **fixed point** is simply a $k = 1$ cycle:

```math
X = \Phi(X)
```

This corresponds to a state where the system (and all Boolean derivatives) have stabilized.

### Cycle Equivalence by Rotation

Because the system is deterministic and synchronous, two cycles are considered **equivalent** if they are **cyclic rotations** of each other:

```math
\{X_0, X_1, \dots, X_{k-1}\} \sim \{X_i, X_{i+1}, \dots, X_{i-1} \mod k\}
```

This avoids double-counting cycles that differ only in the choice of starting index.

### Algorithm for Detecting $k$-Limit Cycles

1. **Initialize**: Start from every possible initial extended state $X_0 \in \{0,1\}^{(k+1)n}$ or a sampled subset.
2. **Simulate**: Iterate the update map $\Phi$ up to a maximum number of steps $T \leq 2^{(k+1)n}$.
3. **Detect loops**: Use a hash table or set to track visited states. A repeat implies a cycle.
4. **Normalize**: Canonically rotate the detected cycle to a fixed starting point for uniqueness.
5. **Record**: Store the unique cycles up to rotation equivalence.

This method enables the classification of the system's long-term dynamics, including:

* **Number and structure of attractors**
* **Period and size of cycles**
* **Transitions between basins of attraction**

This cycle analysis is essential for understanding the **symbolic behavior** of Boolean systems such as ECAs, digital automata, and logic-based network models.

## **Conclusion**

This approach fuses ideas from:

* **Boolean algebra**
* **Discrete dynamical systems**
* **Phase-space analysis**
* **Symbolic computation**

to enable rigorous study of systems where classical calculus-based Lyapunov methods do not apply. It’s especially suited for modeling, simulating, and analyzing **binary-valued logic systems, digital circuits, automata, or symbolic biological models**.
