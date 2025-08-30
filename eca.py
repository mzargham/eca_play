import numpy as np
import matplotlib.pyplot as plt

Bool = np.bool_  # alias for type hints in docstrings


def wolfram_bits(rule: int) -> np.ndarray:
    """
    Return the 8 rule bits in Wolfram order (111,110,101,100,011,010,001,000).
    bits[0] corresponds to pattern 111 (MSB), bits[7] to 000 (LSB).
    """
    if not (0 <= rule < 256):
        raise ValueError("rule must be in [0, 255]")
    as_str = f"{rule:08b}"  # MSB ... LSB
    return np.fromiter((c == "1" for c in as_str), dtype=bool, count=8)


def selector_tensor() -> np.ndarray:
    """
    Build the fixed selector tensor S ∈ Bool^{8×3×2} once.
    S[k, r, b] = 1  iff  the r-th bit of the k-th Wolfram pattern equals b.
    Wolfram pattern order: (111,110,101,100,011,010,001,000).
    """
    patterns = np.array(
        [[1,1,1],
         [1,1,0],
         [1,0,1],
         [1,0,0],
         [0,1,1],
         [0,1,0],
         [0,0,1],
         [0,0,0]],
        dtype=bool
    )
    S = np.zeros((8, 3, 2), dtype=bool)
    for k in range(8):
        for r in range(3):
            S[k, r, int(patterns[k, r])] = True
    return S


class ECA:
    """
    Elementary Cellular Automaton realized as a composition of Boolean
    linear/tensor maps over the Boolean semiring (∨ as +, ∧ as ×).

        map   : 𝔹^n → 𝔹^{n×3}
        lift  : 𝔹^{n×3} → 𝔹^{n×8}
        reduce: 𝔹^{n×8} → 𝔹^n

    One time-step is therefore

        iterate(x) = reduce(lift(map(x))).

    This file emphasizes *clarity of data flow*: `reduce` operates on the
    lifted one-hot array Z and does NOT call `lift` internally. If you want
    a convenience that accepts neighborhoods directly, use
    `reduce_from_neighborhoods`.
    """

    # --- class-level constants built once ---
    _S = selector_tensor()

    def __init__(self, rule: int):
        self.bits = wolfram_bits(rule)          # shape (8,)
        self.rule = rule
        self.name = f"Rule {rule}"

    # ----- TOPOLOGY -----
    @staticmethod
    def get_neighborhoods(n: int) -> np.ndarray:
        """
        C ∈ Bool^{n×3×n} encodes the ring topology (left, self, right).
        C[i,0,(i-1) mod n] = 1; C[i,1,i] = 1; C[i,2,(i+1) mod n] = 1.
        """
        C = np.zeros((n, 3, n), dtype=bool)
        for i in range(n):
            C[i, 0, (i - 1) % n] = True
            C[i, 1, i] = True
            C[i, 2, (i + 1) % n] = True
        return C

    # ----- RULE-AS-MASK (optional, for pedagogy) -----
    def get_mask(self, n: int) -> np.ndarray:
        """
        B' ∈ Bool^{n×n×8}, diagonal in (i=j), carrying rule bits.
        x^+_i = ⋁_k ( B'[i,i,k] ∧ z[i,k] ).
        Not needed for computation (we can use a direct dot), but included for clarity.
        """
        Bp = np.zeros((n, n, 8), dtype=bool)
        for i in range(n):
            Bp[i, i, :] = self.bits
        return Bp

    # ----- STAGE 1: MAP -----
    @staticmethod
    def map(x: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
        """
        Gather local neighborhoods.
        Parameters
        ----------
        x : Bool[n]
            Current state on the ring.
        C : Bool[n,3,n] (optional)
            Precomputed topology tensor.
        Returns
        -------
        y : Bool[n,3]
            y[i,:] = (x_{i-1}, x_i, x_{i+1}).
        """
        n = x.shape[0]
        if C is None:
            C = ECA.get_neighborhoods(n)
        # Boolean contraction over the last dim:
        # y[i,r] = ⋁_j (C[i,r,j] ∧ x[j])
        y = np.any(C & x[None, None, :], axis=2)
        return y

    # ----- STAGE 2: LIFT -----
    def lift(self, y: np.ndarray) -> np.ndarray:
        """
        One-hot encode which of the 8 Wolfram patterns each row of y equals.
        Returns
        -------
        z : Bool[n,8]
            z[i,k] = 1 iff y[i,:] equals pattern k.
        """
        # Stack ¬y and y → shape (n,3,2)
        y_pm = np.stack([~y, y], axis=2)
        # Broadcast and AND with S: (n,1,3,2) & (8,3,2) → (n,8,3,2)
        H = (y_pm[:, None, :, :] & self._S[None, :, :, :])
        # For each (i,k,r), pick the correct b in {0,1} via OR over last axis
        G = np.any(H, axis=3)         # (n,8,3)
        # Require all 3 positions to match
        z = np.all(G, axis=2)         # (n,8)
        return z

    # ----- STAGE 3: REDUCE -----
    def reduce(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the rule bits to one-hot z via a Boolean dot product.
        x^+_i = ⋁_k ( z[i,k] ∧ bits[k] )
        """
        x_next = np.any(z & self.bits[None, :], axis=1)
        return x_next

    # ----- CONVENIENCE: REDUCE FROM NEIGHBORHOODS -----
    def reduce_from_neighborhoods(self, y: np.ndarray) -> np.ndarray:
        """Convenience wrapper: x^+ = reduce(lift(y))."""
        return self.reduce(self.lift(y))

    # ----- ONE STEP -----
    def iterate(self, x: np.ndarray, C: np.ndarray | None = None) -> np.ndarray:
        """One time-step: return reduce(lift(map(x)))."""
        y = self.map(x, C=C)   # Bool[n,3]
        z = self.lift(y)       # Bool[n,8]
        x_next = self.reduce(z)  # Bool[n]
        return x_next

    # ----- RUN MULTIPLE STEPS -----
    def run(self, x0: np.ndarray, steps: int, C: np.ndarray | None = None) -> np.ndarray:
        """
        Evolve for a fixed number of steps; return the space-time array.
        Returns X with shape (steps, n) where X[t] = state at time t.
        """
        n = x0.shape[0]
        if C is None:
            C = self.get_neighborhoods(n)
        X = np.zeros((steps, n), dtype=bool)
        x = x0.copy()
        for t in range(steps):
            X[t] = x
            x = self.iterate(x, C=C)
        return X

    # ----- UTILITIES -----
    @staticmethod
    def get_random_state(n: int, p: float = 0.5) -> np.ndarray:
        """Random Boolean state with P(True)=p."""
        return (np.random.rand(n) < p)

    def __repr__(self) -> str:
        patterns = ["111","110","101","100","011","010","001","000"]
        mapping = ", ".join(f"{p}→{int(b)}" for p, b in zip(patterns, self.bits))
        return f"<ECA {self.rule}: {mapping}>"

    # ----- PLOTTING -----
    def plot_truth_table(self) -> None:
        """Bar plot of the 8 rule outputs in Wolfram order."""
        fig, ax = plt.subplots()
        ax.bar(range(8), self.bits.astype(int))
        ax.set_xticks(range(8), ["111","110","101","100","011","010","001","000"], rotation=45)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Truth table for {self.name}")
        ax.set_ylabel("Output bit")

    def space_time(self, x0: np.ndarray, steps: int = 64, divider_period: int = 0) -> None:
        """Plot the space-time diagram for the given initial condition."""
        X = self.run(x0, steps)
        n = X.shape[1]
        fig_width = max(6, n / 10)
        fig_height = max(3, steps / 10)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if divider_period is not None and divider_period > 0:
            for r in range(1, steps // divider_period):
                ax.axhline(r * divider_period-.5, color='whitesmoke', linestyle='--', linewidth=2)
            for r in range(1, n // divider_period):
                ax.axvline(r * divider_period-.5, color='whitesmoke', linestyle='--', linewidth=2)
        ax.imshow(X.astype(int), aspect="auto")  # default colormap
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Time step")
        ax.set_xlim(-0.5,n-0.5)
        ax.set_ylim(steps-0.5,-0.5)
        ax.set_xticks(range(0, n, max(1, n // 10)))
        ax.set_yticks(range(steps-1, -1, -max(1, steps // 10)))
        ax.set_title(f"Space-time diagram for {self.name}")
        plt.show()

    def random_demo(self, n: int, steps: int = 60) -> None:
        """Random initial condition demo."""
        x0 = self.get_random_state(n)
        self.space_time(x0, steps=steps)
