# lasalle.py
# Phase-space machinery for Boolean ECA with higher-order XOR-differences.
# Relies on eca.py (ECA class with iterate/map/lift).

from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple, List, Optional
from eca import ECA

Bool = np.bool_


def _binom_parity(n: int, k: int) -> bool:
    """
    Parity of C(n,k) mod 2 (Lucas theorem):
    C(n, k) is odd  <=>  (k & (n - k)) == 0  (for nonnegative ints).
    """
    if k < 0 or k > n:
        return False
    return (k & (n - k)) == 0


def _build_parity_table(kmax: int) -> np.ndarray:
    """
    Upper-left (kmax+1)×(kmax+1) Boolean triangle P with P[n,k] = C(n,k) mod 2.
    Only entries with k<=n are meaningful; others are False.
    """
    P = np.zeros((kmax + 1, kmax + 1), dtype=bool)
    for n in range(kmax + 1):
        for k in range(n + 1):
            P[n, k] = _binom_parity(n, k)
    return P


class PhaseSpace:
    """
    Phase-space for order-k XOR-differences of an ECA on a ring of size n.

    Layer convention:
        layers[j, :] == Δ^j x(t)   for j = 0..k,  with Δ^0 x ≡ x.

    Shapes:
        layers: (k+1, n)  Boolean
        flattened: ( (k+1)*n, ) Boolean

    One-step phase update Φ_k:
        For j=0..k-1:   Δ^j x(t+1) = Δ^j x(t) XOR Δ^{j+1} x(t)
        For j= k:       Δ^k x(t+1) = Δ^k x evaluated at time (t+1)
                        = XOR_{m=0..k} [ C(k,m) mod 2 ] * x(t+1+m)

    Implementation strategy (exact, efficient):
        - Reconstruct x(t), x(t+1), ..., x(t+k) from current layers via binomial parity.
        - Compute x(t+k+1) with ONE call to eca.iterate( x(t+k) ).
        - Update lower layers with the XOR chain; update the top layer using
          the binomial combo of x(t+1..t+k+1).
    """

    def __init__(self, eca: ECA, n: int, k: int, precompute_C: bool = True):
        if k < 0:
            raise ValueError("k must be ≥ 0")
        self.eca = eca
        self.n = int(n)
        self.k = int(k)
        self.P = _build_parity_table(k + 1)  # binomial parity table
        self.C = eca.get_neighborhoods(n) if precompute_C else None

    # ---------- representation helpers ----------

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (k+1, n)."""
        return (self.k + 1, self.n)

    def flatten(self, layers: np.ndarray) -> np.ndarray:
        """(k+1, n) -> ((k+1)*n,)"""
        return layers.reshape(-1)

    def unflatten(self, vec: np.ndarray) -> np.ndarray:
        """((k+1)*n,) -> (k+1, n)"""
        return vec.reshape(self.k + 1, self.n)

    # ---------- initialization ----------

    def init_from_x(self, x: np.ndarray) -> np.ndarray:
        """
        Build the order-k phase vector from a base state x at time t.

        Does k calls to eca.iterate to get x(t..t+k), then forms Δ^m via
        XOR-binomial combinations.

        Returns
        -------
        layers : Bool[k+1, n]
            layers[m] = Δ^m x(t)
        """
        x = np.asarray(x, dtype=bool)
        if x.shape != (self.n,):
            raise ValueError(f"x must have shape ({self.n},)")
        # short trajectory x0..xk
        xs: List[np.ndarray] = [x.copy()]
        cur = x
        for _ in range(self.k):
            cur = self.eca.iterate(cur, C=self.C)
            xs.append(cur)
        # build Δ^m from xs using C(m,j) mod 2
        layers = np.zeros((self.k + 1, self.n), dtype=bool)
        for m in range(self.k + 1):
            acc = np.zeros(self.n, dtype=bool)
            for j in range(m + 1):
                if self.P[m, j]:
                    acc ^= xs[j]
            layers[m] = acc
        return layers

    # ---------- core: reconstruct x(t+j) from layers at time t ----------

    def reconstruct_xs(self, layers: np.ndarray, j_max: int) -> List[np.ndarray]:
        """
        Given layers= [Δ^0..Δ^k] at time t, reconstruct x(t+j) for j=0..j_max
        (requires j_max ≤ k). Uses x_j = XOR_{m=0..j} C(j,m) Δ^m.
        """
        if layers.shape != (self.k + 1, self.n):
            raise ValueError(f"layers must have shape {(self.k + 1, self.n)}")
        if j_max > self.k:
            raise ValueError("j_max cannot exceed k when reconstructing without F")
        xs: List[np.ndarray] = []
        for j in range(j_max + 1):
            acc = np.zeros(self.n, dtype=bool)
            for m in range(j + 1):
                if self.P[j, m]:
                    acc ^= layers[m]
            xs.append(acc)
        return xs

    # ---------- one phase step ----------

    def step(self, layers: np.ndarray) -> np.ndarray:
        """
        One Φ_k step on the whole phase vector.

        Parameters
        ----------
        layers : Bool[k+1, n]  at time t

        Returns
        -------
        layers_plus : Bool[k+1, n]  at time t+1
        """
        k, n = self.k, self.n
        if layers.shape != (k + 1, n):
            raise ValueError(f"layers must have shape {(k + 1, n)}")

        # 1) Reconstruct x(t..t+k)
        xs = self.reconstruct_xs(layers, j_max=k)  # x0..xk

        # 2) Compute x(t+k+1) with ONE call to F
        x_next_last = self.eca.iterate(xs[-1], C=self.C)  # x_{k+1}

        # 3) Lower layers update: Δ^j(t+1) = Δ^j(t) XOR Δ^{j+1}(t), j=0..k-1
        layers_plus = np.empty_like(layers)
        for j in range(k):
            layers_plus[j] = layers[j] ^ layers[j + 1]

        # 4) Top layer update: Δ^k(t+1) = XOR_{m=0..k} C(k,m) * x(t+1+m)
        #    Build window x1..x_{k+1} (length k+1)
        x_window = xs[1:] + [x_next_last]  # [x1, x2, ..., x_{k+1}]
        acc = np.zeros(n, dtype=bool)
        for m in range(k + 1):
            if self.P[k, m]:
                acc ^= x_window[m]
        layers_plus[k] = acc

        return layers_plus

    # ---------- run trajectories ----------

    def run(self, layers0: np.ndarray, steps: int) -> np.ndarray:
        """
        Evolve the phase state for a number of steps.

        Returns
        -------
        traj : Bool[steps, k+1, n], where traj[t] = layers at time t.
        """
        layers = np.asarray(layers0, dtype=bool)
        if layers.shape != (self.k + 1, self.n):
            raise ValueError(f"layers0 must have shape {(self.k + 1, self.n)}")
        traj = np.zeros((steps, self.k + 1, self.n), dtype=bool)
        for t in range(steps):
            traj[t] = layers
            layers = self.step(layers)
        return traj

    # ---------- optional: Lyapunov potentials on layers ----------

    def V_triples(self, x: np.ndarray, w8: np.ndarray) -> int:
        """
        V_w(x) = sum_k w_k * count_k(x), using the ECA's lift (triples).
        w8 must be shape (8,), Wolfram order.
        """
        w8 = np.asarray(w8, dtype=int)
        if w8.shape != (8,):
            raise ValueError("w8 must have shape (8,)")
        y = self.eca.map(x, C=self.C)     # (n,3)
        z = self.eca.lift(y)              # (n,8), one-hot per site
        counts = z.sum(axis=0).astype(int)  # (8,)
        return int(np.dot(w8, counts))

    def V_layers(self, layers: np.ndarray, w8_per_layer: Iterable[np.ndarray]) -> int:
        """
        Layered potential: sum_ell V_{w^{(ell)}}( layer_ell ).
        Useful when you want to penalize certain patterns on x, δ, δ², ...
        """
        total = 0
        for ell, (row, w8) in enumerate(zip(layers, w8_per_layer)):
            total += self.V_triples(row, np.asarray(w8, dtype=int))
        return int(total)

    # ---------- tiny utilities ----------

    def zeros(self) -> np.ndarray:
        """Return the all-zero phase vector (k+1, n)."""
        return np.zeros((self.k + 1, self.n), dtype=bool)

    def random_layers(self, p: float = 0.5) -> np.ndarray:
        """Random phase vector (mainly for quick tests)."""
        return (np.random.rand(self.k + 1, self.n) < p)


# ----------- quick self-checks (optional) -----------
if __name__ == "__main__":
    # Minimal smoke test: rule 110, n=16, k=2
    eca = ECA(110)
    n, k = 16, 2
    ps = PhaseSpace(eca, n=n, k=k)

    # Start from a random x; build layers
    x0 = eca.get_random_state(n, p=0.5)
    layers0 = ps.init_from_x(x0)

    # One step in phase space equals recomputing layers at t+1
    layers1 = ps.step(layers0)

    # Independent construction of layers at t+1 for verification:
    x1 = eca.iterate(x0, C=ps.C)
    layers1_ref = ps.init_from_x(x1)

    assert np.array_equal(layers1, layers1_ref), "Phase step mismatch!"

    # Run a short trajectory
    T = 10
    traj = ps.run(layers0, steps=T)
    print(f"OK: ran {T} steps for rule {eca.rule}, n={n}, k={k}.")
