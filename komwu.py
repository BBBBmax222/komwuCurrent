# komwu.py (Treeplex KOMWU + Variants for EFG poker .game files)
# Implements OMWU, OMWU-L, OMWU-EO, OMWU-EOO, OMWU-DECAY, OMWU-EO-END, OMWU-KD, OMWU-EO-KD
# High-precision ready (pass dtype=np.longdouble).
from __future__ import annotations
import numpy as np

# ===================== numerics =====================

def logsumexp_np(v: np.ndarray) -> np.floating:
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m)))

# ===================== KOMWU base (Treeplex) =====================

class Komwu(object):
    """
    Base KOMWU with optimistic direction d_t = 2 g_t - g_{t-1} (OMWU).
    Variants subclass and override observe_gradient to change the direction only.
    The mapping b -> x uses the treeplex recursion (log-space, numerically stable).
    """
    def __init__(self, tpx, eta: float = 0.3, dtype=None):
        self.tpx = tpx
        self.eta = float(eta)
        self.dtype = dtype if dtype is not None else np.float64

        self.last_gradient = np.zeros(tpx.n_sequences, dtype=self.dtype)

        # Logits in log-space for stability; x computed via treeplex recursion
        self.b = np.zeros(tpx.n_sequences, dtype=self.dtype)
        self._compute_x()

        # Book-keeping for diagnostics (optional)
        self.sum_gradients = np.zeros(tpx.n_sequences, dtype=self.dtype)
        self.sum_ev = self.dtype(0.0)

    # --------- public API ----------
    def next_strategy(self) -> np.ndarray:
        return self.x

    def regret(self) -> float:
        return self.tpx.best_response_value(self.sum_gradients) - float(self.sum_ev)

    def regret_components(self) -> tuple[float, float]:
        """
        Returns (best_response_value(sum_gradients), sum_ev) so that:
            regret() = first - second
        """
        br = float(self.tpx.best_response_value(self.sum_gradients))
        ev = float(self.sum_ev)
        return br, ev

    # --------- core KOMWU update (OMWU) ----------
    def observe_gradient(self, gradient: np.ndarray):
        """
        Update with gradient g_t (gradient of this player's utility wrt sequence probs).
        Direction (OMWU): d_t = 2 g_t - g_{t-1}
        """
        g_t = np.asarray(gradient, dtype=self.dtype)

        # Accumulate for diagnostics
        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        # OMWU two-tap direction
        d_t = (self.dtype(2.0) * g_t) - self.last_gradient

        # KOMWU step in logit space
        self.b += self.dtype(self.eta) * d_t
        self._compute_x()

        # Update memory
        self.last_gradient = g_t

    # --------- treeplex softmax (sequence-form) ----------
    def _compute_x(self):
        """
        Compute sequence-form strategy x from logits b using the treeplex recursion.

        Let K_j(b,1) be the local log-partition at infoset j:
          K_j = log sum_{a in A(j)} exp(b_{ja} + sum_{child I' of ja} K_{I'})

        Then the log flow y_{ja} satisfies:
          y_{ja} = y_parent + b_{ja} + sum_{child I'} K_{I'} - K_j
        with y_root = 0. Finally x = exp(y).
        """
        K_j = [None] * self.tpx.n_infosets
        for infoset_id, infoset in enumerate(self.tpx.infosets):
            terms = []
            for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                child_sum = self.dtype(0.0)
                for child_infoset in self.tpx.children[seq]:
                    child_sum += K_j[child_infoset.infoset_id]
                terms.append(self.b[seq] + child_sum)
            K_j[infoset_id] = logsumexp_np(np.asarray(terms, dtype=self.dtype))

        y = np.zeros(self.tpx.n_sequences, dtype=self.dtype)
        for infoset in reversed(self.tpx.infosets):
            Kj = K_j[infoset.infoset_id]
            y_parent = y[infoset.parent_sequence_id]
            for seq in range(infoset.start_sequence_id, infoset.end_sequence_id + 1):
                child_sum = self.dtype(0.0)
                for child_infoset in self.tpx.children[seq]:
                    child_sum += K_j[child_infoset.infoset_id]
                y[seq] = y_parent + self.b[seq] + child_sum - Kj

        self.x = np.exp(y)
        assert self.tpx.is_sf_strategy(self.x), "x is not a valid sequence-form strategy"

# ===================== Variants that keep KOMWU core =====================

class KomwuL(Komwu):
    """
    OMWU-L:
      d_t = (L+1) g_t − sum_{i=1}^L g_{t−i}
    Efficient via ring buffer + running sum.
    """
    def __init__(self, tpx, eta: float = 0.3, L: int = 5, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.L = int(L)
        self._hist = []                 # most recent first
        self._sum_hist = np.zeros(tpx.n_sequences, dtype=self.dtype)

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        d_t = (self.dtype(self.L) + 1.0) * g_t - self._sum_hist

        # Push AFTER computing direction
        self._hist.insert(0, g_t.copy())
        self._sum_hist += g_t
        if len(self._hist) > self.L:
            old = self._hist.pop()
            self._sum_hist -= old

        self.last_gradient = g_t
        self.b += self.dtype(self.eta) * d_t
        self._compute_x()

class KomwuEO(Komwu):
    """
    OMWU-EO (breadth L):
      d_t = (L+1) g_t − L g_{t−1}
    """
    def __init__(self, tpx, eta: float = 0.3, L: int = 5, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.L = int(L)

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        d_t = (self.dtype(self.L) + 1.0) * g_t - (self.dtype(self.L) * self.last_gradient)

        self.b += self.dtype(self.eta) * d_t
        self._compute_x()
        self.last_gradient = g_t

class KomwuEOO(Komwu):
    """
    OMWU-EOO (staggered EO, breadth L):
      d_t = (L+1) g_t − g_{t−1} − pending_t
      pending_{t+1} = max(L−1,0) · g_{t−1}
    """
    def __init__(self, tpx, eta: float = 0.3, L: int = 5, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.L = int(L)
        self.pending = np.zeros(tpx.n_sequences, dtype=self.dtype)

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        d_t = (self.dtype(self.L) + 1.0) * g_t - self.last_gradient - self.pending

        coef = max(self.L - 1, 0)
        self.pending = self.dtype(coef) * self.last_gradient

        self.b += self.dtype(self.eta) * d_t
        self._compute_x()
        self.last_gradient = g_t

class KomwuDecay(Komwu):
    """
    OMWU-DECAY (split α_b, α_s, decay ρ, length K_f):
      General:
        d_t = α_b g_t − (α_b − 1) g_{t−1}
              + sum_{j=0}^{K_f−1} α_s ρ^j (g_{t−j} − g_{t−j−1})
      With ρ = 1:
        d_t = (α_b+α_s) g_t − (α_b−1) g_{t−1} − α_s g_{t−K_f}
    """
    def __init__(self, tpx, eta: float = 0.3,
                 alpha_b: float = 1.0, alpha_s: float = 1.0, Kf: int = 1, rho: float = 1.0,
                 dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.alpha_b = self.dtype(alpha_b)
        self.alpha_s = self.dtype(alpha_s)
        self.Kf = int(Kf)
        self.rho = self.dtype(rho)

        self._hist = []  # recent gradients, most recent first
        self._w = np.array([self.alpha_s * (self.rho ** j) for j in range(self.Kf)], dtype=self.dtype)

    def _get_hist(self, j: int, default_zero: np.ndarray) -> np.ndarray:
        if 0 <= j < len(self._hist):
            return self._hist[j]
        return default_zero

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        d_t = self.alpha_b * g_t - (self.alpha_b - self.dtype(1.0)) * self.last_gradient

        if self.Kf > 0:
            zeros = np.zeros_like(g_t)
            for j in range(self.Kf):
                g_t_minus_j = g_t if j == 0 else self._get_hist(j - 1, zeros)
                g_t_minus_jm1 = self._get_hist(j, zeros)
                d_t += self._w[j] * (g_t_minus_j - g_t_minus_jm1)

        self._hist.insert(0, g_t.copy())
        if len(self._hist) > (self.Kf + 1):
            self._hist.pop()

        self.b += self.dtype(self.eta) * d_t
        self._compute_x()
        self.last_gradient = g_t


class KomwuEOEnd(Komwu):
    """
    OMWU-EO-END (apply EO only at pre-terminal decision nodes):
      - For sequences inside an infoset whose actions all go directly to terminal leaves:
            d_t = (L+1) * g_t - L * g_{t-1}         (extra optimism)
      - Everywhere else:
            d_t = 2 * g_t - g_{t-1}                 (base OMWU)
    """
    def __init__(self, tpx, eta: float = 0.3, L: int = 2, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.L = int(L)
        # Build a mask over sequences that belong to pre-terminal infosets
        # (i.e., infosets one step above terminal leaves).
        m = np.zeros(tpx.n_sequences, dtype=bool)
        for I in tpx.infosets:
            # An infoset is pre-terminal iff all its actions have no child infosets.
            is_preterminal = True
            for s in range(I.start_sequence_id, I.end_sequence_id + 1):
                if len(tpx.children[s]) > 0:
                    is_preterminal = False
                    break
            if is_preterminal:
                # Mark all sequences (actions) in this infoset
                m[I.start_sequence_id: I.end_sequence_id + 1] = True
        self._eo_mask = m

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        # Diagnostics (kept identical to base class)
        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        # Base OMWU direction
        base = (self.dtype(2.0) * g_t) - self.last_gradient

        # EO direction (used only on pre-terminal sequences)
        eo = (self.dtype(self.L + 1) * g_t) - (self.dtype(self.L) * self.last_gradient)

        # Selectively apply EO at pre-terminal positions
        d_t = np.where(self._eo_mask, eo, base)

        # KOMWU step in logit space
        self.b += self.dtype(self.eta) * d_t
        self._compute_x()

        # Memory update
        self.last_gradient = g_t


class KomwuKD(Komwu):
    """
    OMWU-KD ("keep-drop" OMWU).

    - Direction is the standard OMWU two-tap:
          d_t = 2 g_t - g_{t-1}
    - Maintain a FIFO buffer of recent directions.
    - When the buffer would exceed size K, drop the oldest D directions:
          * Subtract η * d_old for each dropped direction from b.
          * Remove them from the buffer.

    Defaults: K=10, D=4.
    """
    def __init__(self, tpx, eta: float = 0.3, K: int = 10, D: int = 4, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        if D < 0 or K <= 0:
            raise ValueError("K must be > 0 and D must be >= 0.")
        if D > K:
            raise ValueError("Require D <= K for KD variant.")
        self.K = int(K)
        self.D = int(D)
        self._buffer: list[np.ndarray] = []

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        # diagnostics
        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        # standard OMWU direction
        d_t = (self.dtype(2.0) * g_t) - self.last_gradient

        # apply this direction
        self._buffer.append(d_t.copy())
        self.b += self.dtype(self.eta) * d_t

        # keep-drop step: if buffer too long, drop D oldest directions
        if self.D > 0 and len(self._buffer) > self.K:
            n_drop = min(self.D, len(self._buffer))
            for _ in range(n_drop):
                d_old = self._buffer.pop(0)
                self.b -= self.dtype(self.eta) * d_old
            self._buffer: list[np.ndarray] = []

        self._compute_x()
        self.last_gradient = g_t


class KomwuEOKD(Komwu):
    """
    OMWU-EO-KD:

    - EO direction with breadth L:
          d_t = (L+1) g_t - L g_{t-1}
    - Then apply the same keep/drop logic as KomwuKD with parameters (K, D).

    Defaults: L=5, K=10, D=4.
    """
    def __init__(self, tpx, eta: float = 0.3,
                 L: int = 5, K: int = 10, D: int = 4, dtype=None):
        super().__init__(tpx, eta, dtype=dtype)
        self.L = int(L)
        if D < 0 or K <= 0:
            raise ValueError("K must be > 0 and D must be >= 0.")
        if D > K:
            raise ValueError("Require D <= K for KD variant.")
        self.K = int(K)
        self.D = int(D)
        self._buffer: list[np.ndarray] = []

    def observe_gradient(self, gradient: np.ndarray):
        g_t = np.asarray(gradient, dtype=self.dtype)

        # diagnostics
        self.sum_gradients += g_t
        self.sum_ev += g_t.dot(self.next_strategy())

        # EO direction
        d_t = (self.dtype(self.L) + 1.0) * g_t - (self.dtype(self.L) * self.last_gradient)

        # apply direction
        self._buffer.append(d_t.copy())
        self.b += self.dtype(self.eta) * d_t

        # keep-drop step
        if self.D > 0 and len(self._buffer) > self.K:
            n_drop = min(self.D, len(self._buffer))
            for _ in range(n_drop):
                d_old = self._buffer.pop(0)
                self.b -= self.dtype(self.eta) * d_old
            self._buffer: list[np.ndarray] = []


        self._compute_x()
        self.last_gradient = g_t
