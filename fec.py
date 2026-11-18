from __future__ import annotations
from typing import Optional
import numpy as np



class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC), following the handout."""

    def __init__(self, k: int = 10, sigma: Optional[float] = None, alpha: float = 0.5,
                 T: int = 25, epsilon: float = 1e-4, beta: Optional[float] = None) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError("X must be a non-empty 2-D array.")
        n = X.shape[0]

        # Step 1: pre-compute
        centroid = X.mean(axis=0)
        d = np.linalg.norm(X - centroid, axis=1)
        sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

        if n == 1:                                  
            self.w_ij = np.zeros((1, 0)); self.flux = np.ones(1)
            self.sinks = np.ones(1, dtype=bool); self.labels = np.zeros(1, dtype=int)
            return self

        sigma = self.sigma                         
        if sigma is None:
            sigma = 0.2 * np.median(np.sqrt(sq[np.triu_indices(n, 1)]))
        if sigma <= 0:
            sigma = 1.0                              
        tau = 0.1 * sigma

        k_eff = min(self.k, n - 1)
        sq_nn = sq.copy()
        np.fill_diagonal(sq_nn, np.inf)              
        knn = np.argsort(sq_nn, axis=1, kind="stable")[:, :k_eff]

        rows = np.arange(n)[:, None]
        w_ij = np.exp(-sq[rows, knn] / sigma ** 2) 
        W = np.zeros((n, n))
        W[rows, knn] = w_ij
        W = np.maximum(W, W.T)                  

        downhill = [knn[i][d[knn[i]] <= d[i] + tau] for i in range(n)]

  
        f = np.ones(n)
        delta = np.zeros((n, n))
        for t in range(self.T):
            delta = np.zeros((n, n))
            for i in range(n):
                nb = downhill[i]
                if nb.size == 0:
                    continue
                if t == 0 or self.beta == 0:
                    phi = np.ones(nb.size)
                elif self.beta is None:
                    phi = np.log1p(f[nb])          
                else:
                    phi = f[nb] ** self.beta
                w = W[i, nb] * phi
                s = w.sum()
                if s > 0:
                    delta[i, nb] = self.alpha * f[i] * w / s
            f = f - delta.sum(axis=1) + delta.sum(axis=0)
        flux = f.copy()


        outgoing = delta.sum(axis=1)
        sinks = outgoing < self.epsilon
        if not sinks.any():
            sinks[np.argmin(d)] = True

        labels = np.full(n, -1, dtype=int)
        sink_idx = np.where(sinks)[0]
        labels[sink_idx] = np.arange(sink_idx.size)
        next_label = sink_idx.size


        steepest = np.full(n, -1, dtype=int)
        for i in range(n):
            if not sinks[i]:
                steepest[i] = np.argmax(delta[i])

        for i in range(n):
            if labels[i] != -1:
                continue
            path, seen, cur = [], set(), i
            while labels[cur] == -1:
                if cur in seen:                      
                    labels[cur] = next_label
                    sinks[cur] = True
                    next_label += 1
                    break
                seen.add(cur)
                path.append(cur)
                cur = steepest[cur]
            lab = labels[cur]
            for node in path:
                if labels[node] == -1:
                    labels[node] = lab


        s_min = max(2, n // 20)
        ids, counts = np.unique(labels, return_counts=True)
        cents = {c: X[labels == c].mean(axis=0) for c in ids}
        large = [c for c, m in zip(ids, counts) if m >= s_min]
        if large:
            for c, m in zip(ids, counts):
                if m < s_min:
                    tgt = min(large, key=lambda o: np.linalg.norm(cents[c] - cents[o]))
                    labels[labels == c] = tgt

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (f"k={getattr(self,'k','?')}", f"sigma={getattr(self,'sigma','?')}",
                  f"alpha={getattr(self,'alpha','?')}", f"T={getattr(self,'T','?')}",
                  f"epsilon={getattr(self,'epsilon','?')}", f"beta={getattr(self,'beta','?')}")
        return f"{self.__class__.__name__}({', '.join(params)})"
