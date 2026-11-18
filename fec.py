from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:
        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta
        # Hint: self.k = k, self.sigma = sigma, etc.


    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Hint: Start with computing centroid = X.mean(axis=0)

        centroid = X.mean(axis=0)
        n = X.shape[0]
        if n < 2:
            raise ValueError("X must contain at least two data points.")
        squared_distances = np.zeros((n,n))
        distances = []
        distances_to_centroid = np.linalg.norm(X - centroid, axis=1)

        for i in range(n):
            for j in range(n):
                squared_distances[i,j] = np.sum((X[i] - X[j]) ** 2)

        if self.sigma is None:
            for i in range(n):
                for j in range(i+1, n):
                    distance = np.linalg.norm(X[i] - X[j])
                    distances.append(distance)
            self.sigma = 0.3 * np.median(distances)

        effective_k = min(self.k, n - 1)
        knn_indices = np.argsort(squared_distances, axis=1)[:, 1:effective_k + 1]

        w_ij = np.zeros((n, effective_k))
        W = np.zeros((n, n))

        for i in range(n):
            for position, j in enumerate(knn_indices[i]):
                weight = np.exp(
                    -squared_distances[i, j] / self.sigma ** 2
                )

                w_ij[i, position] = weight
                W[i, j] = weight


        W = np.maximum(W, W.T)

        f_i = np.ones(n)
        flux = f_i.copy()

        for iteration in range(self.T):

            delta = np.zeros((n, n))

            for j in range(n):

                downhill_neighbors = []

                for ind in knn_indices[j]:
                    if distances_to_centroid[ind] <= (
                            distances_to_centroid[j] + 0.05 * self.sigma
                    ):
                        downhill_neighbors.append(ind)

                if downhill_neighbors:

                    downhill_weights = W[j, downhill_neighbors]

                    # Apply flux-aware reweighting
                    if self.beta is None:
                        phi = np.log1p(f_i[j])
                    elif self.beta == 0:
                        phi = 1.0
                    else:
                        phi = f_i[j] ** self.beta

                    reweighted_weights = downhill_weights * phi

                    total_weight = np.sum(reweighted_weights)

                    if total_weight > 0:
                        delta[j, downhill_neighbors] = (
                                self.alpha
                                * f_i[j]
                                * reweighted_weights / total_weight
                        )

            outgoing = np.sum(delta, axis=1)
            incoming = np.sum(delta, axis=0)

            new_flux = f_i - outgoing + incoming

            if np.max(np.abs(new_flux - f_i)) < self.epsilon:
                f_i = new_flux
                flux = f_i.copy()
                break

            f_i = new_flux
            flux = f_i.copy()

        outgoing_flux = np.sum(delta, axis=1)
        sinks = outgoing_flux <= self.epsilon * np.maximum(f_i, 1e-12)
        sinks |= outgoing_flux == 0

        labels = np.full(n, -1, dtype=int)
        sink_indices = np.where(sinks)[0]
        labels[sink_indices] = np.arange(len(sink_indices))
        next_label = len(sink_indices)

        next_node = np.full(n, -1, dtype=int)
        for i in range(n):
            if not sinks[i]:
                next_node[i] = np.argmax(delta[i])

        for i in range(n):
            if labels[i] != -1:
                continue
            path = []
            seen = set()
            current = i
            cycle = False
            while labels[current] == -1:
                if current in seen:
                    cycle = True
                    break
                seen.add(current)
                path.append(current)
                current = next_node[current]

            if cycle:
                path_centroid = X[path].mean(axis=0)
                nearest_sink = sink_indices[
                    np.argmin(np.linalg.norm(X[sink_indices] - path_centroid, axis=1))
                ]
                cluster_label = labels[nearest_sink]
            else:
                cluster_label = labels[current]

            for node in path:
                if labels[node] == -1:
                    labels[node] = cluster_label
        smin = max(2, int(np.ceil(n / 20)))
        ids, counts = np.unique(labels, return_counts=True)
        centroids = {c: X[labels == c].mean(axis=0) for c in ids}
        large = [c for c, cnt in zip(ids, counts) if cnt >= smin]

        if large:
            for c, cnt in zip(ids, counts):
                if cnt < smin:
                    nearest = min(
                        large, key=lambda o: np.linalg.norm(centroids[c] - centroids[o])
                    )
                    labels[labels == c] = nearest

        _, labels = np.unique(labels, return_inverse=True)
        print(sink_indices)
        print(labels)
        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
