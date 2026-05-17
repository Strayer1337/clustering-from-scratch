import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, random_state=None):
        self.centroids = []
        self.random_state = random_state

    def _init_centroids(self, X, n_clusters):
        np.random.seed(self.random_state)
        return X[
            np.random.choice(X.shape[0], n_clusters, replace=False)
        ]

    def _assign_labels(self, X, centroids):
        distance = cdist(X, centroids)
        return np.argmin(distance, axis=1)

    def _update_centroids(self, X, labels, n_clusters):
        centroids = np.zeros((n_clusters, X.shape[1]))
        for cluster in range(n_clusters):
            Xk = X[labels == cluster]
            if len(Xk) > 0:
                centroids[cluster] = np.mean(Xk, axis=0)
            else:
                centroids[cluster] = X[np.random.randint(0, X.shape[0])]
        return centroids

    def has_converged(self, old_centroids, new_centroids):
        return np.allclose(old_centroids, new_centroids)

    def fit(self, X, n_clusters, max_iter=1000):
        current_centroids = self._init_centroids(X, n_clusters)
        self.centroids.append(current_centroids.copy())

        for _ in range(max_iter):
            labels = self._assign_labels(X, current_centroids)
            new_centroids = self._update_centroids(X, labels, n_clusters)
            self.centroids.append(new_centroids.copy())

            if self.has_converged(current_centroids, new_centroids):
                break

            current_centroids = new_centroids

        self.labels_ = labels
        self.cluster_centers_ = current_centroids
        return labels
