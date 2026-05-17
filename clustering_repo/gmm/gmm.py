import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, K, max_iter=100, tol=1e-6, random_state=42):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _e_step(self, X):
        N = X.shape[0]
        K = self.K
        numerator = np.zeros((N, K))

        for k in range(K):
            numerator[:, k] = self.pi[k] * multivariate_normal.pdf(
                X, mean=self.mu[k], cov=self.sigma[k]
            )

        denominator = numerator.sum(axis=1, keepdims=True)
        gamma = numerator / (denominator + 1e-100)
        return gamma

    def _m_step(self, X, gamma):
        N, D = X.shape
        K = self.K
        N_k = gamma.sum(axis=0)

        for k in range(K):
            # Cập nhật mu_k
            self.mu[k] = (gamma[:, k] @ X) / N_k[k]

            # Cập nhật sigma_k
            diff = X - self.mu[k]
            self.sigma[k] = (gamma[:, k] * diff.T) @ diff / N_k[k]

        # Cập nhật pi_k (nằm ngoài vòng for)
        self.pi = N_k / N

    def log_likelihood(self, X):
        total = np.zeros(X.shape[0])
        for k in range(self.K):
            total += self.pi[k] * multivariate_normal.pdf(
                X, mean=self.mu[k], cov=self.sigma[k]
            )
        return np.sum(np.log(total + 1e-100))

    def fit(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        K = self.K

        # Chọn ngẫu nhiên K điểm trong N điểm làm tâm ban đầu
        idx = np.random.choice(N, K, replace=False)
        self.mu = X[idx].copy()                  # (K, D)
        self.sigma = np.array([np.eye(D)] * K)   # (K, D, D)
        self.pi = np.ones(K) / K                 # (K,)

        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            # Bước E: tính gamma_k(x_n)
            gamma = self._e_step(X)              # (N, K)

            # Bước M: cập nhật tham số
            self._m_step(X, gamma)

            # Tính log-likelihood
            ll = self.log_likelihood(X)
            self.log_likelihoods.append(ll)

            # Kiểm tra hội tụ
            if iteration > 0 and abs(ll - self.log_likelihoods[-2]) < self.tol:
                print(f"Hội tụ sau {iteration + 1} vòng lặp.")
                break

    def predict(self, X):
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)
