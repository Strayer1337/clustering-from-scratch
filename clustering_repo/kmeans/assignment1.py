"""
Assignment 1 (2 điểm):
- Tạo toy dataset 600 điểm: 200 điểm từ N((2,2), I), 200 từ N((8,3), I), 200 từ N((3,6), I).
- Dùng Numpy implement K-Means với EM method.
- Nhận xét ảnh hưởng của khởi tạo tâm ngẫu nhiên đến hiệu năng K-Means.
"""

import numpy as np
from kmeans import KMeans
from plot_utils import plot_data, plot_kmeans_result

# --- Tạo dữ liệu ---
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 200

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)
n_clusters = 3
original_labels = np.asarray([0] * N + [1] * N + [2] * N).T

# --- Visualize dữ liệu gốc ---
plot_data(X, original_labels, title="Original Data Distribution")

# --- Train model ---
model = KMeans()
labels = model.fit(X, n_clusters)
centroids = model.centroids[-1]

# --- Visualize kết quả ---
plot_kmeans_result(X, labels, centroids, title="K-Means Clustering Result")

# --- Nhận xét ---
# Nhận xét — Ảnh hưởng của khởi tạo tâm ngẫu nhiên (K-Means Assignment 1):
#
# Khi ba cụm có số lượng điểm bằng nhau (200 điểm/cụm) và covariance là ma trận
# đơn vị, K-Means thường phân cụm khá chính xác. Tuy nhiên, do tâm ban đầu được
# chọn ngẫu nhiên, thuật toán có thể gặp các vấn đề:
#
# - Hội tụ về local optimum: Nếu hai tâm ban đầu cùng nằm gần một cụm thực,
#   K-Means có thể chia cụm đó làm hai và gộp hai cụm còn lại thành một.
# - Số vòng lặp không ổn định: Khởi tạo tốt → hội tụ nhanh; khởi tạo xấu →
#   cần nhiều iteration hơn.
# - Kết quả không tái lập: Nếu không cố định random_state, mỗi lần chạy có
#   thể cho nhãn cụm khác nhau.
#
# Kết luận: Với dữ liệu tách biệt rõ, ảnh hưởng của khởi tạo ngẫu nhiên là nhỏ.
# Để ổn định hơn, nên dùng K-Means++ hoặc chạy nhiều lần và chọn inertia nhỏ nhất.
