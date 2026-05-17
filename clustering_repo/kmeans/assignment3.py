"""
Assignment 3 (2 điểm):
- Tạo toy dataset 600 điểm: 200 từ N((2,2), Sigma1), 200 từ N((8,3), Sigma1),
  200 từ N((3,6), Sigma2) với Sigma1=I và Sigma2=diag(10,1).
- Dùng Numpy implement K-Means với EM method.
- Nhận xét ảnh hưởng của phân phối N((3,6), Sigma2) đến hiệu năng K-Means.
"""

import numpy as np
from kmeans import KMeans
from plot_utils import plot_data, plot_kmeans_result

# --- Tạo dữ liệu với covariance dị hướng ---
means = [[2, 2], [8, 3], [3, 6]]
cov1 = [[1, 0], [0, 1]]
cov2 = [[10, 0], [0, 1]]
N = 200

X0 = np.random.multivariate_normal(means[0], cov1, N)
X1 = np.random.multivariate_normal(means[1], cov1, N)
X2 = np.random.multivariate_normal(means[2], cov2, N)

X = np.concatenate((X0, X1, X2), axis=0)
n_clusters = 3
original_labels = np.asarray([0] * N + [1] * N + [2] * N).T

# --- Visualize dữ liệu gốc ---
plot_data(X, original_labels, title="Original Data Distribution (Anisotropic Covariance)")

# --- Train model ---
model = KMeans()
labels = model.fit(X, n_clusters)
centroids = model.centroids[-1]

# --- Visualize kết quả ---
plot_kmeans_result(X, labels, centroids, title="K-Means Clustering Result")

# --- Nhận xét ---
# Nhận xét — Ảnh hưởng của phân phối N((3,6), Sigma2) có covariance lớn (Assignment 3):
#
# Cụm thứ ba được sinh từ Sigma2 = diag(10, 1), nghĩa là dữ liệu trải rộng theo
# chiều x (std ≈ 3.16) nhưng hẹp theo chiều y. Điều này gây ra:
#
# - Cụm bị trải dài (elongated cluster): K-Means giả định cụm có dạng hình cầu
#   (isotropic). Khi cụm có hình dạng ellipsoid, ranh giới Voronoi không khớp
#   với hình dạng thực.
# - Điểm biên giới bị gán sai: Các điểm ở rìa xa theo chiều x của cụm 2 có thể
#   gần tâm cụm 0 hơn về Euclidean, dẫn đến bị gán nhầm.
# - Tâm cụm bị lệch: Tâm ước lượng có thể bị kéo khỏi mean thực (3, 6).
#
# Kết luận: K-Means không phù hợp với dữ liệu có covariance phi đẳng hướng
# (anisotropic). GMM với full covariance sẽ xử lý tốt hơn.
