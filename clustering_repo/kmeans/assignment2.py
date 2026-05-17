"""
Assignment 2 (2 điểm):
- Tạo toy dataset mất cân bằng: 1200 từ N((2,2), I), 200 từ N((8,3), I), 1000 từ N((3,6), I).
- Dùng Numpy implement K-Means với EM method.
- Nhận xét ảnh hưởng của kích thước cụm không đều đến hiệu năng K-Means.
"""

import numpy as np
from kmeans import KMeans
from plot_utils import plot_data, plot_kmeans_result

# --- Tạo dữ liệu mất cân bằng ---
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, 1200)
X1 = np.random.multivariate_normal(means[1], cov, 200)
X2 = np.random.multivariate_normal(means[2], cov, 1000)

X = np.concatenate((X0, X1, X2), axis=0)
n_clusters = 3
original_labels = np.asarray([0] * 1200 + [1] * 200 + [2] * 1000).T

# --- Visualize dữ liệu gốc ---
plot_data(X, original_labels, title="Original Data Distribution (Imbalanced)")

# --- Train model ---
model = KMeans()
labels = model.fit(X, n_clusters)
centroids = model.centroids[-1]

# --- Visualize kết quả ---
plot_kmeans_result(X, labels, centroids, title="K-Means Clustering Result")

# --- Nhận xét ---
# Nhận xét — Ảnh hưởng của kích thước cụm không đều (K-Means Assignment 2):
#
# Dataset có sự mất cân bằng rõ rệt: cụm 0 có 1200 điểm, cụm 1 có 200 điểm,
# cụm 2 có 1000 điểm. K-Means chuẩn giả định các cụm có kích thước tương đương:
#
# - Thiên lệch tâm cụm lớn: Tâm của cụm lớn bị "kéo" ra xa tâm thực khi bị
#   ảnh hưởng bởi các điểm biên giới với cụm khác.
# - Cụm nhỏ dễ bị "nuốt": Cụm 1 chỉ có 200 điểm — nếu tâm khởi tạo rơi gần
#   cụm 0 hoặc 2, điểm của cụm 1 có thể bị gán vào cụm lớn hơn.
# - Ranh giới phân cụm sai: K-Means dùng khoảng cách Euclidean và không xét
#   xác suất, nên ranh giới Voronoi không phản ánh đúng xác suất hậu nghiệm.
#
# Kết luận: K-Means hoạt động kém hơn khi kích thước các cụm chênh lệch nhiều.
# GMM phù hợp hơn vì có tham số pi_k mô hình hóa tỉ lệ của từng cụm.
