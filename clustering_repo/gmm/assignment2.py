"""
Assignment 2 (2 điểm):
- Dùng Gaussian Mixture Model (sklearn) để tách nền ảnh cow.jpg.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# --- Load ảnh ---
img = cv2.imread("cow.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape

# Reshape ảnh thành tập điểm dữ liệu RGB
X = img.reshape(-1, 3).astype(np.float64) / 255.0

# --- Fit GMM (sklearn) ---
model = GaussianMixture(n_components=2, random_state=42)
model.fit(X)
labels = model.predict(X)

segmented = labels.reshape(h, w)

# Xác định cụm nền dựa trên pixels viền ảnh
border_pixels = np.concatenate([
    labels[:w],
    labels[-w:],
    labels[::w],
    labels[w - 1::w]
])
background_cluster = np.bincount(border_pixels).argmax()

mask = segmented != background_cluster

result = img.copy()
result[~mask] = [0, 0, 0]

# --- Visualize kết quả ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title("Mask (sklearn GMM)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Kết quả tách nền")
plt.axis("off")

plt.tight_layout()
plt.show()
