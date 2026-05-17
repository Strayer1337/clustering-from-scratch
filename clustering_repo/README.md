# Clustering Assignments

Repo này chứa implementation của **K-Means** và **Gaussian Mixture Model (GMM)** từ bài tập môn học, được tổ chức thành các module Python riêng biệt.

## Cấu trúc repo

```
clustering_repo/
├── kmeans/
│   ├── kmeans.py        # Class KMeans (numpy only)
│   ├── plot_utils.py    # Hàm visualize dùng chung
│   ├── assignment1.py   # Balanced dataset (200 × 3 cụm)
│   ├── assignment2.py   # Imbalanced dataset (1200 / 200 / 1000)
│   └── assignment3.py   # Anisotropic covariance (Sigma2 = diag(10,1))
├── gmm/
│   ├── gmm.py           # Class GMM (numpy only, EM method)
│   ├── assignment1.py   # Tách nền ảnh dùng GMM tự implement
│   └── assignment2.py   # Tách nền ảnh dùng sklearn GaussianMixture
└── README.md
```

## Yêu cầu

```
numpy
scipy
matplotlib
opencv-python
scikit-learn
```

Cài đặt:

```bash
pip install numpy scipy matplotlib opencv-python scikit-learn
```

## Cách chạy

Chạy từng assignment bằng cách `cd` vào thư mục tương ứng:

```bash
# K-Means assignments
cd kmeans
python assignment1.py
python assignment2.py
python assignment3.py

# GMM assignments (cần có file cow.jpg trong thư mục gmm/)
cd gmm
python assignment1.py
python assignment2.py
```

> **Lưu ý:** Các assignment GMM yêu cầu file `cow.jpg` nằm trong thư mục `gmm/`.

## Tóm tắt nội dung

### K-Means

| Assignment | Dataset | Chủ đề nhận xét |
|---|---|---|
| 1 | 3 cụm cân bằng (200×3) | Ảnh hưởng của khởi tạo tâm ngẫu nhiên |
| 2 | 3 cụm mất cân bằng (1200/200/1000) | Ảnh hưởng của kích thước cụm không đều |
| 3 | Covariance dị hướng (Σ₂ = diag(10,1)) | Ảnh hưởng khi cụm có hình ellipsoid |

### GMM

| Assignment | Phương pháp | Mục tiêu |
|---|---|---|
| 1 | Numpy (tự implement EM) | Implement GMM từ đầu, tách nền ảnh |
| 2 | sklearn GaussianMixture | So sánh với thư viện chuẩn |
