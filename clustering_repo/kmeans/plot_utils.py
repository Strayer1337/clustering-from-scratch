import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, labels, title="Data Distribution"):
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels, cmap='Set2',
        s=70, alpha=0.8,
        edgecolors='black', linewidths=0.5
    )
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_kmeans_result(X, labels, centroids, title="K-Means Clustering Result"):
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X[:, 0], X[:, 1],
        c=labels, cmap='Set2',
        s=70, alpha=0.8,
        edgecolors='black', linewidths=0.5
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red', marker='X',
        s=350, edgecolors='black', linewidths=2,
        label='Centroids'
    )
    for i, (x, y) in enumerate(centroids):
        plt.text(x, y + 0.3, f'C{i}', fontsize=12, fontweight='bold', ha='center')
    plt.title(title, fontsize=18, fontweight='bold')
    plt.grid(linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
