"""
visualise.py
────────────
Visualisation utilities for clustering analysis:
  - 2D scatter plots (PCA, t-SNE, UMAP)
  - Optimal k charts (Elbow, Silhouette, DB, CH)
  - Metrics heatmap
  - Dendrogram
  - Cluster-to-class mapping
  - Feature set comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage

LABEL_COLORS = {
    "normal"    : "#4C72B0",
    "murmur"    : "#DD8452",
    "extrastole": "#55A868",
    "artifact"  : "#C44E52",
    "extrahls"  : "#8172B2",
}
PALETTE_CL = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
               "#937860","#DA8BC3","#8C8C8C"]


def plot_optimal_k(k_results: dict, output_path: Path = None):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    k = k_results["k"]

    axes[0].plot(k, k_results["inertia"], "o-", color="#4C72B0", lw=2)
    axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia"); axes[0].grid(alpha=0.3)

    best_sil = k[np.argmax(k_results["silhouette"])]
    axes[1].plot(k, k_results["silhouette"], "o-", color="#55A868", lw=2)
    axes[1].axvline(best_sil, color="red", linestyle="--", alpha=0.7,
                    label=f"Best k={best_sil}")
    axes[1].set_title("Silhouette (higher=better)")
    axes[1].set_xlabel("k"); axes[1].legend(); axes[1].grid(alpha=0.3)

    best_db = k[np.argmin(k_results["davies_bouldin"])]
    axes[2].plot(k, k_results["davies_bouldin"], "o-", color="#DD8452", lw=2)
    axes[2].axvline(best_db, color="red", linestyle="--", alpha=0.7,
                    label=f"Best k={best_db}")
    axes[2].set_title("Davies-Bouldin (lower=better)")
    axes[2].set_xlabel("k"); axes[2].legend(); axes[2].grid(alpha=0.3)

    axes[3].plot(k, k_results["calinski_harabasz"], "o-", color="#C44E52", lw=2)
    axes[3].set_title("Calinski-Harabasz (higher=better)")
    axes[3].set_xlabel("k"); axes[3].grid(alpha=0.3)

    fig.suptitle("Optimal Number of Clusters — ComParE + PCA 50D", fontsize=13)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_2d_projections(projections: list, y_labels, km_labels,
                        class_names, n_classes, output_path=None):
    """
    projections: list of (name, X_2d) tuples
    Row 0: true labels  |  Row 1: K-Means clusters
    """
    fig, axes = plt.subplots(2, len(projections),
                             figsize=(6*len(projections), 11))
    for col, (proj_name, X_2d) in enumerate(projections):
        for label in class_names:
            mask = y_labels == label
            axes[0, col].scatter(X_2d[mask,0], X_2d[mask,1],
                                 c=LABEL_COLORS[label], label=label,
                                 s=25, alpha=0.75, edgecolors="none")
        axes[0, col].set_title(f"{proj_name} — True Labels",
                               fontsize=11, fontweight="bold")
        axes[0, col].legend(fontsize=8)
        axes[0, col].set_xlabel(f"{proj_name} 1")
        axes[0, col].set_ylabel(f"{proj_name} 2")

        for k in range(n_classes):
            mask = km_labels == k
            axes[1, col].scatter(X_2d[mask,0], X_2d[mask,1],
                                 c=PALETTE_CL[k], label=f"Cluster {k}",
                                 s=25, alpha=0.75, edgecolors="none")
        axes[1, col].set_title(f"{proj_name} — K-Means (k={n_classes})",
                               fontsize=11, fontweight="bold")
        axes[1, col].legend(fontsize=8)
        axes[1, col].set_xlabel(f"{proj_name} 1")
        axes[1, col].set_ylabel(f"{proj_name} 2")

    fig.suptitle("2D Projections\nRow 1: True Labels  |  Row 2: K-Means",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_metrics_heatmap(metrics_df, output_path=None):
    metric_cols = ["Silhouette","ARI","NMI","Homogeneity","Completeness","V-Measure"]
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(metrics_df.set_index("Algorithm")[metric_cols],
                annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Clustering Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_dendrogram(X_pca, y_labels, class_names,
                    n_samples=120, output_path=None):
    np.random.seed(42)
    idx      = np.random.choice(len(X_pca), size=min(n_samples, len(X_pca)),
                                replace=False)
    Z        = linkage(X_pca[idx], method="ward")
    fig, ax  = plt.subplots(figsize=(18, 6))
    dendrogram(Z, ax=ax, labels=y_labels[idx], leaf_font_size=7,
               color_threshold=0.7*max(Z[:,2]),
               above_threshold_color="gray")
    ax.set_title("Hierarchical Clustering Dendrogram — Ward Linkage",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.legend(handles=[Patch(facecolor=LABEL_COLORS[l], label=l)
                        for l in class_names],
              loc="upper right", fontsize=9)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
