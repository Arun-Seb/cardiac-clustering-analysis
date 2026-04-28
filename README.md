# 🫀 Cardiac Sound — Clustering Analysis

Unsupervised discovery of structure in heartbeat audio using state-of-the-art feature extraction and clustering algorithms.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)

---

## 📋 Overview

This project applies **unsupervised clustering** to heartbeat audio recordings to discover natural groupings without using labels — then evaluates how well those clusters align with true cardiac conditions.

| Feature Set | Dimensions | Description |
|---|---|---|
| **Handcrafted** | 604 | MFCCs + deltas, Chroma, Spectral Contrast, Tonnetz, ZCR, RMS |
| **ComParE 2016** | 6,373 | INTERSPEECH clinical audio gold standard |
| **Wav2Vec 2.0** | 1,536 | Facebook's self-supervised transformer embeddings |
| **Combined** | 8,513 | All three fused |

---

## 🗂️ Repository Structure

```
cardiac-clustering-analysis/
│
├── src/
│   ├── dataset.py       # Data loading and label parsing
│   ├── features.py      # Feature extraction (Handcrafted, ComParE, Wav2Vec)
│   ├── clustering.py    # Clustering algorithms and evaluation metrics
│   └── visualise.py     # 2D projections, heatmaps, dendrogram
│
├── notebooks/
│   └── clustering_analysis.ipynb   # Full analysis notebook
│
├── results/             # Output charts (auto-generated)
├── data/samples/        # Put sample .wav files here
├── tests/
│   └── test_clustering.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Arun-Seb/cardiac-clustering-analysis.git
cd cardiac-clustering-analysis
pip install -r requirements.txt
```

Open `notebooks/clustering_analysis.ipynb` and run all cells.

---

## 🔬 Methods

### Dimensionality Reduction

| Method | Type | Key Property |
|---|---|---|
| **PCA** | Linear | Fast, interpretable variance |
| **t-SNE** | Non-linear | Best local structure visualisation |
| **UMAP** | Non-linear | Preserves global structure, faster than t-SNE |

### Clustering Algorithms

| Algorithm | Type | Key Property |
|---|---|---|
| **K-Means** | Partition | Classic, fast, assumes spherical clusters |
| **Agglomerative (Ward)** | Hierarchical | Minimises within-cluster variance |
| **Agglomerative (Complete)** | Hierarchical | Maximises between-cluster distance |
| **GMM** | Probabilistic | Soft assignments, handles elliptical clusters |
| **DBSCAN** | Density-based | Auto-detects clusters, handles noise |

### Evaluation Metrics

| Metric | Needs Labels? | What it measures |
|---|---|---|
| **Silhouette Score** | ❌ | Cluster cohesion and separation |
| **Davies-Bouldin** | ❌ | Average cluster similarity (lower=better) |
| **Calinski-Harabasz** | ❌ | Ratio of between/within cluster variance |
| **ARI** | ✅ | Agreement with true labels (chance-corrected) |
| **NMI** | ✅ | Mutual information with true labels |
| **Homogeneity** | ✅ | Each cluster contains only one class |
| **Completeness** | ✅ | All members of a class are in one cluster |
| **V-Measure** | ✅ | Harmonic mean of homogeneity + completeness |

---

## 📊 Results

### Optimal K Selection (ComParE + PCA 50D)

```
Best k (Silhouette)     : 7
Best k (Davies-Bouldin) : 7
True number of classes  : 5
```

### Algorithm Comparison (k=5, ComParE features)

| Algorithm | Silhouette | ARI | NMI | Homogeneity | Completeness |
|---|---|---|---|---|---|
| K-Means | 0.212 | 0.038 | 0.113 | 0.119 | 0.107 |
| Agglomerative (Ward) | 0.181 | -0.031 | 0.077 | 0.072 | 0.083 |
| Agglomerative (Complete) | 0.370 | 0.059 | 0.057 | 0.035 | 0.150 |
| **GMM** | 0.045 | **0.070** | **0.137** | **0.155** | 0.123 |

### Feature Set Comparison (K-Means, k=5)

| Feature Set | Silhouette | ARI | NMI |
|---|---|---|---|
| **Handcrafted** | **0.310** | **0.130** | **0.226** |
| Wav2Vec 2.0 | 0.192 | 0.069 | 0.147 |
| Combined | 0.180 | 0.056 | 0.145 |
| ComParE | 0.213 | 0.038 | 0.110 |

> 🏆 **Handcrafted features give the best clustering** (ARI: 0.130, NMI: 0.226) — despite ComParE winning in supervised classification. This is common: higher-dimensional features can hurt unsupervised methods due to the curse of dimensionality.

### K-Means Cluster → True Label Mapping

```
Cluster 0 → mostly 'artifact' (39%)
Cluster 1 → mostly 'normal'   (58%)
Cluster 2 → mostly 'normal'   (80%)
Cluster 3 → mostly 'artifact' (100%)
Cluster 4 → mostly 'normal'   (59%)
```

### PCA Variance (ComParE features)
```
50 components  → 74.5% variance
295 components → 95.0% variance  (out of 6,373)
```

---

## 🖼️ Output Charts

| File | Description |
|---|---|
| `clustering_optimal_k.png` | Elbow / Silhouette / Davies-Bouldin / Calinski-Harabasz |
| `clustering_2d_projections.png` | PCA / t-SNE / UMAP — true labels vs K-Means |
| `clustering_all_algorithms.png` | All 5 algorithms on t-SNE |
| `clustering_metrics_heatmap.png` | ARI / NMI / Homogeneity / Completeness heatmap |
| `clustering_dendrogram.png` | Ward linkage dendrogram (120 samples) |
| `clustering_kmeans_mapping.png` | Cluster → true class mapping |
| `clustering_feature_comparison.png` | Silhouette / ARI / NMI by feature set |
| `clustering_pca_variance.png` | Explained variance curve |

---

## 🔑 Key Findings

1. **Optimal k=7** (not 5) — the audio data has more natural groupings than the number of diagnostic labels, suggesting sub-types within classes (e.g. different murmur types).

2. **Handcrafted features cluster better** than ComParE or Wav2Vec despite ComParE winning in supervised classification — high-dimensional features suffer from the curse of dimensionality in unsupervised settings.

3. **Low ARI scores (0.04–0.13)** indicate that the true clinical labels don't map cleanly to geometric clusters in feature space — expected given the class imbalance (351 normal vs 19 extrahls).

4. **DBSCAN found 0 clusters** at eps=3.0 — all points were treated as noise, suggesting the data doesn't have clear density-based boundaries in PCA space.

5. **GMM achieved best ARI (0.070) and NMI (0.137)** among algorithms — probabilistic soft assignments handle the overlapping cardiac sound distributions better than hard partitioning.

---

## 📦 Dependencies

```
librosa, opensmile, transformers, torch
scikit-learn, umap-learn, scipy
pandas, numpy, matplotlib, seaborn
```

---

## 📄 Related Repository

Supervised classification version:
👉 [cardiac-sound-classifier](https://github.com/Arun-Seb/cardiac-sound-classifier)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Arun** — [github.com/Arun-Seb](https://github.com/Arun-Seb)
