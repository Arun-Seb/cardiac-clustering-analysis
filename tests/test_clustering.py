"""
test_clustering.py
──────────────────
Unit tests for clustering utilities.
Run with: pytest tests/
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.clustering import compute_metrics, find_optimal_k, make_clusterers


def make_blobs(n=100, k=3, d=10, seed=42):
    """Generate simple blob data for testing."""
    np.random.seed(seed)
    X = np.vstack([np.random.randn(n//k, d) + i*5 for i in range(k)])
    y = np.repeat(np.arange(k), n//k)
    return X.astype(np.float64), y


class TestComputeMetrics:

    def test_returns_all_keys(self):
        X, y = make_blobs()
        labels = np.repeat(np.arange(3), len(X)//3)
        m = compute_metrics(X, labels, y)
        for key in ["Silhouette","ARI","NMI","Homogeneity","Completeness","V-Measure"]:
            assert key in m, f"Missing key: {key}"

    def test_silhouette_range(self):
        X, y = make_blobs()
        labels = np.repeat(np.arange(3), len(X)//3)
        m = compute_metrics(X, labels)
        assert -1 <= m["Silhouette"] <= 1

    def test_perfect_clustering(self):
        X, y = make_blobs()
        m = compute_metrics(X, y, y)
        assert m["ARI"] == pytest.approx(1.0, abs=0.01)
        assert m["NMI"] == pytest.approx(1.0, abs=0.01)

    def test_all_noise_dbscan(self):
        X, y = make_blobs()
        labels = np.full(len(X), -1)   # all noise
        m = compute_metrics(X, labels, y)
        assert m["Silhouette"] == 0.0

    def test_no_true_labels(self):
        X, _ = make_blobs()
        labels = np.repeat(np.arange(3), len(X)//3)
        m = compute_metrics(X, labels, y_true=None)
        assert m["ARI"] == 0.0
        assert m["Silhouette"] != 0.0


class TestFindOptimalK:

    def test_output_keys(self):
        X, _ = make_blobs()
        r = find_optimal_k(X, k_range=range(2, 5))
        for key in ["k","inertia","silhouette","davies_bouldin","calinski_harabasz"]:
            assert key in r

    def test_inertia_decreasing(self):
        X, _ = make_blobs()
        r = find_optimal_k(X, k_range=range(2, 7))
        assert all(r["inertia"][i] >= r["inertia"][i+1]
                   for i in range(len(r["inertia"])-1))

    def test_correct_k_range(self):
        X, _ = make_blobs()
        r = find_optimal_k(X, k_range=range(2, 6))
        assert r["k"] == list(range(2, 6))


class TestMakeClusterers:

    def test_returns_dict(self):
        c = make_clusterers(3)
        assert isinstance(c, dict)
        assert len(c) == 4

    def test_all_fit(self):
        X, _ = make_blobs()
        for name, clf in make_clusterers(3).items():
            if hasattr(clf, "fit_predict"):
                labels = clf.fit_predict(X)
            else:
                clf.fit(X)
                labels = clf.predict(X)
            assert len(labels) == len(X), f"{name} returned wrong label count"
