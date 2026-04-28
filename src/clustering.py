"""
clustering.py
─────────────
Clustering algorithms and evaluation metrics
for cardiac sound analysis.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score,
)

RANDOM_STATE = 42


def make_clusterers(n_clusters: int) -> dict:
    """
    Return all clustering algorithms ready to fit.

    Args:
        n_clusters : number of clusters (typically = number of classes)

    Returns:
        dict of {name: fitted sklearn clusterer}
    """
    return {
        "K-Means": KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=20,
        ),
        "Agglomerative (Ward)": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        ),
        "Agglomerative (Complete)": AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="complete",
        ),
        "GMM": GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",   # stable on high-dim PCA data
            reg_covar=1e-3,           # regularise to prevent singular covariance
            n_init=5,
            random_state=RANDOM_STATE,
        ),
    }


def run_clustering(X: np.ndarray, n_clusters: int,
                   y_true: np.ndarray = None) -> dict:
    """
    Fit all clustering algorithms and compute evaluation metrics.

    Args:
        X          : feature matrix (float64, already scaled + PCA reduced)
        n_clusters : number of clusters
        y_true     : true integer labels (optional, for ARI/NMI/etc.)

    Returns:
        dict with keys = algorithm names,
        values = {"labels": np.ndarray, "metrics": dict}
    """
    results = {}

    for name, clf in make_clusterers(n_clusters).items():
        if hasattr(clf, "fit_predict"):
            labels = clf.fit_predict(X)
        else:
            clf.fit(X)
            labels = clf.predict(X)

        metrics = compute_metrics(X, labels, y_true)
        results[name] = {"labels": labels, "metrics": metrics}
        print(f"  {name:<28} Sil:{metrics['Silhouette']:.3f}  "
              f"ARI:{metrics['ARI']:.3f}  NMI:{metrics['NMI']:.3f}")

    # DBSCAN — density-based, auto-detects clusters
    db_labels = DBSCAN(eps=3.0, min_samples=5).fit_predict(X)
    n_found   = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise   = (db_labels == -1).sum()
    db_metrics = compute_metrics(X, db_labels, y_true)
    results["DBSCAN"] = {"labels": db_labels, "metrics": db_metrics}
    print(f"  {'DBSCAN':<28} Clusters:{n_found}  Noise:{n_noise}")

    return results


def compute_metrics(X: np.ndarray, labels: np.ndarray,
                    y_true: np.ndarray = None) -> dict:
    """
    Compute internal and external clustering metrics.

    Internal (no labels needed): Silhouette, Davies-Bouldin, Calinski-Harabasz
    External (labels needed):    ARI, NMI, Homogeneity, Completeness, V-Measure
    """
    valid = labels != -1   # exclude DBSCAN noise points
    metrics = {
        "Silhouette": 0.0, "Davies-Bouldin": 0.0, "Calinski-Harabasz": 0.0,
        "ARI": 0.0, "NMI": 0.0, "Homogeneity": 0.0,
        "Completeness": 0.0, "V-Measure": 0.0,
    }

    if valid.sum() < 2:
        return metrics

    X_v = X[valid]
    l_v = labels[valid]

    # Internal metrics
    try:
        metrics["Silhouette"]          = silhouette_score(X_v, l_v)
        metrics["Davies-Bouldin"]      = davies_bouldin_score(X_v, l_v)
        metrics["Calinski-Harabasz"]   = calinski_harabasz_score(X_v, l_v)
    except Exception:
        pass

    # External metrics (require ground truth)
    if y_true is not None:
        y_v = y_true[valid]
        metrics["ARI"]          = adjusted_rand_score(y_v, l_v)
        metrics["NMI"]          = normalized_mutual_info_score(y_v, l_v)
        metrics["Homogeneity"]  = homogeneity_score(y_v, l_v)
        metrics["Completeness"] = completeness_score(y_v, l_v)
        metrics["V-Measure"]    = v_measure_score(y_v, l_v)

    return metrics


def find_optimal_k(X: np.ndarray, k_range=range(2, 11)) -> dict:
    """
    Run K-Means for each k and return Elbow, Silhouette,
    Davies-Bouldin, and Calinski-Harabasz scores.
    """
    results = {"k": [], "inertia": [], "silhouette": [],
               "davies_bouldin": [], "calinski_harabasz": []}

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X, labels))
        results["davies_bouldin"].append(davies_bouldin_score(X, labels))
        results["calinski_harabasz"].append(calinski_harabasz_score(X, labels))

    best_k_sil = results["k"][np.argmax(results["silhouette"])]
    best_k_db  = results["k"][np.argmin(results["davies_bouldin"])]

    print(f"  Best k (Silhouette)     : {best_k_sil}")
    print(f"  Best k (Davies-Bouldin) : {best_k_db}")

    return results
