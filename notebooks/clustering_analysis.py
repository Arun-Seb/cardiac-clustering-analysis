# ============================================================
#  CLUSTERING ANALYSIS — HEARTBEAT AUDIO
#  Unsupervised discovery of structure in cardiac sounds
#
#  Algorithms : K-Means | DBSCAN | Agglomerative | GMM
#  Reduction  : PCA | t-SNE | UMAP
#  Features   : Handcrafted | ComParE | Wav2Vec 2.0
#  Dataset    : https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds
#
#  Paste into ONE Jupyter cell and run.
# ============================================================

# ── STEP 1: Install dependencies ────────────────────────────
import sys, subprocess
pkgs = [
    "librosa", "scikit-learn", "pandas", "matplotlib",
    "seaborn", "numpy", "soundfile", "torch",
    "transformers==4.35.2", "opensmile",
    "umap-learn", "scipy",
]
subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"])
print("✅ Packages installed")

# ── STEP 2: Imports ─────────────────────────────────────────
import os, sys
import numpy as np
import pandas as pd
import librosa
import torch
import opensmile
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Evaluation
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

sys.modules.setdefault("torchvision", None)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import warnings
warnings.filterwarnings("ignore")
print("✅ Imports done")

# ── STEP 3: Config ───────────────────────────────────────────
DATASET_DIR  = Path(r"D:\Heart Beat Sound")
SR           = 16000
DURATION     = 4
N_MFCC       = 40
HOP_LENGTH   = 512
RANDOM_STATE = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Device: {DEVICE}")

# ── STEP 4: Auto-detect dataset ─────────────────────────────
def find_dataset(base):
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4: continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths: paths["set_a_csv"] = root/name
            if name == "set_b.csv" and "set_b_csv" not in paths: paths["set_b_csv"] = root/name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths: paths["set_a_dir"] = root/d
            if d == "set_b" and "set_b_dir" not in paths: paths["set_b_dir"] = root/d
    return paths

PATHS = find_dataset(DATASET_DIR)
print("✅ Dataset found")

# ── STEP 5: Load metadata ────────────────────────────────────
VALID = {"normal", "murmur", "extrastole", "artifact", "extrahls"}

def load_metadata():
    df_a = pd.read_csv(PATHS["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns = [c.lower().strip() for c in df_a.columns]
    rows = []
    for f in PATHS["set_b_dir"].iterdir():
        if f.suffix != ".wav": continue
        prefix = f.name.split("_")[0].lower()
        if prefix in VALID:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})
    df = pd.concat([df_a, pd.DataFrame(rows)], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan", "unlabeled", ""])]
    print("📊 Label distribution:")
    print(df["label"].value_counts().to_string())
    return df

df        = load_metadata()
N_CLASSES = df["label"].nunique()

# ── STEP 6: Audio loader ─────────────────────────────────────
def load_audio(fname, dataset):
    folder = PATHS["set_a_dir"] if dataset == "A" else PATHS["set_b_dir"]
    path   = folder / Path(fname).name
    if not path.exists(): return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target = SR * DURATION
        return np.pad(y, (0, max(0, target - len(y))))[:target]
    except: return None

# ── STEP 7: Feature extraction ───────────────────────────────
def extract_handcrafted(y):
    def stats(x): return np.hstack([x.mean(axis=-1), x.std(axis=-1),
                                     np.percentile(x, 25, axis=-1),
                                     np.percentile(x, 75, axis=-1)])
    mfcc     = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_d   = librosa.feature.delta(mfcc)
    mfcc_d2  = librosa.feature.delta(mfcc, order=2)
    chroma   = librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP_LENGTH)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SR, hop_length=HOP_LENGTH)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=SR)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    mel      = librosa.power_to_db(
                   librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP_LENGTH),
                   ref=np.max)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=SR, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=HOP_LENGTH)
    bw       = librosa.feature.spectral_bandwidth(y=y, sr=SR, hop_length=HOP_LENGTH)
    return np.hstack([
        stats(mfcc), stats(mfcc_d), stats(mfcc_d2),
        stats(chroma), stats(contrast), stats(tonnetz),
        stats(zcr), stats(rms),
        [mel.mean(), mel.std(), np.percentile(mel,25), np.percentile(mel,75)],
        stats(rolloff), stats(centroid), stats(bw),
    ]).astype(np.float32)

print("⏳ Initialising ComParE...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)
print(f"✅ ComParE: {len(smile.feature_names)} features")

def extract_compare(y):
    return np.nan_to_num(
        smile.process_signal(y, SR).values.flatten().astype(np.float32))

print("⏳ Loading Wav2Vec 2.0...")
w2v_proc  = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
w2v_model.eval()
print("✅ Wav2Vec ready")

def extract_wav2vec(y):
    y = y / (np.abs(y).max() + 1e-8)
    inp = w2v_proc(y, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = w2v_model(inp.input_values.to(DEVICE))
    h = out.last_hidden_state.squeeze(0)
    return torch.cat([h.mean(0), h.std(0)]).cpu().numpy().astype(np.float32)

# ── STEP 8: Extract all features ─────────────────────────────
print("\n⏳ Extracting features (~15-20 min)...")
X_hc, X_cmp, X_w2v, y_labels = [], [], [], []
skipped = 0

for i, (_, row) in enumerate(df.iterrows(), 1):
    if i % 10 == 0 or i == len(df):
        pct = i / len(df) * 100
        bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
        print(f"  [{bar}] {pct:.0f}%  ({i}/{len(df)})", end="\r")
    audio = load_audio(row["fname"], row["dataset"])
    if audio is None: skipped += 1; continue
    try:
        X_hc.append(extract_handcrafted(audio))
        X_cmp.append(extract_compare(audio))
        X_w2v.append(extract_wav2vec(audio))
        y_labels.append(row["label"])
    except: skipped += 1

X_hc  = np.array(X_hc,  dtype=np.float32)
X_cmp = np.array(X_cmp, dtype=np.float32)
X_w2v = np.array(X_w2v, dtype=np.float32)
for X in [X_hc, X_cmp, X_w2v]:
    np.nan_to_num(X, copy=False)
X_combined = np.hstack([X_hc, X_cmp, X_w2v])

y_labels = np.array(y_labels)
le       = LabelEncoder()
y_true   = le.fit_transform(y_labels)

print(f"\n✅ {len(y_labels)} samples | skipped {skipped}")
print(f"Classes: {le.classes_}")
print(f"Dims — HC:{X_hc.shape[1]} | ComParE:{X_cmp.shape[1]} | W2V:{X_w2v.shape[1]}")

# ── STEP 9: Scale + convert to float64 ───────────────────────
# float64 is required for GMM (Cholesky decomposition stability)
scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X_cmp).astype(np.float64)
X_hc_sc   = StandardScaler().fit_transform(X_hc).astype(np.float64)
X_w2v_sc  = StandardScaler().fit_transform(X_w2v).astype(np.float64)
X_comb_sc = StandardScaler().fit_transform(X_combined).astype(np.float64)

# ── STEP 10: Dimensionality reduction ────────────────────────
print("\n⏳ Running PCA, t-SNE, UMAP...")

pca_50  = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca50 = pca_50.fit_transform(X_scaled)   # already float64

pca_2   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2  = pca_2.fit_transform(X_scaled)

cumvar = np.cumsum(pca_50.explained_variance_ratio_)
print(f"  PCA: {cumvar[-1]*100:.1f}% variance in 50 components")
print(f"  PCA 2D: {pca_2.explained_variance_ratio_.sum()*100:.1f}% variance")

# t-SNE (max_iter replaces deprecated n_iter in newer sklearn)
tsne   = TSNE(n_components=2, perplexity=30, max_iter=1000,
              random_state=RANDOM_STATE, verbose=0)
X_tsne = tsne.fit_transform(X_pca50)
print("  ✅ t-SNE done")

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=RANDOM_STATE)
X_umap  = reducer.fit_transform(X_pca50)
print("  ✅ UMAP done")

# ── STEP 11: Elbow + Silhouette to find optimal K ────────────
print("\n⏳ Finding optimal number of clusters...")
K_range     = range(2, 11)
inertias    = []
silhouettes = []
db_scores   = []
ch_scores   = []

for k in K_range:
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca50)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca50, labels))
    db_scores.append(davies_bouldin_score(X_pca50, labels))
    ch_scores.append(calinski_harabasz_score(X_pca50, labels))

best_k_sil = list(K_range)[np.argmax(silhouettes)]
best_k_db  = list(K_range)[np.argmin(db_scores)]
print(f"  Best k (Silhouette)     : {best_k_sil}")
print(f"  Best k (Davies-Bouldin) : {best_k_db}")
print(f"  Using k = {N_CLASSES}   (matches number of true classes)")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].plot(K_range, inertias, "o-", color="#4C72B0", lw=2)
axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia"); axes[0].grid(alpha=0.3)

axes[1].plot(K_range, silhouettes, "o-", color="#55A868", lw=2)
axes[1].axvline(best_k_sil, color="red", linestyle="--", alpha=0.7,
                label=f"Best k={best_k_sil}")
axes[1].set_title("Silhouette (higher=better)")
axes[1].set_xlabel("k"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(K_range, db_scores, "o-", color="#DD8452", lw=2)
axes[2].axvline(best_k_db, color="red", linestyle="--", alpha=0.7,
                label=f"Best k={best_k_db}")
axes[2].set_title("Davies-Bouldin (lower=better)")
axes[2].set_xlabel("k"); axes[2].legend(); axes[2].grid(alpha=0.3)

axes[3].plot(K_range, ch_scores, "o-", color="#C44E52", lw=2)
axes[3].set_title("Calinski-Harabasz (higher=better)")
axes[3].set_xlabel("k"); axes[3].grid(alpha=0.3)

fig.suptitle("Optimal Number of Clusters — ComParE Features (PCA 50D)", fontsize=13)
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_optimal_k.png", dpi=150)
plt.show()
print("Saved → clustering_optimal_k.png")

# ── STEP 12: Run all clustering algorithms ───────────────────
print(f"\n⏳ Running clustering algorithms with k={N_CLASSES}...")
cluster_labels = {}

CLUSTERERS = {
    "K-Means": KMeans(
        n_clusters=N_CLASSES, random_state=RANDOM_STATE, n_init=20),
    "Agglomerative\n(Ward)": AgglomerativeClustering(
        n_clusters=N_CLASSES, linkage="ward"),
    "Agglomerative\n(Complete)": AgglomerativeClustering(
        n_clusters=N_CLASSES, linkage="complete"),
    "GMM": GaussianMixture(
        n_components=N_CLASSES,
        covariance_type="diag",  # diagonal — stable on high-dim PCA data
        reg_covar=1e-3,          # regularise to prevent singular covariance
        n_init=5,
        random_state=RANDOM_STATE),
}

for name, clf in CLUSTERERS.items():
    if hasattr(clf, "fit_predict"):
        labels = clf.fit_predict(X_pca50)
    else:
        clf.fit(X_pca50)
        labels = clf.predict(X_pca50)
    cluster_labels[name] = labels
    sil = silhouette_score(X_pca50, labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"  {name.replace(chr(10),' '):<28} "
          f"Silhouette:{sil:.3f}  ARI:{ari:.3f}  NMI:{nmi:.3f}")

# DBSCAN — auto-determines number of clusters
dbscan    = DBSCAN(eps=3.0, min_samples=5)
db_labels = dbscan.fit_predict(X_pca50)
n_db      = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise   = (db_labels == -1).sum()
cluster_labels["DBSCAN"] = db_labels
print(f"  {'DBSCAN':<28} Clusters found: {n_db}  Noise: {n_noise}")

# ── STEP 13: Metrics summary table ───────────────────────────
print("\n📊 Clustering Evaluation Summary:")
print(f"{'Algorithm':<30} {'Silhouette':>11} {'ARI':>8} "
      f"{'NMI':>8} {'Homog.':>8} {'Complet.':>9}")
print("─" * 76)

metrics_records = []
for name, labels in cluster_labels.items():
    valid = labels != -1
    if valid.sum() < 2: continue
    sil  = silhouette_score(X_pca50[valid], labels[valid])
    ari  = adjusted_rand_score(y_true[valid], labels[valid])
    nmi  = normalized_mutual_info_score(y_true[valid], labels[valid])
    hom  = homogeneity_score(y_true[valid], labels[valid])
    comp = completeness_score(y_true[valid], labels[valid])
    vmes = v_measure_score(y_true[valid], labels[valid])
    print(f"  {name.replace(chr(10),' '):<28} {sil:>10.3f} {ari:>8.3f} "
          f"{nmi:>8.3f} {hom:>8.3f} {comp:>9.3f}")
    metrics_records.append({
        "Algorithm"  : name.replace("\n", " "),
        "Silhouette" : sil, "ARI": ari, "NMI": nmi,
        "Homogeneity": hom, "Completeness": comp, "V-Measure": vmes,
    })

metrics_df = pd.DataFrame(metrics_records)

# ── STEP 14: 2D scatter — PCA | t-SNE | UMAP ────────────────
print("\n⏳ Plotting 2D projections...")

LABEL_COLORS = {
    "normal"    : "#4C72B0",
    "murmur"    : "#DD8452",
    "extrastole": "#55A868",
    "artifact"  : "#C44E52",
    "extrahls"  : "#8172B2",
}
PALETTE_CL  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
                "#937860","#DA8BC3","#8C8C8C"]
km_labels   = cluster_labels["K-Means"]
projections = [("PCA", X_pca2), ("t-SNE", X_tsne), ("UMAP", X_umap)]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for col, (proj_name, X_2d) in enumerate(projections):
    # Row 0 — true labels
    for label in le.classes_:
        mask = y_labels == label
        axes[0, col].scatter(X_2d[mask,0], X_2d[mask,1],
                             c=LABEL_COLORS[label], label=label,
                             s=25, alpha=0.75, edgecolors="none")
    axes[0, col].set_title(f"{proj_name} — True Labels",
                           fontsize=11, fontweight="bold")
    axes[0, col].legend(fontsize=8, markerscale=1.5)
    axes[0, col].set_xlabel(f"{proj_name} 1")
    axes[0, col].set_ylabel(f"{proj_name} 2")

    # Row 1 — K-Means clusters
    for k in range(N_CLASSES):
        mask = km_labels == k
        axes[1, col].scatter(X_2d[mask,0], X_2d[mask,1],
                             c=PALETTE_CL[k], label=f"Cluster {k}",
                             s=25, alpha=0.75, edgecolors="none")
    axes[1, col].set_title(f"{proj_name} — K-Means (k={N_CLASSES})",
                           fontsize=11, fontweight="bold")
    axes[1, col].legend(fontsize=8, markerscale=1.5)
    axes[1, col].set_xlabel(f"{proj_name} 1")
    axes[1, col].set_ylabel(f"{proj_name} 2")

fig.suptitle("Heartbeat Audio — 2D Projections\n"
             "Row 1: True Labels  |  Row 2: K-Means Clusters",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_2d_projections.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved → clustering_2d_projections.png")

# ── STEP 15: All algorithms on t-SNE ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
algo_items = list(cluster_labels.items())

for idx, (name, labels) in enumerate(algo_items):
    ax = axes[idx//3, idx%3]
    for lbl in sorted(set(labels)):
        mask  = labels == lbl
        color = "lightgray" if lbl == -1 else PALETTE_CL[lbl % len(PALETTE_CL)]
        lab   = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_tsne[mask,0], X_tsne[mask,1], c=color,
                   label=lab, s=20, alpha=0.75, edgecolors="none")
    n_cl = len([l for l in set(labels) if l != -1])
    ax.set_title(f"{name.replace(chr(10),' ')}  ({n_cl} clusters)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

# Last panel — true labels for reference
ax = axes[1, 2]
for label in le.classes_:
    mask = y_labels == label
    ax.scatter(X_tsne[mask,0], X_tsne[mask,1],
               c=LABEL_COLORS[label], label=label,
               s=20, alpha=0.75, edgecolors="none")
ax.set_title("TRUE LABELS (reference)", fontsize=11,
             fontweight="bold", color="darkred")
ax.legend(fontsize=8, markerscale=1.5)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

fig.suptitle("All Clustering Algorithms — t-SNE Projection (ComParE Features)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_all_algorithms.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved → clustering_all_algorithms.png")

# ── STEP 16: Metrics heatmap ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
metric_cols = ["Silhouette","ARI","NMI","Homogeneity","Completeness","V-Measure"]
heat_data   = metrics_df.set_index("Algorithm")[metric_cols]
sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlGnBu",
            vmin=0, vmax=1, ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Score"})
ax.set_title("Clustering Evaluation Metrics\n"
             "(ARI/NMI/Homogeneity/Completeness require true labels)",
             fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=20)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_metrics_heatmap.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved → clustering_metrics_heatmap.png")

# ── STEP 17: Dendrogram ──────────────────────────────────────
print("\n⏳ Building dendrogram...")
np.random.seed(RANDOM_STATE)
sample_idx = np.random.choice(len(X_pca50), size=min(120, len(X_pca50)),
                               replace=False)
X_dendro   = X_pca50[sample_idx]
y_dendro   = y_labels[sample_idx]

Z   = linkage(X_dendro, method="ward")
fig, ax = plt.subplots(figsize=(18, 6))
dendrogram(Z, ax=ax, labels=y_dendro, leaf_font_size=7,
           color_threshold=0.7 * max(Z[:,2]),
           above_threshold_color="gray")
ax.set_title("Hierarchical Clustering Dendrogram — Ward Linkage\n"
             "(120 random samples, ComParE features, PCA 50D)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Distance")
legend_handles = [Patch(facecolor=LABEL_COLORS[l], label=l)
                  for l in le.classes_]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_dendrogram.png", dpi=150)
plt.show()
print("Saved → clustering_dendrogram.png")

# ── STEP 18: K-Means cluster → class mapping ─────────────────
print("\n📊 K-Means Cluster → True Label Mapping:")
km_labels_final = cluster_labels["K-Means"]
mapping = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
for true_lbl, clust_lbl in zip(y_true, km_labels_final):
    mapping[clust_lbl, true_lbl] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(mapping, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=[f"Cluster {i}" for i in range(N_CLASSES)],
            ax=axes[0])
axes[0].set_title("K-Means Cluster vs True Label (counts)", fontsize=11)
axes[0].set_xlabel("True Label"); axes[0].set_ylabel("K-Means Cluster")

mapping_norm = mapping.astype(float) / (mapping.sum(axis=1, keepdims=True) + 1e-8) * 100
sns.heatmap(mapping_norm, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=[f"Cluster {i}" for i in range(N_CLASSES)],
            ax=axes[1])
axes[1].set_title("K-Means Cluster vs True Label (%)", fontsize=11)
axes[1].set_xlabel("True Label"); axes[1].set_ylabel("K-Means Cluster")

plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_kmeans_mapping.png", dpi=150)
plt.show()

for c in range(N_CLASSES):
    dominant = le.classes_[np.argmax(mapping[c])]
    pct      = mapping[c].max() / (mapping[c].sum() + 1e-8) * 100
    print(f"  Cluster {c} → mostly '{dominant}' ({pct:.0f}%)")

# ── STEP 19: Feature set clustering comparison ───────────────
print("\n⏳ Comparing clustering across feature sets...")
feat_sets = {
    "Handcrafted" : X_hc_sc,
    "ComParE"     : X_scaled,
    "Wav2Vec 2.0" : X_w2v_sc,
    "Combined"    : X_comb_sc,
}
feat_sil, feat_ari, feat_nmi = {}, {}, {}

for fname, X_feat in feat_sets.items():
    X_p   = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X_feat)
    km    = KMeans(n_clusters=N_CLASSES, random_state=RANDOM_STATE, n_init=10)
    lbl   = km.fit_predict(X_p)
    feat_sil[fname] = silhouette_score(X_p, lbl)
    feat_ari[fname] = adjusted_rand_score(y_true, lbl)
    feat_nmi[fname] = normalized_mutual_info_score(y_true, lbl)
    print(f"  {fname:<14} Sil:{feat_sil[fname]:.3f}  "
          f"ARI:{feat_ari[fname]:.3f}  NMI:{feat_nmi[fname]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metric_data = {"Silhouette": feat_sil, "ARI": feat_ari, "NMI": feat_nmi}
colors      = ["#4C72B0", "#55A868", "#DD8452"]
for ax, (metric, data), color in zip(axes, metric_data.items(), colors):
    bars = ax.bar(data.keys(), data.values(), color=color, alpha=0.85)
    ax.set_title(f"K-Means {metric} by Feature Set", fontsize=11)
    ax.set_ylim(0, max(data.values()) * 1.25)
    ax.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars, data.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_feature_comparison.png", dpi=150)
plt.show()
print("Saved → clustering_feature_comparison.png")

# ── STEP 20: PCA explained variance ──────────────────────────
pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_) * 100
n95      = np.argmax(cumvar >= 95) + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, 21),
            pca_full.explained_variance_ratio_[:20] * 100,
            color="#4C72B0", alpha=0.8)
axes[0].set_title("PCA — Individual Explained Variance (Top 20 Components)")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance Explained (%)")

axes[1].plot(range(1, len(cumvar)+1), cumvar, color="#4C72B0", lw=2)
axes[1].axhline(95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
axes[1].axvline(n95, color="orange", linestyle="--", alpha=0.7,
                label=f"{n95} components")
axes[1].set_title("PCA — Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend(); axes[1].set_xlim(0, 300)

fig.suptitle("PCA Analysis — ComParE Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR / "clustering_pca_variance.png", dpi=150)
plt.show()
print(f"Saved → clustering_pca_variance.png")
print(f"  95% variance in {n95} components (out of {X_scaled.shape[1]})")

# ── STEP 21: Final summary ───────────────────────────────────
print(f"\n{'='*60}")
print(f"  CLUSTERING ANALYSIS — COMPLETE")
print(f"{'='*60}")
print(f"\n  Dataset   : {len(y_labels)} samples | {N_CLASSES} true classes")
print(f"  Features  : ComParE (primary) | HC | Wav2Vec | Combined")
print(f"  Reduction : PCA 50D → t-SNE 2D / UMAP 2D")
print(f"\n  Algorithm ranking by ARI (↑ = better alignment with true labels):")
for _, row in metrics_df.sort_values("ARI", ascending=False).iterrows():
    print(f"    {row['Algorithm']:<28} "
          f"ARI:{row['ARI']:.3f}  NMI:{row['NMI']:.3f}  "
          f"Sil:{row['Silhouette']:.3f}")
best_feat = max(feat_ari, key=feat_ari.get)
print(f"\n  Best feature set for clustering (K-Means ARI):")
print(f"    {best_feat} → ARI:{feat_ari[best_feat]:.3f}")
print(f"\n  Output files saved to: {DATASET_DIR}")
print(f"\n  Charts generated:")
for f in ["clustering_optimal_k.png",
          "clustering_2d_projections.png",
          "clustering_all_algorithms.png",
          "clustering_metrics_heatmap.png",
          "clustering_dendrogram.png",
          "clustering_kmeans_mapping.png",
          "clustering_feature_comparison.png",
          "clustering_pca_variance.png"]:
    print(f"    → {f}")
print(f"{'='*60}")
