# IMPROVED Temporal toxicity clustering: role-agnostic, per-subreddit dynamics
# Implements critical recommendations from analysis
# 
# TO USE: Copy this code into the notebook cell after "### TEST" markdown

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats

try:
    import umap
    umap_available = True
except Exception:
    umap_available = False

try:
    import hdbscan
    hdbscan_available = True
except Exception:
    hdbscan_available = False

# Ensure toxicity score exists
if "TOXICITY_SCORE" not in interactions_df.columns:
    interactions_df = compute_toxicity_score(interactions_df)

# Ensure binary toxicity label exists
if "TOXICITY_CAT" not in interactions_df.columns:
    if "TOXICITY_THRESHOLD" in globals():
        toxicity_threshold = TOXICITY_THRESHOLD
    else:
        toxicity_threshold = interactions_df["TOXICITY_SCORE"].quantile(0.75)
    interactions_df["TOXICITY_CAT"] = (interactions_df["TOXICITY_SCORE"] >= toxicity_threshold).astype(int)


# ====================
# IMPROVED FEATURE ENGINEERING
# ====================

# Helper: median gap in days between consecutive timestamps
def median_gap_days(ts: pd.Series) -> float:
    ts_sorted = ts.sort_values()
    if ts_sorted.shape[0] < 2:
        return np.nan
    deltas = ts_sorted.diff().dropna().dt.total_seconds() / 86400
    return deltas.median()


# Helper: std of gaps (VOLATILITY measure - NEW)
def gap_std(ts: pd.Series) -> float:
    ts_sorted = ts.sort_values()
    if ts_sorted.shape[0] < 2:
        return np.nan
    deltas = ts_sorted.diff().dropna().dt.total_seconds() / 86400
    return deltas.std()


# Helper: toxicity trend (linear regression slope over time - NEW)
def toxicity_trend(ts: pd.Series) -> float:
    """Returns slope of toxic event count over time (events/day)"""
    ts_sorted = ts.sort_values()
    if ts_sorted.shape[0] < 3:
        return 0.0
    
    # Convert to days since first event
    days = (ts_sorted - ts_sorted.min()).dt.total_seconds() / 86400
    counts = np.arange(1, len(ts_sorted) + 1)  # cumulative count
    
    try:
        slope, _, _, _, _ = stats.linregress(days.values, counts)
        return slope
    except:
        return 0.0


# Helper: recency score (days since last toxic event - NEW)
def days_since_last(ts: pd.Series, reference_date: pd.Timestamp) -> float:
    """Days from last toxic event to reference date"""
    if ts.empty:
        return np.nan
    return (reference_date - ts.max()).total_seconds() / 86400


# Helper: maximum number of toxic events in any rolling window
def max_events_in_window(ts: pd.Series, window_days: int = 30) -> int:
    ts_sorted = ts.sort_values().reset_index(drop=True)
    if ts_sorted.empty:
        return 0
    window = pd.Timedelta(days=window_days)
    left = 0
    max_count = 1
    for right in range(len(ts_sorted)):
        while ts_sorted[right] - ts_sorted[left] > window:
            left += 1
        max_count = max(max_count, right - left + 1)
    return max_count


# Helper: acceleration (recent vs previous activity - NEW)
def recent_acceleration(ts: pd.Series, recent_days: int = 90) -> float:
    """Ratio of recent activity rate to historical rate"""
    ts_sorted = ts.sort_values()
    if ts_sorted.shape[0] < 4:
        return 1.0
    
    cutoff = ts_sorted.max() - pd.Timedelta(days=recent_days)
    recent_count = (ts_sorted >= cutoff).sum()
    historical_count = (ts_sorted < cutoff).sum()
    
    # Avoid division by zero - return high value instead of inf
    if historical_count == 0:
        return 10.0 if recent_count > 0 else 1.0
    
    # Rate = count / days
    total_days = (ts_sorted.max() - ts_sorted.min()).total_seconds() / 86400
    if total_days == 0 or total_days <= recent_days:
        return 1.0
    
    recent_rate = recent_count / recent_days
    historical_rate = historical_count / (total_days - recent_days)
    
    # Cap the ratio to avoid extreme values
    if historical_rate > 0:
        ratio = recent_rate / historical_rate
        return min(ratio, 10.0)  # Cap at 10x acceleration
    else:
        return 10.0


# Build role-agnostic toxic event list per subreddit
toxic_df = interactions_df[interactions_df["TOXICITY_CAT"] == 1].copy()

long_toxic = pd.concat([
    toxic_df[["TIMESTAMP", "SOURCE_SUBREDDIT"]].rename(columns={"SOURCE_SUBREDDIT": "SUBREDDIT"}),
    toxic_df[["TIMESTAMP", "TARGET_SUBREDDIT"]].rename(columns={"TARGET_SUBREDDIT": "SUBREDDIT"}),
])
long_toxic.dropna(subset=["SUBREDDIT"], inplace=True)

grp = long_toxic.groupby("SUBREDDIT")

# Reference date for recency calculation
reference_date = long_toxic["TIMESTAMP"].max()

temporal_features = pd.DataFrame({
    "tox_count_total": grp["TIMESTAMP"].size(),
    "first_tox_ts": grp["TIMESTAMP"].min(),
    "last_tox_ts": grp["TIMESTAMP"].max(),
})

temporal_features["active_span_days"] = (
    (temporal_features["last_tox_ts"] - temporal_features["first_tox_ts"])
    .dt.total_seconds() / 86400
)

# Original features
temporal_features["median_gap_days"] = grp["TIMESTAMP"].apply(median_gap_days)
temporal_features["max_30d_count"] = grp["TIMESTAMP"].apply(
    lambda s: max_events_in_window(s, window_days=30)
)

# NEW FEATURES
temporal_features["gap_std"] = grp["TIMESTAMP"].apply(gap_std)  # Volatility
temporal_features["toxicity_trend"] = grp["TIMESTAMP"].apply(toxicity_trend)  # Trend
temporal_features["days_since_last"] = grp["TIMESTAMP"].apply(lambda s: days_since_last(s, reference_date))  # Recency
temporal_features["recent_acceleration"] = grp["TIMESTAMP"].apply(recent_acceleration)  # Acceleration

# REMOVED: median_gap_freq_30d (redundant as noted in critique)

print(f"Total subreddits before filtering: {len(temporal_features)}")

# ====================
# CRITICAL: FILTER LOW-ACTIVITY SUBREDDITS
# ====================
# Remove subreddits with < 3 toxic events (reduces noise)
temporal_features = temporal_features[temporal_features["tox_count_total"] >= 3].copy()
print(f"Subreddits after filtering (>= 3 toxic events): {len(temporal_features)}")

# ====================
# IMPROVED NaN HANDLING
# ====================
# For gap metrics: use median instead of sentinel values
temporal_features["median_gap_days"] = temporal_features["median_gap_days"].fillna(
    temporal_features["median_gap_days"].median()
)
temporal_features["gap_std"] = temporal_features["gap_std"].fillna(
    temporal_features["gap_std"].median()
)

# Other NaN fills
temporal_features["days_since_last"] = temporal_features["days_since_last"].fillna(0)
temporal_features["recent_acceleration"] = temporal_features["recent_acceleration"].fillna(1.0)
temporal_features["toxicity_trend"] = temporal_features["toxicity_trend"].fillna(0)

# ====================
# DIAGNOSTIC: Check for extreme outliers
# ====================
print("\n" + "="*60)
print("EXTREME VALUE DIAGNOSTICS")
print("="*60)
for col in ["toxicity_trend", "recent_acceleration"]:
    p99 = temporal_features[col].quantile(0.99)
    extreme = temporal_features[temporal_features[col] > p99]
    print(f"\n{col}:")
    print(f"  99th percentile: {p99:.2f}")
    print(f"  Max value: {temporal_features[col].max():.2f}")
    print(f"  N extreme (>99th): {len(extreme)}")
    if len(extreme) <= 10:
        print(f"  Extreme subreddits: {list(extreme.index[:10])}")

# ====================
# OUTLIER HANDLING WITH LOG TRANSFORM
# ====================
# Select features for clustering (REMOVED is_limited_data - it was constant)
feature_cols = [
    "tox_count_total",
    "active_span_days",
    "median_gap_days",
    "gap_std",
    "max_30d_count",
    "toxicity_trend",
    "days_since_last",
    "recent_acceleration",
]

features = temporal_features[feature_cols].copy()

# Log-transform count-based features to handle outliers
for col in ["tox_count_total", "active_span_days", "median_gap_days", "gap_std", "max_30d_count", "days_since_last"]:
    features[col] = np.log1p(features[col])  # log1p handles 0 values

# More aggressive winsorization for extreme features
for col in ["toxicity_trend", "recent_acceleration"]:
    p95 = features[col].quantile(0.95)  # Use 95th instead of 99th
    features[col] = features[col].clip(upper=p95)
    print(f"Winsorized {col} at 95th percentile: {p95:.2f}")

# CRITICAL: Replace any remaining inf/nan values before scaling
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features.fillna(features.median(), inplace=True)

# Verify no inf/nan remain
print(f"\nData quality check before scaling:")
print(f"  NaN values: {features.isna().sum().sum()}")
print(f"  Inf values: {np.isinf(features.values).sum()}")
print(f"  Max value: {features.max().max():.2f}")

# ====================
# ROBUST SCALING
# ====================
# Use RobustScaler instead of StandardScaler (median/IQR-based, outlier-resistant)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(features)

print(f"\nFeature summary after preprocessing:")
print(features.describe())

# ====================
# DIMENSIONALITY REDUCTION (for visualization only)
# ====================
if umap_available:
    reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=20, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    reducer_name = "UMAP"
else:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    reducer_name = "PCA"

# ====================
# CLUSTER IN ORIGINAL SCALED SPACE (CRITICAL FIX)
# ====================
# Do NOT cluster on 2D embedding - cluster on full scaled features

print("\n" + "="*60)
print("CLUSTERING ALGORITHM COMPARISON")
print("="*60)

if hdbscan_available:
    print("\nTrying HDBSCAN (density-based)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,  # Increased from 8 to avoid tiny clusters
        min_samples=10,       # Increased for more conservative clustering
        metric="euclidean",
        cluster_selection_epsilon=0.5
    )
    cluster_labels_hdbscan = clusterer.fit_predict(X_scaled)
    n_clusters_hdbscan = len(set(cluster_labels_hdbscan)) - (1 if -1 in cluster_labels_hdbscan else 0)
    n_noise = list(cluster_labels_hdbscan).count(-1)
    
    print(f"  HDBSCAN found: {n_clusters_hdbscan} clusters, {n_noise} noise points")
    
    # Check cluster balance
    unique, counts = np.unique(cluster_labels_hdbscan[cluster_labels_hdbscan != -1], return_counts=True)
    if len(unique) > 0:
        max_size = counts.max()
        min_size = counts.min()
        balance_ratio = min_size / max_size if max_size > 0 else 0
        print(f"  Cluster balance: min={min_size}, max={max_size}, ratio={balance_ratio:.2f}")
    
    cluster_labels = cluster_labels_hdbscan
    cluster_method = f"HDBSCAN (k={n_clusters_hdbscan}, noise={n_noise})"
else:
    cluster_labels = None
    cluster_method = None

# Always try KMeans with multiple k values for comparison
print("\nTrying KMeans with multiple k values...")
kmeans_results = []

for k in [5, 7, 10, 12, 15]:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    # Compute metrics
    silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
    
    # Check cluster balance
    unique, counts = np.unique(labels, return_counts=True)
    max_size = counts.max()
    min_size = counts.min()
    balance_ratio = min_size / max_size
    
    kmeans_results.append({
        'k': k,
        'silhouette': silhouette,
        'balance_ratio': balance_ratio,
        'min_size': min_size,
        'max_size': max_size,
        'labels': labels
    })
    
    print(f"  k={k}: silhouette={silhouette:.3f}, balance={balance_ratio:.3f}, "
          f"sizes=[{min_size}–{max_size}]")

# Select best k based on balance AND silhouette
# Prefer balanced clusters (ratio > 0.1) over perfect silhouette
print("\nSelecting best KMeans configuration...")
valid_results = [r for r in kmeans_results if r['balance_ratio'] > 0.05]

if valid_results:
    # Among balanced options, pick highest silhouette
    best_kmeans = max(valid_results, key=lambda x: x['silhouette'])
else:
    # Fallback: just pick highest silhouette
    best_kmeans = max(kmeans_results, key=lambda x: x['silhouette'])

print(f"  Selected: k={best_kmeans['k']} (silhouette={best_kmeans['silhouette']:.3f}, "
      f"balance={best_kmeans['balance_ratio']:.3f})")

# Use KMeans if HDBSCAN not available or produced poor results
if cluster_labels is None or n_clusters_hdbscan < 3:
    cluster_labels = best_kmeans['labels']
    cluster_method = f"KMeans (k={best_kmeans['k']}, silhouette={best_kmeans['silhouette']:.3f})"
    best_k = best_kmeans['k']
    print(f"\nUsing KMeans clustering")
else:
    print(f"\nUsing HDBSCAN clustering")
    best_k = n_clusters_hdbscan

# ====================
# CLUSTER STABILITY VALIDATION (bootstrap)
# ====================
print("\n" + "="*60)
print("CLUSTER STABILITY VALIDATION (Bootstrap)")
print("="*60)

n_bootstrap = 10
ari_scores = []
for i in range(n_bootstrap):
    # Sample 80% of data
    sample_idx = np.random.choice(len(X_scaled), size=int(0.8 * len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]
    
    if hdbscan_available and 'n_clusters_hdbscan' in locals():
        clusterer_boot = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric="euclidean")
        labels_boot = clusterer_boot.fit_predict(X_sample)
    else:
        km_boot = KMeans(n_clusters=best_k, n_init=20, random_state=i)
        labels_boot = km_boot.fit_predict(X_sample)
    
    # Compute ARI between bootstrap sample and full clustering
    ari = adjusted_rand_score(cluster_labels[sample_idx], labels_boot)
    ari_scores.append(ari)

print(f"Bootstrap ARI (mean ± std): {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")
print(f"Higher ARI (close to 1.0) = more stable clusters")
print(f"Interpretation: {'STABLE' if np.mean(ari_scores) > 0.7 else 'MODERATE' if np.mean(ari_scores) > 0.5 else 'UNSTABLE'}")

# ====================
# SILHOUETTE COMPARISON (embedding vs original space)
# ====================
silhouette_original = silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else -1
silhouette_embedding = silhouette_score(embedding, cluster_labels) if len(set(cluster_labels)) > 1 else -1

print(f"\nSilhouette Score Comparison:")
print(f"  Original {len(feature_cols)}D space: {silhouette_original:.3f}")
print(f"  {reducer_name} 2D embedding: {silhouette_embedding:.3f}")
print(f"  → Clustering on original space is {'BETTER' if silhouette_original > silhouette_embedding else 'WORSE'}")

# WARNING for suspiciously high scores
if silhouette_original > 0.9:
    print(f"\n⚠️  WARNING: Silhouette > 0.9 suggests outlier-driven clustering")
    print(f"    This often means a few extreme points are being separated,")
    print(f"    not meaningful pattern discovery in the main data.")

# ====================
# RESULTS ASSEMBLY
# ====================
result_df = features.copy()
result_df["cluster"] = cluster_labels
result_df["embed_x"] = embedding[:, 0]
result_df["embed_y"] = embedding[:, 1]

# Add back original (non-transformed) features for interpretation
for col in ["tox_count_total", "active_span_days", "median_gap_days", "gap_std", 
            "max_30d_count", "days_since_last"]:
    result_df[f"{col}_raw"] = temporal_features[col]

print(f"\n" + "="*60)
print(f"Reducer: {reducer_name} | Clustering: {cluster_method}")
print("="*60)
print("\nCluster sizes:")
cluster_size_df = result_df.groupby("cluster").size().rename("n_subreddits").sort_values(ascending=False)
print(cluster_size_df)

# Check for imbalance
if len(cluster_size_df) > 1:
    max_cluster_size = cluster_size_df.max()
    min_cluster_size = cluster_size_df.min()
    imbalance_ratio = max_cluster_size / min_cluster_size
    
    print(f"\nCluster Balance Analysis:")
    print(f"  Largest cluster: {max_cluster_size} ({max_cluster_size/len(result_df)*100:.1f}%)")
    print(f"  Smallest cluster: {min_cluster_size} ({min_cluster_size/len(result_df)*100:.1f}%)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if max_cluster_size / len(result_df) > 0.9:
        print(f"\n⚠️  WARNING: One cluster contains >90% of data!")
        print(f"    This suggests poor segmentation. Consider:")
        print(f"    - Using different k values")
        print(f"    - Adding more discriminative features")
        print(f"    - Checking if data is naturally homogeneous")

# Cluster interpretation
print("\n" + "="*60)
print("CLUSTER CHARACTERISTICS (median values)")
print("="*60)
cluster_summary = result_df.groupby("cluster")[[
    "tox_count_total_raw", 
    "active_span_days_raw", 
    "gap_std_raw",
    "days_since_last_raw",
    "recent_acceleration"
]].median()
print(cluster_summary)

# ====================
# VISUALIZATION
# ====================
plt.figure(figsize=(12, 5))

# Plot 1: Embedding colored by cluster
plt.subplot(1, 2, 1)
palette = sns.color_palette("husl", n_colors=len(set(cluster_labels))) if cluster_labels is not None else None
sns.scatterplot(
    x="embed_x", y="embed_y", hue="cluster", data=result_df,
    palette=palette, s=50, alpha=0.85, edgecolor="white", linewidth=0.6
)
plt.title(f"Subreddit temporal toxicity dynamics\n({reducer_name} projection, clustered in {len(feature_cols)}D)")
plt.xlabel(f"{reducer_name} 1")
plt.ylabel(f"{reducer_name} 2")
plt.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.2)

# Plot 2: Feature importance (PCA variance)
plt.subplot(1, 2, 2)
pca_full = PCA(n_components=min(len(feature_cols), len(X_scaled)), random_state=42)
pca_full.fit(X_scaled)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
        pca_full.explained_variance_ratio_ * 100)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.title(f"PCA Variance Explained\n(Total: {pca_full.explained_variance_ratio_.sum()*100:.1f}%)")
plt.grid(alpha=0.2, axis='y')

plt.tight_layout()
plt.show()

result_df.head()
