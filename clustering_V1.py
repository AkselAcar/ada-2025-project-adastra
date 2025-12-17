import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats


def preliminary_control(df: pd.DataFrame) -> pd.DataFrame: 
    """
    This function ensures that the dataframe has the necessary toxicity score column.

    Parameters:
    df (pd.DataFrame): Input dataframe with interactions data.

    Returns:
    pd.DataFrame: Dataframe with ensured toxicity score column.
    """
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

    # We check if TOXICITY_SCORE column exists, if not we create it
    if "TOXICITY_SCORE" not in df.columns:
        df = compute_toxicity_score(df)

    # We check if TOXICITY_CAT column exists, if not we create it
    if "TOXICITY_CAT" not in df.columns:
        # We check if a global threshold variable exists, if not we use the 75th percentile of the TOXICITY_SCORE
        if "TOXICITY_THRESHOLD" in globals():
            toxicity_threshold = TOXICITY_THRESHOLD
        else:
            toxicity_threshold = df["TOXICITY_SCORE"].quantile(0.75)
        df["TOXICITY_CAT"] = (df["TOXICITY_SCORE"] >= toxicity_threshold).astype(int)

    return df, umap_available, hdbscan_available


preliminary_control(interactions_df)

# ====================
# FEATURE ENGINEERING
# ====================

def median_gap_days(ts: pd.Series) -> float:
    """
    This function calculates the median gap in days between consecutive timestamps in a pandas Series.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.

    Returns:
    float: The median gap in days between consecutive timestamps. Returns np.nan if not computable.
    """
    # We ensure datetime type, drop missing values and sort in ascending order.
    ts_sorted = pd.to_datetime(ts, errors='coerce').dropna().sort_values()
    # If we have less than 2 timestamps, we cannot compute gaps, so we return np.nan
    if len(ts_sorted) < 2:
        return np.nan
    # Remove duplicate timestamps (if multiple identical events, we should not generate a gap of 0)
    ts_sorted = ts_sorted.drop_duplicates()
    if len(ts_sorted) < 2:
        return np.nan
    # We compute the difference between each timestamp and the previous one (in seconds) and then convert it to days
    deltas = ts_sorted.diff().dt.total_seconds() / 86400.0
    return float(deltas.dropna().median())


def gap_std(ts: pd.Series) -> float:
    """
    This function calculates the standard deviation of gaps in days between consecutive timestamps in a pandas Series.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.

    Returns:
    float: The standard deviation of gaps in days between consecutive timestamps. Returns np.nan if not
    """
    # We ensure datetime type, drop missing values, sort in ascending order and remove duplicates.
    ts_sorted = pd.to_datetime(ts, errors='coerce').dropna().sort_values().drop_duplicates()
    # If we have less than 2 timestamps, we cannot compute gaps, so we return np.nan
    if len(ts_sorted) < 2:
        return np.nan
    # We compute the difference between each timestamp and the previous one (in seconds) and then convert it to days
    deltas_days = ts_sorted.diff().dt.total_seconds() / 86400
    return float(deltas_days.dropna().std(ddof=0))  # With ddof=0, the std of a single value is 0.0 (which is reasonable: no variation if only one gap)
    

def toxicity_trend(ts: pd.Series) -> float:
    """
    This function returns the slope of toxic event count over time (events/day) using regression.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.

    Returns:
    float: The slope of toxic event count over time. Returns np.nan if not computable
    """
    # We ensure datetime type, drop missing values, sort in ascending order and remove duplicates.
    ts_sorted = pd.to_datetime(ts, errors='coerce').dropna().sort_values().drop_duplicates()
    # If we have less than 3 timestamps, we cannot compute a trend, so we return np.nan
    if len(ts_sorted) < 3:
        return np.nan

    # We convert to days since first event happened
    days = (ts_sorted - ts_sorted.min()).dt.total_seconds() / 86400.0
    # Cumulative count
    counts = np.arange(1, len(ts_sorted) + 1)  
    
    # We check if all days are the same (should not happen due to previous checks, but just in case)
    # If all days are the same, we cannot compute a slope 
    if np.allclose(days.values, days.values[0]):
        return np.nan

    try:
        # We use Theil-Sen estimator for robustness to compute the slope as Theil-Sen is 
        # more robust to outliers and non-normal data than simple linear regression
        slope, _, _, _ = stats.theilslopes(counts, days.values, 0.95)
        return float(slope)
    except Exception:
        # We fallback to linear regression if Theil-Sen fails
        try:
            slope, _, _, _, _ = stats.linregress(days.values, counts)
            return float(slope)
        except Exception:
            return np.nan


def days_since_last(ts: pd.Series, reference_date: pd.Timestamp) -> float:
    """
    This function calculates the number of days since the last timestamp in the series relative to a reference date.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.
    reference_date (pd.Timestamp): The reference date to calculate recency from.

    Returns:
    float: The number of days since the last timestamp. Returns np.nan if not computable
    """
    # We ensure datetime type and drop missing values.
    ts_filtered = pd.to_datetime(ts, errors="coerce").dropna()
    # If the series is empty or reference_date is NaT, return np.nan
    if ts_filtered.empty or pd.isna(reference_date):
        return np.nan
    last = ts_filtered.max()
    delta = reference_date - last
    return float(delta.total_seconds() / 86400.0) # Convert time delta to days


# def max_events_in_window(ts: pd.Series, window_days: int = 30) -> int:
#     ts_sorted = ts.sort_values().reset_index(drop=True)
#     if ts_sorted.empty:
#         return 0
#     window = pd.Timedelta(days=window_days)
#     left = 0
#     max_count = 1
#     for right in range(len(ts_sorted)):
#         while ts_sorted[right] - ts_sorted[left] > window:
#             left += 1
#         max_count = max(max_count, right - left + 1)
#     return max_count


def max_events_in_window(ts: pd.Series, window_days: int = 30) -> int:
    """
    This function returns the maximum number of events (equivalent to timestamps) contained in any sliding window of window_days.
    
    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.
    window_days (int): The size of the sliding window in days.

    Returns:
    int: The maximum number of events in any sliding window of the specified size.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    # We ensure datetime type, drop missing values and sort in ascending order.
    ts_sorted = pd.to_datetime(ts, errors="coerce", utc=True).dropna().sort_values().reset_index(drop=True)
    if ts_sorted.empty:
        return 0

    # We use a two-pointer technique to find the maximum number of events in any window
    window = pd.Timedelta(days=int(window_days))
    left = 0
    max_count = 0

    # Two-pointer sliding window approach
    for right in range(len(ts_sorted)):
        # Move the right pointer forward one event at a time (expanding the window)
        # For each new right, check if the window [left, right] is within the allowed window size
        # If the time difference between the rightmost and leftmost timestamps exceeds the window,
        # move the left pointer forward to shrink the window until it fits
        while left <= right and (ts_sorted.iloc[right] - ts_sorted.iloc[left]) > window:
            # The window is too large, so increment left to shrink it
            left += 1
        # After adjusting, the window [left, right] now fits within window_days
        curr = right - left + 1  # Number of events in the current window
        # Update max_count if this window contains more events than any previous window
        if curr > max_count:
            max_count = curr

    return int(max_count)


# def recent_acceleration(ts: pd.Series, recent_days: int = 90) -> float:
#     """Ratio of recent activity rate to historical rate"""
#     ts_sorted = ts.sort_values()
#     if ts_sorted.shape[0] < 4:
#         return 1.0
    
#     cutoff = ts_sorted.max() - pd.Timedelta(days=recent_days)
#     recent_count = (ts_sorted >= cutoff).sum()
#     historical_count = (ts_sorted < cutoff).sum()
    
#     # Avoid division by zero - return high value instead of inf
#     if historical_count == 0:
#         return 10.0 if recent_count > 0 else 1.0
    
#     # Rate = count / days
#     total_days = (ts_sorted.max() - ts_sorted.min()).total_seconds() / 86400
#     if total_days == 0 or total_days <= recent_days:
#         return 1.0
    
#     recent_rate = recent_count / recent_days
#     historical_rate = historical_count / (total_days - recent_days)
    
#     # Cap the ratio to avoid extreme values
#     if historical_rate > 0:
#         ratio = recent_rate / historical_rate
#         return min(ratio, 10.0)  # Cap at 10x acceleration
#     else:
#         return 10.0


def recent_acceleration(ts: pd.Series, recent_days: int = 90, cap: float = 10.0) -> float:
    """
    This function calculates the ratio of recent event rate to historical event rate.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.
    recent_days (int): The number of days to consider as the recent period.

    Returns:
    float: The ratio of recent event rate to historical event rate, capped at 10.0. Returns np.nan if not computable.
    """
    if recent_days <= 0:
        raise ValueError("recent_days must be positive")

    # We ensure datetime type, drop missing values and sort in ascending order.
    ts_sorted = pd.to_datetime(ts, errors="coerce", utc=True).dropna().sort_values()
    if len(ts_sorted) < 4:
        return np.nan  # We don't have enough data to compute a stable ratio

    # We define cutoff as being the start of the recent window
    cutoff = ts_sorted.max() - pd.Timedelta(days=int(recent_days))

    # The recent window starts either at cutoff or at the dataset's min if the dataset is shorter
    # recent_start = max(cutoff, ts_sorted.min())
    # recent_span_days = (ts_sorted.max() - recent_start).total_seconds() / 86400.0
    # recent_span_days = max(recent_span_days, 1.0)  # To avoid zero-division, we treat same-day span as 1 day

    recent_count = int((ts_sorted >= cutoff).sum())
    historical_count = int((ts_sorted < cutoff).sum())

    # If there's no historical data, we cannot compute a ratio, so we return np.nan
    if historical_count == 0:
        return np.nan 

    # We compute the number of days in historical period (i.e., from first event to cutoff)
    historical_days = (cutoff - ts_sorted.min()).total_seconds() / 86400.0
    if historical_days <= 0:
        return np.nan # Degenerate case

    # We compute rates as recent events per recent days and historical events per historical days
    recent_rate = recent_count / float(recent_days) # float(recent_span_days)
    historical_rate = historical_count / float(historical_days)

    if historical_rate <= 0:
        return np.nan # Degenerate case

    ratio = recent_rate / historical_rate
    return float(min(ratio, float(cap)))


# We build role-agnostic toxic event list to extract temporal features
toxic_df = interactions_df[interactions_df["TOXICITY_CAT"] == 1].copy()

# We rename and stack SOURCE_SUBREDDIT and TARGET_SUBREDDIT into a single column
long_toxic = pd.concat([
    toxic_df[["TIMESTAMP", "SOURCE_SUBREDDIT"]].rename(columns={"SOURCE_SUBREDDIT": "SUBREDDIT"}),
    toxic_df[["TIMESTAMP", "TARGET_SUBREDDIT"]].rename(columns={"TARGET_SUBREDDIT": "SUBREDDIT"}),
])
long_toxic.dropna(subset=["SUBREDDIT"], inplace=True)

grp = long_toxic.groupby("SUBREDDIT")

# We extract the most recent day as reference for recency calculations
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
temporal_features = temporal_features[temporal_features["tox_count_total"] >= 10].copy()
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


def generate_receipt(dictionnaire, liste):
    receipt = {}

    for produit in liste:
        if produit not in receipt:
            receipt[produit] = 0
        if produit in dictionnaire:
            receipt[produit] += dictionnaire[produit]

    return receipt



