"""
Clustering utilities for subreddit toxicity analysis.

This module provides functions for:
- Feature engineering from interaction data
- Dimensionality reduction (UMAP/PCA)
- Clustering (KMeans/HDBSCAN)
- Cluster evaluation and validation
- Cluster profiling and interpretation
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def build_subreddit_features(interactions_df, toxicity_threshold, min_activity=5):
    """
    Build behavioral features for each subreddit from interaction data.
    
    Improved feature engineering:
    - Separates no-activity (NaN) from zero-toxicity (0.0)
    - Handles division by zero in rate calculations
    - Filters low-activity subreddits to reduce noise
    
    Parameters:
    -----------
    interactions_df : pd.DataFrame
        Interaction data with TOXICITY_SCORE, SOURCE_SUBREDDIT, TARGET_SUBREDDIT
    toxicity_threshold : float
        Threshold for classifying messages as toxic
    min_activity : int
        Minimum number of interactions (sent or received) to include subreddit
        
    Returns:
    --------
    features : pd.DataFrame
        Feature matrix with subreddits as index, behavioral metrics as columns
    """
    # Source-side aggregates
    src_group = interactions_df.groupby("SOURCE_SUBREDDIT")
    src_features = pd.DataFrame({
        "tox_mean_src": src_group["TOXICITY_SCORE"].mean(),
        "tox_std_src": src_group["TOXICITY_SCORE"].std(),
        "tox_count_src": src_group.apply(lambda df: (df["TOXICITY_SCORE"] >= toxicity_threshold).sum()),
        "msg_count_src": src_group.size(),
        "sentiment_mean_src": src_group["LINK_SENTIMENT"].mean(),
    })
    
    # Target-side aggregates
    tgt_group = interactions_df.groupby("TARGET_SUBREDDIT")
    tgt_features = pd.DataFrame({
        "tox_mean_tgt": tgt_group["TOXICITY_SCORE"].mean(),
        "tox_std_tgt": tgt_group["TOXICITY_SCORE"].std(),
        "tox_count_tgt": tgt_group.apply(lambda df: (df["TOXICITY_SCORE"] >= toxicity_threshold).sum()),
        "msg_count_tgt": tgt_group.size(),
        "sentiment_mean_tgt": tgt_group["LINK_SENTIMENT"].mean(),
    })
    
    # Combine source + target with outer join
    all_subs = src_features.join(tgt_features, how="outer")
    
    # IMPROVED: Fill NaN values appropriately
    # For count columns: NaN means no activity, fill with 0
    count_cols = ["tox_count_src", "msg_count_src", "tox_count_tgt", "msg_count_tgt"]
    all_subs[count_cols] = all_subs[count_cols].fillna(0)
    
    # For mean/std columns: NaN means no data, fill with global mean after filtering
    mean_std_cols = ["tox_mean_src", "tox_std_src", "sentiment_mean_src",
                     "tox_mean_tgt", "tox_std_tgt", "sentiment_mean_tgt"]
    
    # IMPROVED: Compute toxicity rates with proper division by zero handling
    # Use np.divide with where parameter to avoid division by zero warnings
    all_subs["tox_rate_src"] = np.divide(
        all_subs["tox_count_src"].values,
        all_subs["msg_count_src"].values,
        out=np.zeros_like(all_subs["tox_count_src"].values, dtype=float),
        where=all_subs["msg_count_src"].values != 0
    )
    all_subs["tox_rate_tgt"] = np.divide(
        all_subs["tox_count_tgt"].values,
        all_subs["msg_count_tgt"].values,
        out=np.zeros_like(all_subs["tox_count_tgt"].values, dtype=float),
        where=all_subs["msg_count_tgt"].values != 0
    )
    
    # Behavioral balance metrics
    all_subs["tox_balance"] = all_subs["tox_count_src"] - all_subs["tox_count_tgt"]
    all_subs["msg_balance"] = all_subs["msg_count_src"] - all_subs["msg_count_tgt"]
    
    # IMPROVED: Filter low-activity subreddits
    total_activity = all_subs["msg_count_src"] + all_subs["msg_count_tgt"]
    active_mask = total_activity >= min_activity
    all_subs = all_subs[active_mask].copy()
    
    # Fill remaining NaN values in mean/std columns with column means
    for col in mean_std_cols:
        if col in all_subs.columns:
            all_subs[col] = all_subs[col].fillna(all_subs[col].mean())
    
    # Drop constant columns
    constant_cols = [c for c in all_subs.columns if all_subs[c].nunique() <= 1]
    features = all_subs.drop(columns=constant_cols)
    
    return features


def reduce_dimensions(X_scaled, method="umap", n_components=2, random_state=42, **kwargs):
    """
    Reduce dimensionality of scaled feature matrix.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix
    method : str
        "umap" or "pca"
    n_components : int
        Number of dimensions to reduce to
    random_state : int
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for the reducer
        
    Returns:
    --------
    embedding : np.ndarray
        Low-dimensional embedding
    reducer_name : str
        Name of the reduction method used
    """
    if method == "umap" and UMAP_AVAILABLE:
        # IMPROVED: Better default hyperparameters for global structure
        default_params = {
            "n_components": n_components,
            "n_neighbors": 50,  # Increased from 20 for better global structure
            "min_dist": 0.1,
            "metric": "euclidean",
            "random_state": random_state
        }
        default_params.update(kwargs)
        
        reducer = umap.UMAP(**default_params)
        embedding = reducer.fit_transform(X_scaled)
        reducer_name = "UMAP"
    else:
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(X_scaled)
        reducer_name = "PCA"
        
    return embedding, reducer_name


def find_optimal_clusters(X, X_scaled, method="kmeans", k_range=range(3, 21), 
                          eval_on_features=True, random_state=42, **kwargs):
    """
    Find optimal number of clusters using silhouette score.
    
    IMPROVED: Evaluates silhouette on original feature space, not embedding.
    
    Parameters:
    -----------
    X : np.ndarray
        Data to cluster on (usually 2D embedding)
    X_scaled : np.ndarray
        Scaled features for silhouette evaluation
    method : str
        "kmeans" or "hdbscan"
    k_range : range or list
        Range of k values to test (for KMeans)
    eval_on_features : bool
        If True, evaluate silhouette on X_scaled; if False, on X
    random_state : int
        Random seed
    **kwargs : dict
        Additional parameters for the clusterer
        
    Returns:
    --------
    cluster_labels : np.ndarray
        Cluster assignments
    cluster_method : str
        Description of clustering method and parameters
    best_score : float
        Best silhouette score achieved
    """
    if method == "hdbscan" and HDBSCAN_AVAILABLE:
        default_params = {
            "min_cluster_size": 10,
            "min_samples": 5,
            "metric": "euclidean"
        }
        default_params.update(kwargs)
        
        clusterer = hdbscan.HDBSCAN(**default_params)
        cluster_labels = clusterer.fit_predict(X)
        
        # Evaluate silhouette (excluding noise points labeled -1)
        mask = cluster_labels != -1
        if mask.sum() > 0 and len(set(cluster_labels[mask])) > 1:
            eval_data = X_scaled[mask] if eval_on_features else X[mask]
            score = silhouette_score(eval_data, cluster_labels[mask])
        else:
            score = -1
            
        cluster_method = f"HDBSCAN (silhouette={score:.3f})"
        best_score = score
        
    else:  # KMeans
        best_k, best_score, best_labels = None, -1, None
        
        for k in k_range:
            if k >= len(X):  # Can't have more clusters than samples
                continue
                
            km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            labels = km.fit_predict(X)
            
            # IMPROVED: Evaluate silhouette on feature space, not embedding
            eval_data = X_scaled if eval_on_features else X
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = silhouette_score(eval_data, labels) if len(set(labels)) > 1 else -1
                
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
                
        cluster_labels = best_labels
        cluster_method = f"KMeans (k={best_k}, silhouette={best_score:.3f})"
        
    return cluster_labels, cluster_method, best_score


def create_result_dataframe(features, cluster_labels, embedding):
    """
    Assemble clustering results into a DataFrame.
    
    Parameters:
    -----------
    features : pd.DataFrame
        Original feature matrix
    cluster_labels : np.ndarray
        Cluster assignments
    embedding : np.ndarray
        2D embedding coordinates
        
    Returns:
    --------
    result_df : pd.DataFrame
        DataFrame with features, cluster labels, and embedding coordinates
    """
    result_df = features.copy()
    result_df["cluster"] = cluster_labels
    result_df["embed_x"] = embedding[:, 0]
    result_df["embed_y"] = embedding[:, 1]
    
    return result_df


def profile_clusters(result_df, feature_cols=None):
    """
    Generate comprehensive statistical profiles for each cluster.
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        Clustering results with 'cluster' column
    feature_cols : list or None
        List of feature columns to profile. If None, uses all numeric columns
        except cluster, embed_x, embed_y
        
    Returns:
    --------
    profiles : pd.DataFrame
        Statistical summary for each cluster (mean, std, median, etc.)
    """
    if feature_cols is None:
        # Auto-detect feature columns
        exclude_cols = ['cluster', 'embed_x', 'embed_y']
        feature_cols = [c for c in result_df.columns 
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(result_df[c])]
    
    profiles = result_df.groupby('cluster')[feature_cols].agg(['mean', 'std', 'median', 'min', 'max'])
    
    return profiles


def identify_cluster_archetypes(result_df):
    """
    Classify clusters into behavioral archetypes based on their feature profiles.
    
    Archetypes:
    - Aggressor: High outgoing toxicity, low incoming
    - Victim: High incoming toxicity, low outgoing
    - Toxic Hub: High bidirectional toxicity
    - Neutral: Low toxicity in both directions
    - Active Non-toxic: High activity but low toxicity
    
    Parameters:
    -----------
    result_df : pd.DataFrame
        Clustering results with toxicity features
        
    Returns:
    --------
    archetypes : pd.DataFrame
        Cluster ID, archetype label, and key statistics
    """
    cluster_means = result_df.groupby('cluster').mean()
    
    archetypes = []
    for cluster_id in cluster_means.index:
        row = cluster_means.loc[cluster_id]
        
        # Extract key metrics
        tox_out = row.get('tox_rate_src', 0)
        tox_in = row.get('tox_rate_tgt', 0)
        msg_out = row.get('msg_count_src', 0)
        msg_in = row.get('msg_count_tgt', 0)
        tox_balance = row.get('tox_balance', 0)
        
        # Classify archetype
        high_tox_threshold = 0.15  # 15% toxicity rate
        
        if tox_out > high_tox_threshold and tox_in < high_tox_threshold:
            archetype = "Aggressor"
        elif tox_in > high_tox_threshold and tox_out < high_tox_threshold:
            archetype = "Victim"
        elif tox_out > high_tox_threshold and tox_in > high_tox_threshold:
            archetype = "Toxic Hub"
        elif tox_out < 0.05 and tox_in < 0.05:
            archetype = "Neutral"
        elif (msg_out + msg_in) > 100:
            archetype = "Active Non-toxic"
        else:
            archetype = "Low Activity"
            
        archetypes.append({
            'cluster': cluster_id,
            'archetype': archetype,
            'tox_rate_src': tox_out,
            'tox_rate_tgt': tox_in,
            'tox_balance': tox_balance,
            'total_messages': msg_out + msg_in,
            'n_subreddits': len(result_df[result_df['cluster'] == cluster_id])
        })
    
    return pd.DataFrame(archetypes)


def compute_feature_importance(X_scaled, feature_names, n_components=10):
    """
    Compute feature importance using PCA variance analysis.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix
    feature_names : list
        Names of features
    n_components : int
        Number of principal components to analyze
        
    Returns:
    --------
    importance_df : pd.DataFrame
        Feature importance scores and contributions to top PCs
    explained_variance : np.ndarray
        Variance explained by each PC
    """
    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Get component loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Compute importance as sum of absolute contributions across top PCs
    importance = np.abs(loadings[:, :5]).sum(axis=1)  # Top 5 PCs
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'PC1': loadings[:, 0] if n_components >= 1 else 0,
        'PC2': loadings[:, 1] if n_components >= 2 else 0,
        'PC3': loadings[:, 2] if n_components >= 3 else 0,
    }).sort_values('importance', ascending=False)
    
    return importance_df, pca.explained_variance_ratio_


def validate_cluster_stability(X, X_scaled, n_bootstrap=50, sample_frac=0.8, 
                               k=None, method="kmeans", random_state=42):
    """
    Validate cluster stability using bootstrap resampling.
    
    Computes adjusted rand index between clustering on full dataset
    and clusterings on bootstrap samples.
    
    Parameters:
    -----------
    X : np.ndarray
        Data to cluster on (2D embedding)
    X_scaled : np.ndarray
        Scaled features
    n_bootstrap : int
        Number of bootstrap iterations
    sample_frac : float
        Fraction of data to sample in each bootstrap
    k : int or None
        Number of clusters (if None, will find optimal)
    method : str
        "kmeans" or "hdbscan"
    random_state : int
        Random seed
        
    Returns:
    --------
    stability_scores : np.ndarray
        ARI scores for each bootstrap iteration
    mean_ari : float
        Mean ARI across all iterations
    """
    np.random.seed(random_state)
    
    # Get baseline clustering
    if k is None:
        base_labels, _, _ = find_optimal_clusters(X, X_scaled, method=method, random_state=random_state)
        k = len(set(base_labels))
    else:
        if method == "kmeans":
            km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
            base_labels = km.fit_predict(X)
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
            base_labels = clusterer.fit_predict(X)
    
    n_samples = int(len(X) * sample_frac)
    stability_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        X_boot = X[indices]
        
        # Cluster bootstrap sample
        if method == "kmeans":
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state + i)
            boot_labels = km.fit_predict(X_boot)
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
            boot_labels = clusterer.fit_predict(X_boot)
        
        # Compute ARI
        ari = adjusted_rand_score(base_labels[indices], boot_labels)
        stability_scores.append(ari)
    
    return np.array(stability_scores), np.mean(stability_scores)




# ======================================



# Add this cell BEFORE your clustering cell to engineer temporal features

# ===== TEMPORAL FEATURE ENGINEERING =====
import pandas as pd
import numpy as np

print("Engineering temporal features...")

# Ensure we have timestamps
if "TIMESTAMP" not in interactions_df.columns:
    raise ValueError("TIMESTAMP column required for temporal analysis")

# 1. TIME-BINNED TOXICITY RATES
# Calculate toxicity rate evolution for each subreddit over time windows

def compute_temporal_features(df, min_months=3):
    """
    Compute temporal toxicity patterns for each subreddit.
    
    Parameters:
    -----------
    min_months : int
        Minimum number of months of data required (default: 3)
    """
    temporal_features = []
    
    for subreddit in df['SOURCE_SUBREDDIT'].unique():
        sub_df = df[df['SOURCE_SUBREDDIT'] == subreddit].sort_values('TIMESTAMP')
        
        if len(sub_df) < 5:  # Skip very low-activity subreddits
            continue
        
        # Create time-binned data
        sub_df = sub_df.set_index('TIMESTAMP')
        
        # Resample to monthly frequency
        monthly = sub_df.resample('M').agg({
            'TOXICITY_SCORE': ['mean', 'std', 'count'],
            'TOXICITY_CAT': 'sum'
        })
        monthly.columns = ['tox_mean', 'tox_std', 'msg_count', 'tox_count']
        monthly = monthly[monthly['msg_count'] > 0]  # Keep only active months
        
        # Skip if insufficient temporal coverage
        if len(monthly) < min_months:
            continue
            
        monthly['tox_rate'] = monthly['tox_count'] / monthly['msg_count']
        
        # Safe series for calculations
        tox_rate_series = monthly['tox_rate'].fillna(0)
        msg_count_series = monthly['msg_count'].fillna(0)
        
        # TEMPORAL FEATURES
        features = {'subreddit': subreddit}
        
        # 1. TREND: Linear slope of toxicity over time
        try:
            features['tox_trend'] = np.polyfit(range(len(tox_rate_series)), tox_rate_series, 1)[0] if len(tox_rate_series) >= 2 else 0.0
        except:
            features['tox_trend'] = 0.0
        
        # 2. VOLATILITY: Standard deviation of monthly toxicity rates
        features['tox_volatility'] = tox_rate_series.std() if len(tox_rate_series) > 1 else 0.0
        
        # 3. ACCELERATION: Change in trend (quadratic term)
        try:
            features['tox_acceleration'] = np.polyfit(range(len(tox_rate_series)), tox_rate_series, 2)[0] if len(tox_rate_series) >= 3 else 0.0
        except:
            features['tox_acceleration'] = 0.0
        
        # 4. BURST FREQUENCY: How often does toxicity spike above mean + 2*std?
        if len(tox_rate_series) > 0:
            mean_rate = tox_rate_series.mean()
            std_rate = tox_rate_series.std()
            if std_rate > 0:
                features['burst_frequency'] = (tox_rate_series > mean_rate + 2*std_rate).sum() / len(tox_rate_series)
            else:
                features['burst_frequency'] = 0.0
        else:
            features['burst_frequency'] = 0.0
        
        # 5. PERSISTENCE: Longest consecutive period of above-average toxicity
        if len(tox_rate_series) > 0:
            median_rate = tox_rate_series.median()
            above_median = (tox_rate_series > median_rate).astype(int)
            if above_median.sum() > 0:
                streaks = above_median.groupby((above_median != above_median.shift()).cumsum()).sum()
                features['tox_persistence'] = streaks.max()
            else:
                features['tox_persistence'] = 0.0
        else:
            features['tox_persistence'] = 0.0
        
        # 6. PERIODICITY: Detect if toxicity follows patterns (use autocorrelation)
        try:
            if len(tox_rate_series) >= 4:
                features['tox_autocorr'] = tox_rate_series.autocorr(lag=1)
                if pd.isna(features['tox_autocorr']):
                    features['tox_autocorr'] = 0.0
            else:
                features['tox_autocorr'] = 0.0
        except:
            features['tox_autocorr'] = 0.0
        
        # 7. ACTIVITY PATTERN: Variance in monthly activity
        mean_activity = msg_count_series.mean()
        if mean_activity > 0:
            features['activity_variance'] = msg_count_series.std() / mean_activity
        else:
            features['activity_variance'] = 0.0
        
        # 8. EARLY vs LATE: Compare first half vs second half behavior
        mid_point = len(tox_rate_series) // 2
        if mid_point >= 2:
            features['tox_rate_early'] = tox_rate_series.iloc[:mid_point].mean()
            features['tox_rate_late'] = tox_rate_series.iloc[mid_point:].mean()
        else:
            features['tox_rate_early'] = 0.0
            features['tox_rate_late'] = 0.0
        
        features['tox_evolution'] = features['tox_rate_late'] - features['tox_rate_early']
        
        temporal_features.append(features)
    
    return pd.DataFrame(temporal_features).set_index('subreddit')

# Compute temporal features
temporal_df = compute_temporal_features(interactions_df, min_months=3)

print(f"Temporal features computed for {len(temporal_df)} subreddits")

# ===== APPLY SEMANTICALLY APPROPRIATE DEFAULTS FOR REMAINING NaN =====
print("\nHandling remaining NaN values with feature-specific defaults...")

# Compute medians for features where "unknown" = "typical behavior"
volatility_median = temporal_df['tox_volatility'].median()
burst_median = temporal_df['burst_frequency'].median()
activity_var_median = temporal_df['activity_variance'].median()

# Apply feature-specific defaults
default_values = {
    'tox_trend': 0.0,              # No trend = flat line
    'tox_volatility': volatility_median,  # Unknown volatility = typical volatility
    'tox_acceleration': 0.0,       # No acceleration = constant velocity
    'burst_frequency': burst_median,      # Unknown bursts = typical burst rate
    'tox_persistence': 0.0,        # No persistence = no sustained periods
    'tox_autocorr': 0.0,           # No autocorrelation = random pattern
    'activity_variance': activity_var_median, # Unknown activity variance = typical variance
    'tox_rate_early': 0.0,         # No early data = no early toxicity
    'tox_rate_late': 0.0,          # No late data = no late toxicity
    'tox_evolution': 0.0           # No evolution = no change
}

temporal_df = temporal_df.fillna(default_values)

# Final cleanup: ensure no inf values
temporal_df = temporal_df.replace([np.inf, -np.inf], 0.0)

# Validation
nan_count = temporal_df.isna().sum().sum()
inf_count = np.isinf(temporal_df.values).sum()

print(f"\n✓ Validation complete:")
print(f"  - NaN values remaining: {nan_count}")
print(f"  - Inf values remaining: {inf_count}")
print(f"  - Subreddits with complete features: {len(temporal_df)}")

print("\nTemporal feature columns:")
print(temporal_df.columns.tolist())
print("\nTemporal feature statistics:")
print(temporal_df.describe())

# Save for later use
temporal_df.to_csv('subreddit_temporal_features.csv')
print("\n✓ Temporal features saved to 'subreddit_temporal_features.csv'")



# ======================================



# Subreddit toxicity clustering WITH TEMPORAL FEATURES
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from src.utils.clustering_utils import (
    build_subreddit_features,
    reduce_dimensions,
    find_optimal_clusters,
    create_result_dataframe
)

# Ensure toxicity score exists
if "TOXICITY_SCORE" not in interactions_df.columns:
    interactions_df = compute_toxicity_score(interactions_df)

# ===== FEATURE ENGINEERING =====
print("Building subreddit features...")
features = build_subreddit_features(
    interactions_df=interactions_df,
    toxicity_threshold=TOXICITY_THRESHOLD,
    min_activity=10
)

# ===== MERGE TEMPORAL FEATURES =====
print("\nMerging temporal features...")
# Load temporal features if computed separately
# temporal_df = pd.read_csv('subreddit_temporal_features.csv', index_col=0)

# Join temporal features with static features
features_combined = features.join(temporal_df, how='inner')  # Inner join to keep only subreddits with both
print(f"Combined features shape: {features_combined.shape}")
print(f"New temporal features added: {list(temporal_df.columns)}")

# ===== SCALING =====
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_combined)

# ===== DIMENSIONALITY REDUCTION =====
print("\nReducing dimensions...")
embedding, reducer_name = reduce_dimensions(
    X_scaled=X_scaled,
    method="umap",
    n_components=2,
    n_neighbors=min(50, len(features_combined)//10),
    min_dist=0.3,
    random_state=42
)

# ===== CLUSTERING =====
print("\nFinding optimal clusters...")
max_k = min(20, len(features_combined)//300)
k_range = range(3, max_k + 1)

cluster_labels, cluster_method, best_score = find_optimal_clusters(
    X=embedding,
    X_scaled=X_scaled,
    method="kmeans",
    k_range=k_range,
    eval_on_features=True,
    random_state=42
)

print(f"\nClustering method: {cluster_method}")
print(f"Silhouette score: {best_score:.3f}")

# ===== ASSEMBLE RESULTS =====
result_df = create_result_dataframe(
    features=features_combined,
    cluster_labels=cluster_labels,
    embedding=embedding
)

# ===== CLUSTER INTERPRETATION =====
print("\n" + "="*60)
print("CLUSTER CHARACTERIZATION")
print("="*60)

cluster_profiles = result_df.groupby("cluster")[
    ['tox_rate_src', 'tox_rate_tgt', 'tox_balance', 
     'tox_trend', 'tox_volatility', 'tox_evolution', 
     'burst_frequency', 'tox_persistence']
].mean()

# Classify clusters by temporal behavior
def classify_cluster(row):
    if row['tox_trend'] > 0.001 and row['tox_evolution'] > 0:
        return "🔺 Escalating Aggressors"
    elif row['tox_trend'] < -0.001 and row['tox_evolution'] < 0:
        return "📉 Reforming Communities"
    elif row['burst_frequency'] > 0.2:
        return "💥 Episodic Volatiles"
    elif row['tox_persistence'] > 5:
        return "⚠️ Chronic Toxic"
    elif row['tox_rate_tgt'] > row['tox_rate_src']:
        return "🎯 Targeted Victims"
    elif row['tox_volatility'] < 0.05:
        return "😌 Stable Communities"
    else:
        return "➖ Neutral/Mixed"

cluster_profiles['type'] = cluster_profiles.apply(classify_cluster, axis=1)

print("\nCluster Profiles:")
print(cluster_profiles[['type', 'tox_trend', 'tox_evolution', 'burst_frequency', 'tox_persistence']])

# ===== ENHANCED VISUALIZATION =====
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Clusters colored by ID
palette = sns.color_palette("husl", n_colors=len(set(cluster_labels)))
sns.scatterplot(
    x="embed_x", y="embed_y", hue="cluster", data=result_df,
    palette=palette, s=30, alpha=0.7, edgecolor="none", ax=axes[0,0]
)
axes[0,0].set_title(f"Cluster Assignment\n{cluster_method}")
axes[0,0].legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot 2: Toxicity Trend (escalating vs reforming)
scatter2 = axes[0,1].scatter(
    result_df["embed_x"], result_df["embed_y"],
    c=result_df["tox_trend"], cmap="RdYlGn_r", s=30, alpha=0.7,
    vmin=-0.01, vmax=0.01
)
axes[0,1].set_title("Toxicity Trend\nRed=Escalating, Green=Reforming")
plt.colorbar(scatter2, ax=axes[0,1], label="Trend (slope)")

# Plot 3: Burst Frequency
scatter3 = axes[1,0].scatter(
    result_df["embed_x"], result_df["embed_y"],
    c=result_df["burst_frequency"], cmap="YlOrRd", s=30, alpha=0.7
)
axes[1,0].set_title("Toxicity Burst Frequency\nDarker = More Volatile")
plt.colorbar(scatter3, ax=axes[1,0], label="Burst Frequency")

# Plot 4: Evolution (early vs late behavior change)
scatter4 = axes[1,1].scatter(
    result_df["embed_x"], result_df["embed_y"],
    c=result_df["tox_evolution"], cmap="PuOr", s=30, alpha=0.7,
    vmin=-0.1, vmax=0.1
)
axes[1,1].set_title("Behavioral Evolution\nPurple=Got Worse, Orange=Got Better")
plt.colorbar(scatter4, ax=axes[1,1], label="Evolution")

for ax in axes.flat:
    ax.set_xlabel(f"{reducer_name} Dimension 1")
    ax.set_ylabel(f"{reducer_name} Dimension 2")
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()

# ===== SAMPLE SUBREDDITS FROM EACH CLUSTER TYPE =====
print("\n" + "="*60)
print("SAMPLE SUBREDDITS BY CLUSTER TYPE")
print("="*60)

for cluster_id in sorted(result_df["cluster"].unique()):
    cluster_type = cluster_profiles.loc[cluster_id, 'type']
    cluster_subs = result_df[result_df["cluster"] == cluster_id].index[:5]
    print(f"\nCluster {cluster_id}: {cluster_type}")
    print(f"  Examples: {', '.join(cluster_subs)}")
    
result_df.head()