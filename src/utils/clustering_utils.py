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