import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import umap


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

## df, umap_available, hdbscan_available = preliminary_control(interactions_df)


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
    

def event_rate(ts: pd.Series) -> float:
    """
    This function calculates the average event rate (events per day) over the active period.
    This is a simpler, more interpretable alternative to regression-based trend.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.

    Returns:
    float: The average event rate (events/day). Returns np.nan if not computable.
    """
    # We ensure datetime type, drop missing values, sort in ascending order and remove duplicates.
    ts_sorted = pd.to_datetime(ts, errors='coerce').dropna().sort_values().drop_duplicates()
    # If we have less than 2 timestamps, we cannot compute a rate
    if len(ts_sorted) < 2:
        return np.nan

    # We compute the time span from first to last event
    time_span_days = (ts_sorted.max() - ts_sorted.min()).total_seconds() / 86400.0
    
    # We avoid division by zero for same-day events
    if time_span_days < 1.0:
        time_span_days = 1.0
    
    # We compute average rate as total events / days active
    event_count = len(ts_sorted)
    rate = event_count / time_span_days
    
    return float(rate)


def pct_events_recent(ts: pd.Series, recent_days: int = 90) -> float:
    """
    This function calculates the percentage of events that occurred in the recent period.
    More interpretable than recent_activity ratio.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.
    recent_days (int): The number of days to consider as the recent period (default: 90).

    Returns:
    float: Percentage of events in recent period (0-100). Returns np.nan if not computable.
    """
    # We ensure datetime type, drop missing values and sort in ascending order.
    ts_sorted = pd.to_datetime(ts, errors="coerce", utc=True).dropna().sort_values()
    if len(ts_sorted) < 2:
        return np.nan
    
    # We define cutoff as being the start of the recent window
    cutoff = ts_sorted.max() - pd.Timedelta(days=int(recent_days))
    
    recent_count = int((ts_sorted >= cutoff).sum())
    total_count = len(ts_sorted)
    
    # We compute percentage
    pct = 100.0 * recent_count / total_count
    
    return float(pct)


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


def gap_cv(ts: pd.Series) -> float:
    """
    This function calculates the coefficient of variation for gaps between events.
    CV = std/mean, which is scale-independent unlike raw std.

    Parameters:
    ts (pd.Series): A pandas Series expected to contain datetime objects.

    Returns:
    float: Coefficient of variation for gaps. Returns np.nan if not computable.
    """
    # We ensure datetime type, drop missing values, sort in ascending order and remove duplicates.
    ts_sorted = pd.to_datetime(ts, errors='coerce').dropna().sort_values().drop_duplicates()
    # If we have less than 3 timestamps, we cannot compute stable CV
    if len(ts_sorted) < 3:
        return np.nan
    # We compute the difference between each timestamp and the previous one (in days)
    deltas_days = ts_sorted.diff().dt.total_seconds() / 86400
    deltas_days = deltas_days.dropna()
    
    if len(deltas_days) < 2:
        return np.nan
    
    mean_gap = deltas_days.mean()
    std_gap = deltas_days.std(ddof=1)
    
    # We avoid division by zero
    if mean_gap <= 0:
        return np.nan
    
    cv = std_gap / mean_gap
    return float(cv)


# ====================
# ROLE-AWARE FEATURE EXTRACTION
# ====================

# # # We build separate features for source (aggressor) and target (victim) roles
# # toxic_df = interactions_df[interactions_df["TOXICITY_CAT"] == 1].copy()

# ===== SOURCE FEATURES (subreddit as aggressor) =====
print("\nComputing SOURCE (aggressor) features...")

def comp_source_features(toxic_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes source (aggressor) features for subreddits.

    Parameters:
    toxic_df (pd.DataFrame): Dataframe containing toxic interactions with TIMESTAMP, SOURCE_SUBREDDIT, TARGET_SUBREDDIT columns.

    Returns:
    pd.DataFrame: Dataframe with computed source features for each subreddit.
    """
    # We remove any rows with missing SOURCE_SUBREDDIT to build source features
    source_df = toxic_df[["TIMESTAMP", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].dropna(subset=["SOURCE_SUBREDDIT"])
    grp_src = source_df.groupby("SOURCE_SUBREDDIT")

    # We compute source features for each subreddit
    source_features = pd.DataFrame({
        "src_tox_count": grp_src["TIMESTAMP"].size(),
        "src_first_tox_ts": grp_src["TIMESTAMP"].min(),
        "src_last_tox_ts": grp_src["TIMESTAMP"].max(),
    })

    source_features["src_active_span_days"] = (
        (source_features["src_last_tox_ts"] - source_features["src_first_tox_ts"])
        .dt.total_seconds() / 86400
    )

    source_features["src_median_gap_days"] = grp_src["TIMESTAMP"].apply(median_gap_days)
    source_features["src_gap_cv"] = grp_src["TIMESTAMP"].apply(gap_cv)
    source_features["src_max_30d_count"] = grp_src["TIMESTAMP"].apply(
        lambda s: max_events_in_window(s, window_days=30)
    )
    source_features["src_event_rate"] = grp_src["TIMESTAMP"].apply(event_rate)
    source_features["src_pct_events_recent"] = grp_src["TIMESTAMP"].apply(pct_events_recent)
    print(f"  SOURCE features computed for {len(source_features)} subreddits")

    return source_df, source_features

# # source_df, source_features = comp_source_features(toxic_df)


# ===== TARGET FEATURES (subreddit as victim) =====
# Here we perfom the same operations as above but for target subreddits


def comp_target_features(toxic_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes target (victim) features for subreddits.

    Parameters:
    toxic_df (pd.DataFrame): Dataframe containing toxic interactions with TIMESTAMP, SOURCE_SUBREDDIT, TARGET_SUBREDDIT columns.

    Returns:
    pd.DataFrame: Dataframe with computed target features for each subreddit.
    """
    target_df = toxic_df[["TIMESTAMP", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].dropna(subset=["TARGET_SUBREDDIT"])
    grp_tgt = target_df.groupby("TARGET_SUBREDDIT")

    target_features = pd.DataFrame({
        "tgt_tox_count": grp_tgt["TIMESTAMP"].size(),
        "tgt_first_tox_ts": grp_tgt["TIMESTAMP"].min(),
        "tgt_last_tox_ts": grp_tgt["TIMESTAMP"].max(),
    })

    target_features["tgt_active_span_days"] = (
        (target_features["tgt_last_tox_ts"] - target_features["tgt_first_tox_ts"])
        .dt.total_seconds() / 86400
    )

    target_features["tgt_median_gap_days"] = grp_tgt["TIMESTAMP"].apply(median_gap_days)
    target_features["tgt_gap_cv"] = grp_tgt["TIMESTAMP"].apply(gap_cv)
    target_features["tgt_pct_events_recent"] = grp_tgt["TIMESTAMP"].apply(pct_events_recent)
    print(f"  TARGET features computed for {len(target_features)} subreddits")

    return target_df, target_features

# # target_df, target_features = comp_target_features(toxic_df)


# ===== NETWORK FEATURES =====


def compute_out_in_degree(source_df: pd.DataFrame, target_df: pd.DataFrame) -> (pd.Series, pd.Series):
    """
    This function computes out-degree and in-degree network features for subreddits.

    Parameters:
    source_df (pd.DataFrame): Dataframe containing SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns.
    target_df (pd.DataFrame): Dataframe containing SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns.

    Returns:
    (pd.Series, pd.Series): Tuple of Series containing out-degree and in-degree features.
    """
    # We compute the out-degree as the number of unique subreddits one attacks
    out_degree = source_df.groupby("SOURCE_SUBREDDIT")["TARGET_SUBREDDIT"].nunique().rename("net_out_degree")

    # We compute the in-degree as the number of unique subreddits that attack one
    in_degree = target_df.groupby("TARGET_SUBREDDIT")["SOURCE_SUBREDDIT"].nunique().rename("net_in_degree")

    return out_degree, in_degree

# # out_degree, in_degree = compute_out_in_degree(source_df, target_df)

def compute_reciprocity(toxic_df: pd.DataFrame,subreddit_name: str) -> float:
    """
    This function computes the fraction of outgoing toxic edges that are reciprocated
    
    Parameters:
    toxic_df (pd.DataFrame): Dataframe containing toxic interactions with SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns.
    subreddit_name (str): The name of the subreddit to compute reciprocity for.

    Returns:
    float: The reciprocity value (0.0 to 1.0) for the subreddit.
    """
    # We compute reciprocity: for each source->target pair, we check if target->source pair also exists
    edges = toxic_df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].dropna()
    edges_set = set(zip(edges["SOURCE_SUBREDDIT"], edges["TARGET_SUBREDDIT"]))
    
    outgoing = edges[edges["SOURCE_SUBREDDIT"] == subreddit_name]["TARGET_SUBREDDIT"].unique()
    if len(outgoing) == 0:
        return 0.0
    
    # Count how many outgoing edges also exist in the opposite direction
    reciprocated = sum(1 for target in outgoing if (target, subreddit_name) in edges_set)

    # Reciprocity is the fraction of outgoing edges that are reciprocated
    return reciprocated / len(outgoing)


def call_compute_reciprocity(source_df:pd.DataFrame, target_df: pd.DataFrame, toxic_df: pd.DataFrame) -> float:
    """"
    Wrapper function to call compute_reciprocity.

    Parameters:
    source_df (pd.DataFrame): Dataframe containing SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns.
    toxic_df (pd.DataFrame): Dataframe containing toxic interactions with SOURCE_SUBREDDIT and TARGET_SUBREDDIT columns.

    Returns:
    reciprocity (pd.Series): Series containing reciprocity value for each subreddit.
    """
    # We get all unique subreddits involved in toxic interactions
    all_subreddits = set(source_df["SOURCE_SUBREDDIT"].unique()) | set(target_df["TARGET_SUBREDDIT"].unique())

    # We compute reciprocity for each subreddit
    reciprocity = pd.Series({sub: compute_reciprocity(toxic_df, sub) for sub in all_subreddits}, name="net_reciprocity")

    return reciprocity

# # reciprocity = call_compute_reciprocity(source_df, target_df, toxic_df)

# ===== MERGE ALL FEATURES =====

def merge_all_features(source_features: pd.DataFrame, target_features: pd.DataFrame, 
                       out_degree: pd.Series, in_degree: pd.Series, reciprocity: pd.Series) -> pd.DataFrame:
    """
    This function merges source, target, and network features into a single dataframe.

    Parameters:
    source_features (pd.DataFrame): Dataframe with source features.
    target_features (pd.DataFrame): Dataframe with target features.
    out_degree (pd.Series): Series with out-degree network feature.
    in_degree (pd.Series): Series with in-degree network feature.
    reciprocity (pd.Series): Series with reciprocity network feature.

    Returns:
    temporal_features(pd.DataFrame): Dataframe with all merged features.
    """
    # We merge source, target, and network features into a single dataframe
    temporal_features = source_features.join(target_features, how="outer")
    temporal_features = temporal_features.join(out_degree, how="outer")
    temporal_features = temporal_features.join(in_degree, how="outer")
    temporal_features = temporal_features.join(reciprocity, how="outer")

    temporal_features.fillna(0, inplace=True)

    # We compute compute derived features
    temporal_features["net_toxicity"] = temporal_features["src_tox_count"] - temporal_features["tgt_tox_count"]
    temporal_features["total_tox_count"] = temporal_features["src_tox_count"] + temporal_features["tgt_tox_count"]
    
    print(f"\nTotal subreddits before filtering: {len(temporal_features)}")

    return temporal_features

# # temporal_features = merge_all_features(source_features, target_features, out_degree, in_degree, reciprocity)


# ====================
# FILTER LOW-ACTIVITY SUBREDDITS
# ====================


def filter_low_activity(temporal_features: pd.DataFrame, min_toxic_events: int = 10) -> pd.DataFrame:
    """
    This function filters out subreddits with total toxic events below a specified threshold.

    Parameters:
    temporal_features (pd.DataFrame): Dataframe with all merged features.
    min_toxic_events (int): Minimum number of total toxic events required to keep a subreddit.

    Returns:
    pd.DataFrame: Filtered dataframe with subreddits meeting the activity threshold.
    """
    filtered_features = temporal_features[temporal_features["total_tox_count"] >= min_toxic_events].copy()
    
    print(f"Subreddits after filtering (>= 10 total toxic events): {len(filtered_features)}")

    return filtered_features

# # temporal_features = filter_low_activity(temporal_features, min_toxic_events=10)


# ====================
# NaN HANDLING
# ====================


def temporal_nan_handling(temporal_features: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles NaN values in temporal features according to specified rules.

    Parameters:
    temporal_features (pd.DataFrame): Dataframe with all merged features.

    Returns:
    pd.DataFrame: Dataframe with NaN values handled.
    """    
    # For temporal gap metrics, we replace 0 with median of non-zero values
    for prefix in ["src", "tgt"]:
        for col in ["median_gap_days", "gap_std"]:
            full_col = f"{prefix}_{col}"
            if full_col in temporal_features.columns:
                median_val = temporal_features[full_col].replace(0, np.nan).median()
                temporal_features[full_col] = temporal_features[full_col].replace(0, median_val)

    return temporal_features


def other_nan_handling(temporal_features: pd.DataFrame) -> pd.DataFrame:
    """
    This function handles NaN values in non-temporal features according to specified rules.

    Parameters:
    temporal_features (pd.DataFrame): Dataframe with all merged features.

    Returns:
    pd.DataFrame: Dataframe with NaN values handled.
    """
    # For other metrics, we handle NaNs as follows
    for prefix in ["src", "tgt"]:
        # For event_rate and pct_events_recent, fill NaN with median
        for col in ["event_rate", "pct_events_recent", "gap_cv"]:
            full_col = f"{prefix}_{col}"
            if full_col in temporal_features.columns:
                temporal_features[full_col] = temporal_features[full_col].fillna(
                    temporal_features[full_col].median()
                )
    
    return temporal_features

# temporal_features = temporal_nan_handling(temporal_features)
# temporal_features = other_nan_handling(temporal_features)


# ====================
# DIAGNOSTIC: Check for extreme outliers
# ====================


def check_extreme_outliers(temporal_features: pd.DataFrame):
    """
    This function checks for extreme outliers in specified columns of the temporal features dataframe.

    Parameters:
    temporal_features (pd.DataFrame): Dataframe with all merged features.
    """
    diag_cols = ["src_event_rate", "src_pct_events_recent", 
                "tgt_pct_events_recent",
                "net_toxicity", "net_in_degree", "net_out_degree"]
    for col in diag_cols:
        if col in temporal_features.columns:
            p99 = temporal_features[col].quantile(0.99)
            # For each feature, we identify subreddits above 99th percentile
            # Those subreddits are considered extreme outliers for diagnostics
            extreme = temporal_features[temporal_features[col] > p99]
            print(f"\n{col}:")
            print(f"  99th percentile: {p99:.2f}")
            print(f"  Max value: {temporal_features[col].max():.2f}")
            print(f"  N extreme (>99th): {len(extreme)}")
            if len(extreme) <= 10:
                print(f"  Extreme subreddits: {list(extreme.index[:10])}")

# # check_extreme_outliers(temporal_features)

# ====================
# OUTLIER HANDLING WITH LOG TRANSFORM
# ====================

# We select features for clustering
feature_cols = [
    # Source (aggressor) features
    "src_tox_count",
    "src_active_span_days",
    "src_median_gap_days",
    "src_gap_cv",
    "src_max_30d_count",
    "src_event_rate",
    "src_pct_events_recent",
    # Target (victim) features
    "tgt_tox_count",
    "tgt_active_span_days",
    "tgt_median_gap_days",
    "tgt_gap_cv",
    "tgt_pct_events_recent",
    # Network features
    "net_in_degree",
    "net_out_degree",
    "net_reciprocity",
]

# Note: is_mega will be added after it's computed (line ~870)
# We'll add it to features dataframe separately below

# # features = temporal_features[feature_cols].copy()

# ====================
# NaN IMPUTATION IMPACT REPORT
# ====================


def report_nan_impact(temporal_features: pd.DataFrame, feature_cols: list):
    """
    This function reports the percentage of NaN values before imputation and documents imputation choices.
    
    Parameters:
    temporal_features (pd.DataFrame): Dataframe with all merged features after imputation.
    feature_cols (list): List of feature column names.
    """
    # We define the imputation strategy used for each feature category
    imputation_strategy = {
        "gap_metrics": "Replaced 0 with median of non-zero values (median_gap_days)",
        "event_rate": "Filled NaN with feature median (events per day)",
        "pct_events_recent": "Filled NaN with feature median (percentage 0-100)",
        "gap_cv": "Filled NaN with feature median (coefficient of variation)",
        "other": "Filled remaining NaN with feature median"
    }
    
    print("\nImputation strategy by feature category:")
    for category, strategy in imputation_strategy.items():
        print(f"  {category}: {strategy}")

# # report_nan_impact(temporal_features, feature_cols)

# ====================
# FEATURE CORRELATION AND MULTICOLLINEARITY ANALYSIS
# ====================


def compute_feature_correlations(features: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes the correlation matrix for features and identifies highly correlated pairs.
    
    Parameters:
    features (pd.DataFrame): Dataframe with features before transformation.
    
    Returns:
    pd.DataFrame: Correlation matrix.
    """
    # We compute Pearson correlation matrix
    corr_matrix = features.corr(method='pearson')
    
    # We identify highly correlated feature pairs (|r| > 0.8)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            # We compute correlation between each pair of distinct features
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print("\nHighly correlated feature pairs (|r| > 0.8):")
        for feat1, feat2, r in high_corr_pairs:
            print(f"  {feat1} <-> {feat2}: r={r:.3f}")
    else:
        print("\nNo highly correlated feature pairs found (|r| > 0.8)")
    
    return corr_matrix


def compute_vif(features: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes Variance Inflation Factor (VIF) for each feature to detect multicollinearity.
    
    Parameters:
    features (pd.DataFrame): Dataframe with features before transformation.
    
    Returns:
    pd.DataFrame: Dataframe with VIF scores for each feature.
    """
    from sklearn.linear_model import LinearRegression
    
    vif_data = []
    # We compute VIF for each feature by regressing it against all other features
    for i, col in enumerate(features.columns):
        # We use all other features as predictors
        X = features.drop(columns=[col]).values
        y = features[col].values
        
        # We fit linear regression
        lr = LinearRegression()
        lr.fit(X, y)
        r_squared = lr.score(X, y)
        
        # We compute VIF = 1 / (1 - R**2)
        # VIF > 10 indicates severe multicollinearity, VIF > 5 suggests investigation
        vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
        vif_data.append({'feature': col, 'VIF': vif})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    print("\nVariance Inflation Factor (VIF) scores:")
    print("  (VIF > 10 = severe multicollinearity, VIF > 5 = moderate)")
    for _, row in vif_df.head(10).iterrows():
        flag = "HIGH" if row['VIF'] > 10 else "MOD" if row['VIF'] > 5 else "LOW"
        print(f"  {row['feature']:30s}: {row['VIF']:7.2f} {flag}")
    
    return vif_df

# We perform correlation and VIF analysis
# corr_matrix = compute_feature_correlations(features)
# vif_scores = compute_vif(features)

# ====================
# ITERATIVE VIF-BASED FEATURE REDUCTION
# ====================


def remove_high_vif_features(features: pd.DataFrame, threshold: float = 10.0, max_iterations: int = 20) -> pd.DataFrame:
    """
    This function iteratively removes features with VIF above threshold until all features have acceptable VIF.
    
    Parameters:
    features (pd.DataFrame): Dataframe with features before transformation.
    threshold (float): VIF threshold above which features are removed (default: 10.0).
    max_iterations (int): Maximum number of iterations to prevent infinite loops.
    
    Returns:
    pd.DataFrame: Dataframe with high-VIF features removed.
    """
    from sklearn.linear_model import LinearRegression
    
    features_reduced = features.copy()
    removed_features = []
    iteration = 0
    
    print(f"\nRemoving features with VIF > {threshold}:")
    print(f"  Starting with {len(features_reduced.columns)} features")
    
    while iteration < max_iterations:
        iteration += 1
        
        # We compute VIF for all current features
        vif_data = []
        for col in features_reduced.columns:
            X = features_reduced.drop(columns=[col]).values
            y = features_reduced[col].values
            
            lr = LinearRegression()
            lr.fit(X, y)
            r_squared = lr.score(X, y)
            
            # We compute VIF = 1 / (1 - R**2)
            vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
            vif_data.append({'feature': col, 'VIF': vif})
        
        vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
        
        # We find the feature with highest VIF
        max_vif_feature = vif_df.iloc[0]['feature']
        max_vif_value = vif_df.iloc[0]['VIF']
        
        # We stop if all VIF values are below threshold
        if max_vif_value <= threshold:
            print(f"\n All features now have VIF <= {threshold}")
            break
        
        # We remove the feature with highest VIF
        print(f"  Iteration {iteration}: Removing '{max_vif_feature}' (VIF={max_vif_value:.2f})")
        features_reduced = features_reduced.drop(columns=[max_vif_feature])
        removed_features.append((max_vif_feature, max_vif_value))
    
    if iteration >= max_iterations:
        print(f"\n Warning: Reached maximum iterations ({max_iterations})")
    
    print(f"\nFeature reduction summary:")
    print(f"  Features removed: {len(removed_features)}")
    print(f"  Features remaining: {len(features_reduced.columns)}")
    
    if removed_features:
        print(f"\nRemoved features (in order):")
        for feat, vif_val in removed_features:
            print(f"  - {feat:30s} (VIF={vif_val:.2f})")
    
    # We update feature_cols to reflect removed features
    feature_cols = list(features_reduced.columns)
    features = features_reduced.copy()

    print(f"\nUpdated feature list ({len(feature_cols)} features):")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    return feature_cols, features, features_reduced, removed_features

# We apply VIF-based feature reduction
# # feature_cols, features, features_reduced, removed_features = remove_high_vif_features(features, threshold=10.0)

# ====================
# FINAL VIF VERIFICATION
# ====================


def verify_final_vif(features: pd.DataFrame) -> pd.DataFrame:
    """
    This function computes and displays final VIF scores after feature reduction to verify multicollinearity is resolved.
    
    Parameters:
    features (pd.DataFrame): Dataframe with reduced features.
    
    Returns:
    pd.DataFrame: Dataframe with final VIF scores.
    """
    from sklearn.linear_model import LinearRegression
    
    vif_data = []
    for col in features.columns:
        X = features.drop(columns=[col]).values
        y = features[col].values
        
        lr = LinearRegression()
        lr.fit(X, y)
        r_squared = lr.score(X, y)
        
        vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
        vif_data.append({'feature': col, 'VIF': vif})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    
    print(f"\nFinal VIF scores for all {len(features.columns)} remaining features:")
    print("  (All should be ≤ 10.0 for acceptable multicollinearity)")
    
    all_acceptable = True
    for _, row in vif_df.iterrows():
        if row['VIF'] > 10:
            flag = "HIGH"
            all_acceptable = False
        elif row['VIF'] > 5:
            flag = "MOD"
        else:
            flag = "LOW"
        print(f"  {row['feature']:30s}: {row['VIF']:7.2f} {flag}")
    
    if all_acceptable:
        print(f"\n SUCCESS: All features have VIF <= 10.0")
        print(f"  Multicollinearity has been successfully resolved!")
    else:
        print(f"\n WARNING: Some features still have VIF > 10.0")
    
    return vif_df

# # final_vif_scores = verify_final_vif(features)

def log_transform (features: pd.DataFrame) -> pd.DataFrame:
    """
    This function applies log-transformation to specified count-based features to reduce the effect of outliers.

    Parameters:
    features (pd.DataFrame): Dataframe with features.

    Returns:
    features(pd.DataFrame): Dataframe with log-transformed features.
    """
    # Log-transform count-based columns to reduce effect of outliers
    # Note: is_mega is NOT log-transformed since it's binary (0/1)
    log_cols = [
        "src_tox_count", "src_active_span_days", "src_median_gap_days", "src_gap_cv", 
        "src_max_30d_count", "src_event_rate",
        "tgt_tox_count", "tgt_active_span_days", "tgt_median_gap_days", "tgt_gap_cv",
        "net_in_degree", "net_out_degree", "net_reciprocity"
    ]
    for col in log_cols:
        if col in features.columns:
            features[col] = np.log1p(features[col])  # log1p handles 0 values by having lop1p(0) = 0

    return features

# # features = log_transform(features)

def apply_winsorization(features: pd.DataFrame, percentile: float) -> pd.DataFrame:
    """
    This function applies winsorization to specified columns of the features dataframe.

    Parameters:
    features (pd.DataFrame): Dataframe with features.
    percentile (float): The percentile threshold for winsorization (e.g., 0.90 for 90th percentile).

    Returns:
    features(pd.DataFrame): Dataframe with winsorized features.
    """
    # We only apply winsorization for non-logged features (percentages, ratios, net_toxicity)
    # Note: We don't winsorize log-transformed features since log already handles outliers
    cols = ["net_toxicity", "src_pct_events_recent", "tgt_pct_events_recent"]  # net_toxicity is not log-transformed, needs capping
    for col in cols:
        if col in features.columns:
            # For each specified column, we compute the percentile threshold and clip values above it
            p_thresh = features[col].quantile(percentile)
            features[col] = features[col].clip(upper=p_thresh)
            print(f"Winsorized {col} at {int(percentile*100)}th percentile: {p_thresh:.2f}")
    return features

# # features = apply_winsorization(features, 0.90)

# We replace any remaining inf/nan values before scaling
def handle_inf_nan(features: pd.DataFrame) -> pd.DataFrame:
    """
    This function replaces inf values with NaN and fills NaN values with median of each column.

    Parameters:
    features (pd.DataFrame): Dataframe with features.

    Returns:
    features(pd.DataFrame): Dataframe with inf and NaN values handled.
    """
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(features.median(), inplace=True)
    
    return features

# # features = handle_inf_nan(features)


# ====================
# DETECT MEGA-SUBREDDITS
# ====================


def detect_mega_subreddits(features: pd.DataFrame, temporal_features: pd.DataFrame, feature_cols: list) -> tuple:
    """
    This function detects mega-subreddits based on 99th percentile thresholds for key features.
    Mega-subreddits are flagged for interpretation but not removed from clustering.

    Parameters:
    features (pd.DataFrame): Dataframe with features.
    temporal_features (pd.DataFrame): Dataframe with all merged features.
    feature_cols (list): List of feature column names.

    Returns:
    temporal_features(pd.DataFrame): Dataframe with 'is_mega' flag added.
    feature_cols(list): Updated list of feature column names with 'is_mega' added.
    """
    print(f"  Total subreddits for clustering: {len(temporal_features)}")

    # We identify mega-subreddits for labeling but DO NOT remove them
    mega_mask = (
        (temporal_features["net_in_degree"] > temporal_features["net_in_degree"].quantile(0.99)) |
        (temporal_features["net_out_degree"] > temporal_features["net_out_degree"].quantile(0.99)) |
        (temporal_features["total_tox_count"] > temporal_features["total_tox_count"].quantile(0.99))
    )

    print(f"  Mega-subreddits (>99th percentile): {mega_mask.sum()}")
    if mega_mask.sum() > 0:
        print(f"    These will be included in clustering and labeled for interpretation")
        print(f"\n  Top 10 mega-subreddits by total activity:")
        mega_subreddits = temporal_features[mega_mask]
        for idx, row in mega_subreddits.nlargest(10, 'total_tox_count')[['total_tox_count', 'src_tox_count', 'tgt_tox_count', 'net_in_degree', 'net_out_degree']].iterrows():
            print(f"    {idx}: total={row['total_tox_count']:.0f}, src={row['src_tox_count']:.0f}, tgt={row['tgt_tox_count']:.0f}, in_deg={row['net_in_degree']:.0f}, out_deg={row['net_out_degree']:.0f}")

    # Add mega-subreddit flag to temporal_features for later interpretation
    temporal_features['is_mega'] = mega_mask

    # Add is_mega to features for clustering (binary feature: 0 or 1)
    features['is_mega'] = temporal_features.loc[features.index, 'is_mega'].astype(int)
    feature_cols.append('is_mega')
    print(f"\n  Added 'is_mega' as binary feature (0/1) to clustering features")
    print(f"  Total features for clustering: {len(feature_cols)}")

    # We verify no inf/nan remain
    print(f"\nData quality check before scaling:")
    print(f"  NaN values: {features.isna().sum().sum()}")
    print(f"  Inf values: {np.isinf(features.values).sum()}")
    print(f"  Max value: {features.max().max():.2f}")

    return features, temporal_features, feature_cols

# # features, temporal_features, feature_cols = detect_mega_subreddits(features, temporal_features, feature_cols)

# ====================
# STANDARD SCALING
# ====================
# We use StandardScaler (mean=0, variance=1) as it is appropriate after outlier removal.
# This makes all features equally important for PCA

def standard_scaling(features: pd.DataFrame) -> np.ndarray:
    """
    This function applies standard scaling to the features dataframe.

    Parameters:
    features (pd.DataFrame): Dataframe with features.

    Returns:
    X_scaled(np.ndarray): Numpy array with scaled features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled

# # X_scaled = standard_scaling(features)

# ====================
# FEATURE IMPORTANCE ANALYSIS
# ====================


def analyze_pca_components(feature_cols: list, X_scaled: np.ndarray,variance_threshold: float = 0.80) -> tuple:
    """
    This function analyzes PCA components and determines optimal dimensionality.
    
    Parameters:
    feature_cols (list): List of feature column names.
    X_scaled (np.ndarray): Numpy array with scaled features.
    variance_threshold (float): Cumulative variance threshold for component selection (default: 0.80).
    
    Returns:
    n_components_optimal(int): Optimal number of components to retain.
    feature_importance(pd.DataFrame): Dataframe with feature importance scores.
    """
    # We create a PCA model that keeps all components for analysis
    pca_full = PCA(n_components=min(len(feature_cols), len(X_scaled)), random_state=42)
    pca_full.fit(X_scaled)

    # We compute the cumulative variance explained
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    # We find the number of components needed to explain variance_threshold of variance
    n_components_optimal = np.argmax(cumvar >= variance_threshold) + 1
    
    print(f"\nAnalysis Results:")
    print(f"  Total components: {pca_full.n_components_}")
    print(f"  Components for {variance_threshold*100:.0f}% variance: {n_components_optimal}")
    print(f"  Variance explained by first {n_components_optimal} PCs: {cumvar[n_components_optimal-1]*100:.1f}%")
    print(f"  DIMENSIONALITY REDUCTION: Clustering will use {n_components_optimal}D PC space instead of {len(feature_cols)}D feature space")
    
    # We show variance for first 10 components
    print(f"\nVariance explained by first 10 components:")
    for i in range(min(10, pca_full.n_components_)):
        var_pct = pca_full.explained_variance_ratio_[i] * 100
        cumvar_pct = cumvar[i] * 100
        bar = '█' * int(var_pct / 2)
        highlight = ' <- RETAINED' if i < n_components_optimal else ' (discarded)'
        print(f"  PC{i+1:2d}: {var_pct:5.1f}% (cumulative: {cumvar_pct:5.1f}%) {bar}{highlight}")
    
    # We analyze which original features are most important for retained PCs    
    # We compute total contribution of each feature across retained PCs
    feature_importance = np.zeros(len(feature_cols))
    for pc_idx in range(n_components_optimal):
        # We compute the weighted loading of each original feature for a given PC, taking into account both
        # the absolute contribution and the variance explained by that PC
        weighted_loadings = np.abs(pca_full.components_[pc_idx]) * pca_full.explained_variance_ratio_[pc_idx]
        feature_importance += weighted_loadings
    
    # We create a dataframe for easy interpretation
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance,
        'importance_pct': 100 * feature_importance / feature_importance.sum()
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 most contributing original features to {n_components_optimal} retained PCs:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s}: {row['importance_pct']:5.1f}%")
    
    # We display the top features for first 3 PCs
    print(f"\nTop features for first 3 principal components:")
    for pc_idx in range(min(3, pca_full.n_components_)):
        loadings = pca_full.components_[pc_idx]
        variance = pca_full.explained_variance_ratio_[pc_idx] * 100
        
        # Get top 5 most important features ([::-1] is used to revese the order for descending instead of ascending)
        top_indices = np.argsort(np.abs(loadings))[-5:][::-1]
        
        print(f"\nPC{pc_idx+1} ({variance:.1f}% variance):")
        for idx in top_indices:
            feat_name = feature_cols[idx]
            loading = loadings[idx]
            print(f"  {feat_name:30s}: {loading:+.3f}")

    return n_components_optimal, importance_df

# # n_components_optimal, feature_importance_df = analyze_pca_components(feature_cols, X_scaled, variance_threshold=0.80)


# ====================
# TRANSFORM TO OPTIMAL PC SPACE FOR CLUSTERING
# ====================

def dimensionality_reduction_2d(n_components_optimal: int, X_scaled: np.ndarray, umap_available: bool) -> np.ndarray:
    """
    This function reduces the PC space to 2D for visualization using UMAP if available, otherwise PCA.

    Parameters:
    n_components_optimal (int): Optimal number of components to retain.
    X_scaled (np.ndarray): Numpy array with scaled features.

    Returns:
    embedding(np.ndarray): Numpy array with 2D projection of the data.
    reducer_name(str): Name of the dimensionality reduction method used.
    """
    pca_optimal = PCA(n_components=n_components_optimal, random_state=42)
    X_pca = pca_optimal.fit_transform(X_scaled)

    if umap_available:
        # We apply UMAP to the PC space (not original features)
        # We chose the number of neighbors as the square roolt of the number of samples (roughly 35 for 1338 samples)
        reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=35, random_state=42)
        embedding = reducer.fit_transform(X_pca)  # We project the PC space to 2D
        reducer_name = "UMAP"
        print(f"\nUsing UMAP to project {X_pca.shape[1]}D PC space → 2D for visualization")
    else:
        # We use first 2 PCs for visualization if UMAP is unavailable
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)
        reducer_name = "PCA"
        print(f"\nUsing first 2 PCs for 2D visualization (UMAP not available)")

    print(f"  Note: 2D projection is for visualization only, clustering uses {X_pca.shape[1]}D PC space")

    return X_pca, embedding, reducer_name

# # X_pca, embedding, reducer_name = dimensionality_reduction_2d(n_components_optimal, X_scaled, umap_available)

# ====================
# KMEANS EVALUATION - Find best k
# ====================


def compute_optimal_k(X_pca: np.ndarray, features: pd.DataFrame) -> int:
    """
    This function evaluates KMeans clustering for multiple k values and computes silhouette scores and cluster balance.

    Parameters:
    X_pca (np.ndarray): Numpy array with data in PC space.
    features (pd.DataFrame): Dataframe with features.

    Returns:
    best_k(int): Optimal number of clusters based on silhouette score and cluster balance."""
    kmeans_results = []
    for k in [2, 3, 4, 5, 7, 10, 12, 15]:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_pca)  # We cluster on the PC space
        
        silhouette = silhouette_score(X_pca, labels) if len(set(labels)) > 1 else -1
        
        # We check cluster balance
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
        
    # We select best k based on balance AND the silhouette
    # Note: we prioretize balanced clusters (ratio > 0.1) over perfect silhouette
    print("\nSelecting best KMeans configuration...")
    valid_results = [r for r in kmeans_results if r['balance_ratio'] > 0.05 and r['min_size'] >= 0.05 * len(features)]

    if valid_results:
        # Among balanced options, we pick the highest silhouette
        best_kmeans = max(valid_results, key=lambda x: x['silhouette'])
    else:
        # We use a relaxed threshold: at least 2% of data is in smallest cluster
        relaxed_results = [r for r in kmeans_results if r['min_size'] >= 0.02 * len(features)]
        if relaxed_results:
            best_kmeans = max(relaxed_results, key=lambda x: x['silhouette'])
        else:
            # We pick the best balance
            best_kmeans = max(kmeans_results, key=lambda x: x['balance_ratio'])

    best_k = best_kmeans['k']
    print(f"  Selected: k={best_k} (silhouette={best_kmeans['silhouette']:.3f}, "
        f"balance={best_kmeans['balance_ratio']:.3f})")
    
    # We use KMeans clustering results
    cluster_labels = best_kmeans['labels']
    cluster_method = f"KMeans (k={best_kmeans['k']}, silhouette={best_kmeans['silhouette']:.3f})"
    best_k = best_kmeans['k']
    print(f"\nUsing KMeans clustering on {X_pca.shape[1]}D PC space")

    return cluster_labels, cluster_method, best_k, best_kmeans

# # cluster_labels, cluster_method, best_k, best_kmeans = compute_optimal_k(X_pca, features)


# ====================
# CLUSTER STABILITY VALIDATION (bootstrap)
# ====================

def cluster_validation_bootstrap(best_k: int, X_pca: np.ndarray, cluster_labels: np.ndarray, n_bootstrap: int = 100):
    """
    This function performs bootstrap validation of clustering stability using Adjusted Rand Index (ARI).
    
    Parameters:
    best_k (int): Optimal number of clusters.
    X_pca (np.ndarray): PC-transformed feature matrix (reduced dimensionality).
    cluster_labels (np.ndarray): Cluster assignments from original clustering.
    n_bootstrap (int): Number of bootstrap samples to use (default: 100).

    Returns:
    None
    """
    ari_scores = []
    for i in range(n_bootstrap):
        # We sample 80% of data
        sample_idx = np.random.choice(len(X_pca), size=int(0.8 * len(X_pca)), replace=False)
        X_sample = X_pca[sample_idx]  # Use PC space
        
        # We run KMeans on bootstrap sample
        km_boot = KMeans(n_clusters=best_k, n_init=20, random_state=i)
        labels_boot = km_boot.fit_predict(X_sample)
        
        # We compute ARI between bootstrap sample and full clustering
        ari = adjusted_rand_score(cluster_labels[sample_idx], labels_boot)
        ari_scores.append(ari)
    
    # We compute bootstrap statistics with confidence intervals
    ari_mean = np.mean(ari_scores)
    ari_std = np.std(ari_scores)
    ari_ci_lower = np.percentile(ari_scores, 2.5)
    ari_ci_upper = np.percentile(ari_scores, 97.5)

    print(f"Bootstrap ARI (mean ± std): {ari_mean:.3f} ± {ari_std:.3f}")
    print(f"  95% Confidence Interval: [{ari_ci_lower:.3f}, {ari_ci_upper:.3f}]")
    print(f"  Higher ARI (close to 1.0) = more stable clusters")
    print(f"  Interpretation: {'STABLE' if ari_mean > 0.7 else 'MODERATE' if ari_mean > 0.5 else 'UNSTABLE'}")
    

# # cluster_validation_bootstrap(best_k, X_pca, cluster_labels, n_bootstrap=100)


# ====================
# COMPREHENSIVE CLUSTER VALIDATION
# ====================


def validate_clusters_statistical(X_pca: np.ndarray, features: pd.DataFrame, cluster_labels: np.ndarray, feature_cols: list):
    """
    This function performs comprehensive statistical validation of clusters.
    
    Parameters:
    X_pca (np.ndarray): PC-transformed feature matrix (reduced dimensionality).
    features (pd.DataFrame): Original features dataframe.
    cluster_labels (np.ndarray): Cluster assignments.
    feature_cols (list): List of feature column names.
    
    Returns:
    dict: Dictionary containing validation metrics and test results.
    """
    from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
    from scipy.stats import f_oneway, kruskal
    
    results = {}
    
    # We compute additional clustering quality metrics on PC space
    # Davies-Bouldin Index assesses cluster quality by computing the average similarity measure of each 
    # cluster with the cluster most similar to it (the lower the value, the better the clustering)
    results['davies_bouldin'] = davies_bouldin_score(X_pca, cluster_labels)
    # Calinski-Harabasz Score (Variance Ratio Criterion) evaluates cluster dispersion by measuring the ratio of 
    # between-cluster variance to within-cluster variance (the higher the value, the better the clustering)
    results['calinski_harabasz'] = calinski_harabasz_score(X_pca, cluster_labels)
    
    print("\nClustering Quality Metrics:")
    print(f"  Davies-Bouldin Index: {results['davies_bouldin']:.3f} (lower is better, < 1.0 is good)")
    print(f"  Calinski-Harabasz Score: {results['calinski_harabasz']:.1f} (higher is better)")
    
    # We perform statistical tests for each feature across clusters
    print("\nStatistical Tests for Feature Discrimination Across Clusters:")
    print("  (Testing if clusters differ significantly on each feature)")
    
    significant_features = []
    feature_test_results = []
    
    for i, col in enumerate(feature_cols):
        # We group feature values by cluster
        groups = [features.iloc[:, i][cluster_labels == label].values 
                 for label in np.unique(cluster_labels) if label != -1]
        
        # We skip if any group is too small
        if any(len(g) < 3 for g in groups):
            continue
        
        # We perform Kruskal-Wallis test: a non-parametric statistical test used to compare three or more 
        # independent groups to see if their medians (or distributions) differ
        h_stat, p_value = kruskal(*groups)
        
        # We compute effect size (eta-squared approximation for Kruskal-Wallis): a higher value means that
        # a higher proportion of the variance in that feature is due to differences between clusters, the 
        # lower it is, the larger the proportion of variance is within clusters, not between them
        n_total = sum(len(g) for g in groups)
        eta_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))
        
        feature_test_results.append({
            'feature': col,
            'h_statistic': h_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05/len(feature_cols)  # We use a Bonferroni-adjusted threshold 0.05/#features (standard 0.05 
                                                        # threshold divided by #features as we perform 1 test per feature and we 
                                                        # have #features)
        })
        
        if p_value < 0.05/len(feature_cols):  # We use strict threshold due to multiple comparisons
            significant_features.append(col)
    
    # We sort by effect size and show top discriminative features
    feature_test_results.sort(key=lambda x: x['eta_squared'], reverse=True)
    
    print(f"\nTop 10 most discriminative features (by effect size):")
    for i, result in enumerate(feature_test_results[:10]):
        print(f"  {i+1:2d}. {result['feature']:30s}: η²={result['eta_squared']:.3f}, p={result['p_value']:.2e}")
    
    results['feature_tests'] = feature_test_results
    results['n_significant_features'] = len(significant_features)
    
    print(f"\nSignificant features (p < 0.0.05/len(feature_cols)): {len(significant_features)}/{len(feature_cols)}")
    if len(significant_features) < len(feature_cols) * 0.5:
        print(f"    WARNING: Less than 50% of features significantly discriminate clusters")
        print(f"    This suggests clusters may not be well-separated or features may be redundant")
    
    return results

# # validation_results = validate_clusters_statistical(X_pca, features, cluster_labels, feature_cols)

# ====================
# SILHOUETTE COMPARISON (PC space vs original space)
# ====================

def silhouette_comparison(X_pca: np.ndarray, X_scaled: np.ndarray, cluster_labels: np.ndarray, feature_cols: list):
    """
    This function compares silhouette scores in PC space vs original feature space.
    
    Parameters:
    X_pca (np.ndarray): Numpy array with data in PC space.
    X_scaled (np.ndarray): Numpy array with scaled original features.
    cluster_labels (np.ndarray): Cluster assignments.
    feature_cols (list): List of feature column names.
    
    Returns:
    None
    """
    silhouette_pca = silhouette_score(X_pca, cluster_labels) if len(set(cluster_labels)) > 1 else -1
    silhouette_original = silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else -1

    print(f"\nSilhouette Score Comparison:")
    print(f"  {X_pca.shape[1]}D PC space (USED FOR CLUSTERING): {silhouette_pca:.3f}")
    print(f"  Original {len(feature_cols)}D feature space: {silhouette_original:.3f}")
    print(f"\n  PC space ({X_pca.shape[1]}D) vs Original space ({len(feature_cols)}D): {'BETTER' if silhouette_pca > silhouette_original else 'WORSE'}")
    print(f"    Improvement: {(silhouette_pca - silhouette_original):.3f} ({((silhouette_pca - silhouette_original)/abs(silhouette_original)*100):.1f}%)")

    # Warninig raised for suspiciously high scores
    if silhouette_original > 0.9:
        print(f"\n  WARNING: Silhouette > 0.9 suggests outlier-driven clustering")
        print(f"    This often means a few extreme points are being separated,")
        print(f"    not meaningful pattern discovery in the main data.")

# ====================
# RESULTS ASSEMBLY
# ====================

def assemble_results_dataframe(features: pd.DataFrame, feature_cols: list,cluster_labels: np.ndarray, embedding: np.ndarray, temporal_features: pd.DataFrame, reducer_name: str, cluster_method: str, X_pca: np.ndarray) -> pd.DataFrame:
    """
    This function assembles the final results dataframe with cluster assignments and 2D embeddings.
    
    Parameters:
    features (pd.DataFrame): Dataframe with features.
    feature_cols (list): List of feature column names.
    cluster_labels (np.ndarray): Cluster assignments.
    embedding (np.ndarray): 2D embedding of the data.
    temporal_features (pd.DataFrame): Original temporal features dataframe.
    
    Returns:
    result_df(pd.DataFrame): Dataframe with clusters and embeddings.
    cluster_summary(pd.DataFrame): Dataframe summarizing cluster characteristics.
    """
    result_df = features.copy()
    result_df["cluster"] = cluster_labels
    result_df["embed_x"] = embedding[:, 0]
    result_df["embed_y"] = embedding[:, 1]

    # We add back original (non-transformed) features for interpretation
    raw_feature_cols = [
        "src_tox_count", "src_active_span_days", "src_median_gap_days", "src_gap_cv", 
        "src_max_30d_count", "src_event_rate", "src_pct_events_recent",
        "tgt_tox_count", "tgt_active_span_days", "tgt_median_gap_days", "tgt_gap_cv",
        "tgt_pct_events_recent",
        "net_in_degree", "net_out_degree", "net_toxicity", "total_tox_count"
    ]
    for col in raw_feature_cols:
        if col in temporal_features.columns:
            result_df[f"{col}_raw"] = temporal_features[col]
    
    # Add mega-subreddit flag if it exists
    if 'is_mega' in temporal_features.columns:
        result_df['is_mega'] = temporal_features['is_mega']

    print(f"\n" + "="*60)
    print(f"Reducer: {reducer_name} | Clustering: {cluster_method}")
    print(f"Clustering Space: {X_pca.shape[1]}D PC space (from {len(feature_cols)} original features)")
    print("="*60)
    print("\nCluster sizes:")
    cluster_size_df = result_df.groupby("cluster").size().rename("n_subreddits").sort_values(ascending=False)
    print(cluster_size_df)

    # We compute the balance of clusters
    if len(cluster_size_df) > 1:
        max_cluster_size = cluster_size_df.max()
        min_cluster_size = cluster_size_df.min()
        imbalance_ratio = max_cluster_size / min_cluster_size
        
        print(f"\nCluster Balance Analysis:")
        print(f"  Largest cluster: {max_cluster_size} ({max_cluster_size/len(result_df)*100:.1f}%)")
        print(f"  Smallest cluster: {min_cluster_size} ({min_cluster_size/len(result_df)*100:.1f}%)")
        print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
        
        if max_cluster_size / len(result_df) > 0.9:
            print(f"\n  WARNING: One cluster contains >80% of data!")

    # Cluster interpretation
    print("\n" + "="*60)
    print("CLUSTER CHARACTERISTICS (median values)")
    print("="*60)
    interpretation_cols = [
        "total_tox_count_raw",
        "src_tox_count_raw", 
        "tgt_tox_count_raw",
        "net_toxicity_raw",
        "net_in_degree_raw",
        "net_out_degree_raw",
        "src_event_rate",
        "src_pct_events_recent",
        "tgt_pct_events_recent"
    ]
    # Filter to only existing columns
    interpretation_cols = [col for col in interpretation_cols if col in result_df.columns]
    cluster_summary = result_df.groupby("cluster")[interpretation_cols].median()
    print(cluster_summary)

    return result_df, cluster_summary

# # result_df, cluster_summary = assemble_results_dataframe(features, feature_cols, cluster_labels, embedding, temporal_features, reducer_name, cluster_method, X_pca)

# ====================
# TEMPORAL TOXICITY PATTERN CLASSIFICATION
# ====================


def classify_temporal_patterns(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function classifies subreddits into temporal toxicity patterns: chronic, bursty, or occasional.
    Uses role-aware classification: aggressors use src_* features, victims use tgt_* features.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with cluster assignments and raw features.
    
    Returns:
    pd.DataFrame: Result dataframe with added temporal_pattern column.
    """
    # We define temporal patterns using data-driven thresholds (tertile splits)
    # Chronic: high count + low gap_cv (consistent, sustained toxicity)
    # Bursty: high max_30d + high gap_cv (intense bursts with irregular gaps)
    # Occasional: low count + high median gap (infrequent toxicity)
    
    patterns = []
    roles = []
    
    for idx, row in result_df.iterrows():
        src_count = row.get('src_tox_count_raw', 0)
        tgt_count = row.get('tgt_tox_count_raw', 0)
        
        # We determine the dominant role for this subreddit
        # This determines which feature set to use for temporal pattern classification
        if src_count > tgt_count:
            role = 'aggressor'
            # Use source (aggressor) features
            count = src_count
            gap_cv = row.get('src_gap_cv', 0)
            max_30d = row.get('src_max_30d_count_raw', 0)
            median_gap_days = row.get('src_median_gap_days_raw', 0)
            
            # Compute tertile thresholds based on source features
            count_high = count > result_df['src_tox_count_raw'].quantile(0.67)
            count_low = count < result_df['src_tox_count_raw'].quantile(0.33)
            gap_cv_low = gap_cv < result_df['src_gap_cv'].quantile(0.33)
            gap_cv_high = gap_cv > result_df['src_gap_cv'].quantile(0.67)
            median_gap_high = median_gap_days > result_df['src_median_gap_days_raw'].quantile(0.67)
            burst_high = max_30d > result_df['src_max_30d_count_raw'].quantile(0.67)
        else:
            role = 'victim'
            # Use target (victim) features
            count = tgt_count
            gap_cv = row.get('tgt_gap_cv', 0)
            # Note: tgt_max_30d doesn't exist, so we use a proxy based on count distribution
            max_30d = 0  # Victims don't have max_30d feature, use count for burst detection
            median_gap_days = row.get('tgt_median_gap_days_raw', 0)
            
            # Compute tertile thresholds based on target features
            count_high = count > result_df['tgt_tox_count_raw'].quantile(0.67)
            count_low = count < result_df['tgt_tox_count_raw'].quantile(0.33)
            gap_cv_low = gap_cv < result_df['tgt_gap_cv'].quantile(0.33)
            gap_cv_high = gap_cv > result_df['tgt_gap_cv'].quantile(0.67)
            median_gap_high = median_gap_days > result_df['tgt_median_gap_days_raw'].quantile(0.67)
            # For victims, use high count + high cv as proxy for burst (coordinated attacks)
            burst_high = count_high and gap_cv_high
        
        # We classify based on temporal patterns
        if count_high and gap_cv_low:
            pattern = "Chronic"
        elif burst_high and gap_cv_high:
            pattern = "Bursty"
        elif count_low and median_gap_high:
            pattern = "Occasional"
        else:
            pattern = "Mixed"
        
        patterns.append(pattern)
        roles.append(role)
    
    result_df['temporal_pattern'] = patterns
    result_df['dominant_role'] = roles
    
    # We report pattern distribution
    print("\nTemporal Pattern Distribution (Role-Aware):")
    pattern_counts = result_df['temporal_pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        pct = count / len(result_df) * 100
        print(f"  {pattern:12s}: {count:4d} ({pct:5.1f}%)")
    
    # We also report role distribution
    print("\nDominant Role Distribution:")
    role_counts = result_df['dominant_role'].value_counts()
    for role, count in role_counts.items():
        pct = count / len(result_df) * 100
        print(f"  {role:12s}: {count:4d} ({pct:5.1f}%)")

    # We analyze temporal patterns by cluster
    print("\nTemporal Pattern Distribution by Cluster:")
    # We create a frequency table where each row is a cluster, each column is a temporal pattern and the each value is 
    # the percentage of subreddits in that cluster with that temporal pattern
    pattern_by_cluster = pd.crosstab(result_df['cluster'], result_df['temporal_pattern'], normalize='index') * 100
    print(pattern_by_cluster.round(1))

    # We analyze dominant role distribution by cluster
    print("\nDominant Role Distribution by Cluster:")
    role_by_cluster = pd.crosstab(result_df['cluster'], result_df['dominant_role'], normalize='index') * 100
    print(role_by_cluster.round(1))

    return result_df

# # result_df = classify_temporal_patterns(result_df)

# ====================
# TEMPORAL PATTERN DEEP DIVE ANALYSIS (ROLE-AWARE)
# ====================


def pattern_characteristics_analysis(result_df: pd.DataFrame):
    """
    This function analyzes characteristics of temporal patterns split by dominant role.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with temporal patterns and dominant roles.
    """
    
    # We analyze what differentiates temporal patterns, split by role
    print("\nAGGRESSOR Patterns (using src_* features):")
    aggressor_subset = result_df[result_df['dominant_role'] == 'aggressor']
    if len(aggressor_subset) > 0:
        aggressor_analysis = aggressor_subset.groupby('temporal_pattern').agg({
            'src_tox_count_raw': ['median', 'mean', 'count'],
            'src_gap_cv': ['median', 'mean'],
            'src_max_30d_count_raw': ['median', 'mean'],
            'src_median_gap_days_raw': ['median', 'mean']
        }).round(2)
        print(aggressor_analysis)

    print("\nVICTIM Patterns (using tgt_* features):")
    victim_subset = result_df[result_df['dominant_role'] == 'victim']
    if len(victim_subset) > 0:
        victim_analysis = victim_subset.groupby('temporal_pattern').agg({
            'tgt_tox_count_raw': ['median', 'mean', 'count'],
            'tgt_gap_cv': ['median', 'mean'],
            'tgt_median_gap_days_raw': ['median', 'mean']
        }).round(2)
        print(victim_analysis)

    # We calculate the tertile thresholds used in classification
    print("\n\nClassification Thresholds (Tertiles) - Aggressor Features:")
    src_count_tertiles = result_df['src_tox_count_raw'].quantile([0.33, 0.67])
    src_cv_tertiles = result_df['src_gap_cv'].quantile([0.33, 0.67])
    src_burst_tertile = result_df['src_max_30d_count_raw'].quantile(0.67)
    print(f"  Source Count: Low < {src_count_tertiles[0.33]:.0f} | High > {src_count_tertiles[0.67]:.0f}")
    print(f"  Source Gap CV: Low < {src_cv_tertiles[0.33]:.2f} | High > {src_cv_tertiles[0.67]:.2f}")
    print(f"  Source Max 30d: Burst Threshold > {src_burst_tertile:.0f}")

    print("\nClassification Thresholds (Tertiles) - Victim Features:")
    tgt_count_tertiles = result_df['tgt_tox_count_raw'].quantile([0.33, 0.67])
    tgt_cv_tertiles = result_df['tgt_gap_cv'].quantile([0.33, 0.67])
    print(f"  Target Count: Low < {tgt_count_tertiles[0.33]:.0f} | High > {tgt_count_tertiles[0.67]:.0f}")
    print(f"  Target Gap CV: Low < {tgt_cv_tertiles[0.33]:.2f} | High > {tgt_cv_tertiles[0.67]:.2f}")

    print("\n" + "="*60)
    print("KEY INSIGHT: Role-Aware Temporal Pattern Interpretation")
    print("="*60)
    print("\nChronic: Subreddits with regular, sustained toxic activity")
    print("  AGGRESSORS: Consistent attackers with low gap variability")
    print("  VICTIMS: Constantly targeted with regular incoming attacks")
    print("\nBursty: Coordinated campaigns with concentrated spikes")
    print("  AGGRESSORS: Organized brigading with intense bursts + irregular timing")
    print("  VICTIMS: Coordinated attacks from multiple sources (pile-ons)")
    print("\nOccasional: Low-activity subreddits with infrequent events")
    print("  AGGRESSORS: Rare attackers, only engage sporadically")
    print("  VICTIMS: Rarely targeted, peripheral to conflicts")
    print("\nMixed: The 'normal' irregular pattern (catch-all)")
    print("  Moderate activity without clear burst/chronic signature")
    print("  Ongoing but unpredictable conflicts (event-driven, not sustained)")
    print("\nCRITICAL FIX: Patterns now reflect ROLE-SPECIFIC behavior.")
    print("  A frequently-attacked victim is no longer misclassified as 'Occasional'")
    print("  due to low aggressor activity. Each subreddit is classified based on")
    print("  their dominant role's temporal dynamics.")
    print("="*60)


def cluster_role_interpretation(result_df: pd.DataFrame, cluster_summary: pd.DataFrame):
    """
    This function interprets clusters based on activity levels, roles, and dominant temporal patterns.
    Includes identification of mega-subreddits within clusters.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with cluster assignments and temporal patterns.
    cluster_summary (pd.DataFrame): Dataframe summarizing cluster characteristics.
    
    Returns:
    result_df (pd.DataFrame): Dataframe with added cluster_interpretation column.
    """
    # We compute activity level thresholds based on all subreddits
    total_count_tertiles = result_df['total_tox_count_raw'].quantile([0.33, 0.67])
    
    # Create cluster interpretation mapping for hover labels
    cluster_interpretations = {}
    
    # Report mega-subreddit distribution across clusters
    if 'is_mega' in result_df.columns:
        mega_count = result_df['is_mega'].sum()
        if mega_count > 0:
            print(f"\nMega-Subreddit Distribution Across Clusters:")
            mega_by_cluster = result_df[result_df['is_mega']].groupby('cluster').size()
            for cluster_id, count in mega_by_cluster.items():
                pct = 100 * count / len(result_df[result_df['cluster'] == cluster_id])
                print(f"  Cluster {cluster_id}: {count} mega-subreddits ({pct:.1f}% of cluster)")

    for cluster_id in cluster_summary.index:
        cluster_data = cluster_summary.loc[cluster_id]
        src_count = cluster_data.get("src_tox_count_raw", 0)
        tgt_count = cluster_data.get("tgt_tox_count_raw", 0)
        total_count = cluster_data.get("total_tox_count_raw", 0)
        net_tox = cluster_data.get("net_toxicity_raw", 0)
        in_deg = cluster_data.get("net_in_degree_raw", 0)
        out_deg = cluster_data.get("net_out_degree_raw", 0)
        
        # We get dominant temporal pattern for this cluster
        cluster_patterns = result_df[result_df['cluster'] == cluster_id]['temporal_pattern'].value_counts()
        dominant_pattern = cluster_patterns.index[0] if len(cluster_patterns) > 0 else "Unknown"
        pattern_pct = cluster_patterns.iloc[0] / cluster_patterns.sum() * 100 if len(cluster_patterns) > 0 else 0
        
        # We classify activity level based on tertile thresholds
        if total_count > total_count_tertiles[0.67]:
            activity = "High-volume"
        elif total_count > total_count_tertiles[0.33]:
            activity = "Moderate-volume"
        else:
            activity = "Low-volume"
        
        # We classify roles using relative thresholds (robust across different magnitudes)
        # Balanced: difference < 20% of total count
        # Strong role: difference >= 50% of total count
        # Leaning: difference >= 20% but < 50% of total count
        total_for_role = src_count + tgt_count
        if total_for_role > 0:
            diff_ratio = abs(src_count - tgt_count) / total_for_role
            
            if diff_ratio < 0.20:
                role = "Balanced (Hub)"
            elif diff_ratio >= 0.50:
                role = "Aggressor" if src_count > tgt_count else "Victim"
            else:  # 0.20 <= diff_ratio < 0.50
                role = "Aggressor-leaning" if src_count > tgt_count else "Victim-leaning"
        else:
            role = "Inactive"
        
        label = f"{activity} {role} - {dominant_pattern}"
        cluster_interpretations[cluster_id] = label
        
        print(f"Cluster {cluster_id}: {label}")
        print(f"  Median: {total_count:.0f} events (src={src_count:.0f}, tgt={tgt_count:.0f}), net={net_tox:.0f}, in_deg={in_deg:.0f}, out_deg={out_deg:.0f}")
        print(f"  Dominant pattern: {dominant_pattern} ({pattern_pct:.0f}% of cluster)")

    # Add cluster interpretation to result_df for hover labels
    result_df['cluster_interpretation'] = result_df['cluster'].map(cluster_interpretations)

    return result_df

# # result_df = cluster_role_interpretation(result_df, cluster_summary)

# ====================
# VISUALIZATION
# ====================

def plot_cluster_results(result_df: pd.DataFrame, reducer_name: str, feature_cols: list, cluster_labels: np.ndarray, X_scaled: np.ndarray, n_components_optimal: int):
    """
    This function creates visualizations of clustering results using Matplotlib and Seaborn.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with cluster assignments, embeddings, and features.
    reducer_name (str): Name of the dimensionality reduction method used (e.g., "UMAP", "PCA").
    feature_cols (list): List of feature column names.
    cluster_labels (np.ndarray): Cluster assignments.
    X_scaled (np.ndarray): Scaled original features.
    n_components_optimal (int): Number of principal components used for clustering.
    
    Returns:
    None
    """
    plt.clf()
    plt.figure(figsize=(16, 5))

    # Plot 1: Embedding colored by cluster
    plt.subplot(1, 3, 1)
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

    # Plot 2: Embedding colored by temporal pattern
    plt.subplot(1, 3, 2)
    pattern_colors = {'Chronic': 'red', 'Bursty': 'orange', 'Mixed': 'gray', 'Occasional': 'blue'}
    sns.scatterplot(
        x="embed_x", y="embed_y", hue="temporal_pattern", data=result_df,
        palette=pattern_colors, s=50, alpha=0.85, edgecolor="white", linewidth=0.6
    )
    plt.title(f"Subreddit temporal patterns\n({reducer_name} projection)")
    plt.xlabel(f"{reducer_name} 1")
    plt.ylabel(f"{reducer_name} 2")
    plt.legend(title="temporal pattern", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.2)

    # Plot 3: Feature importance (PCA variance)
    plt.subplot(1, 3, 3)
    pca_full = PCA(n_components=min(len(feature_cols), len(X_scaled)), random_state=42)
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_ * 100
    num_pcs = len(explained_var)
    bar_positions = np.arange(1, num_pcs + 1)
    bars = plt.bar(bar_positions, explained_var)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.title(f"PCA Variance Explained\n(Total: {pca_full.explained_variance_ratio_.sum()*100:.1f}%)")
    plt.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    plt.show()

# # plot_cluster_results(result_df, reducer_name, feature_cols, cluster_labels, X_scaled, n_components_optimal)

def plot_variance_explained(X_scaled: np.ndarray, feature_cols: list, n_components_optimal: int):
    """
    This function plots the variance explained by each principal component and identifies top features.
    Parameters:
    pca_full (PCA): Fitted PCA object on the full feature set.
    feature_cols (list): List of feature column names.
    n_components_optimal (int): Number of principal components used for clustering.

    Returns:
    None
    """
    pca_full = PCA(n_components=min(len(feature_cols), len(X_scaled)), random_state=42)
    pca_full.fit(X_scaled)
    explained_var = pca_full.explained_variance_ratio_ * 100

    # Print variance explained by each PC and its top feature in the output cell only
    print("\nVariance explained by each principal component and its top feature:")
    print(f"(First {n_components_optimal} PCs were retained for clustering, explaining {np.sum(pca_full.explained_variance_ratio_[:n_components_optimal])*100:.1f}% variance)")
    for i, (var, comp) in enumerate(zip(explained_var, pca_full.components_)):
        top_idx = np.argmax(np.abs(comp))
        top_feature = feature_cols[top_idx]
        retained_marker = " ← RETAINED" if i < n_components_optimal else " (discarded)"
        print(f"PC{i+1}: {var:.2f}% variance, top feature: {top_feature}{retained_marker}")

# # plot_variance_explained(X_scaled, feature_cols, n_components_optimal)


# ====================
# INTERACTIVE PLOTLY VISUALIZATION
# ====================


def create_interactive_plotly_visualization(result_df: pd.DataFrame, reducer_name: str, 
                                           feature_cols: list, n_components_optimal: int, output_path: str = "subreddit_clustering_interactive.html"):
    """
    This function creates an interactive Plotly scatter plot with cluster visualization and hover information.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with cluster assignments, embeddings, and features.
    reducer_name (str): Name of the dimensionality reduction method used (e.g., "UMAP", "PCA").
    feature_cols (list): List of feature column names.
    output_path (str): Path to save the HTML file (default: "subreddit_clustering_interactive.html").
    
    Returns:
    fig: Plotly figure object.
    """
    # We define distinctive color palette for clusters (categorical)
    distinctive_colors = {
        0: '#e41a1c',  # Red
        1: '#377eb8',  # Blue
        2: '#4daf4a',  # Green
        3: '#ff7f00',  # Orange
        4: '#984ea3',  # Purple
        5: '#f781bf',  # Pink
        6: '#a65628',  # Brown
        7: '#999999'   # Gray
    }

    # We convert cluster to string for categorical coloring and add subreddit name as column
    result_df = result_df.copy()  # Create explicit copy to avoid modifying original
    result_df['cluster_str'] = result_df['cluster'].astype(str)
    result_df['subreddit_name'] = result_df.index

    # We create interactive scatter plot with hover information
    # Get sorted cluster order
    unique_clusters = sorted(result_df['cluster'].unique())
    cluster_order = [str(c) for c in unique_clusters]

    fig = px.scatter(
        result_df,
        x='embed_x',
        y='embed_y',
        color='cluster_str',
        color_discrete_map={str(k): v for k, v in distinctive_colors.items()},
        category_orders={'cluster_str': cluster_order},  # Force legend order
        custom_data=['subreddit_name', 'cluster', 'cluster_interpretation', 
                     'src_tox_count_raw', 'tgt_tox_count_raw', 'net_toxicity_raw'],
        labels={
            'embed_x': f'{reducer_name} 1',
            'embed_y': f'{reducer_name} 2',
            'cluster_str': 'Cluster'
        },
        title=f'Subreddits mapped by Toxicity Dynamics<br><sub>({reducer_name} projection, clustered in {n_components_optimal}D PC space)</sub>'
    )

    # We update hover template with custom formatting and set hover colors to match markers
    for trace in fig.data:
        trace.update(
            hovertemplate='<b>Name:</b> %{customdata[0]}<br>' +
                          '<b>Cluster:</b> %{customdata[1]}<br>' +
                          '<b>Cluster Description:</b> %{customdata[2]}<br>' +
                          '<b>Source:</b> %{customdata[3]:.0f} (# Sent Messages)<br>' +
                          '<b>Target:</b> %{customdata[4]:.0f} (# Messages Targeting this Subreddit)<br>' +
                          '<b>Net Toxicity:</b> %{customdata[5]:.0f} (Source - Target)<br>' +
                          '<extra></extra>',
            hoverlabel=dict(
                bgcolor=trace.marker.color,  # Match hover background to marker color
                font_size=13,
                font_family="Arial",
                font_color="white"  # White text for better contrast
            )
        )

    # We update layout for interactivity
    fig.update_layout(
        autosize=True,  # Make plot responsive to container size
        height=700,
        font=dict(size=12),
        dragmode='pan',  # Default drag behavior is pan
        # Enable scroll to zoom
        xaxis=dict(fixedrange=False),
        yaxis=dict(fixedrange=False),
        legend=dict(
            title="Cluster",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # We enable scroll zoom configuration
    config = {
        'scrollZoom': True,  # Enable scroll to zoom
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }

    fig.show(config=config)

    print("\n   Interactive plot created. Hover over points to see details.")
    print("  Each point shows: Subreddit name, Cluster interpretation, Temporal pattern, Event counts")
    print("  Controls: Scroll to zoom, drag to pan, double-click to reset view")

    # We export to HTML file
    fig.write_html(output_path, config=config)
    print(f"\n  Interactive plot exported to: {output_path}")
    print(f"  Ready for embedding in Jekyll website")
    

# # create_interactive_plotly_visualization(result_df, reducer_name, feature_cols)

# ====================
# MEGA-SUBREDDIT ANALYSIS (WITHIN CLUSTERS)
# ====================

def analyze_mega_subreddits_in_clusters(result_df: pd.DataFrame, n_top: int = 15):
    """
    This function analyzes mega-subreddits that are now included in clustering.
    
    Parameters:
    result_df (pd.DataFrame): Dataframe with cluster assignments and features.
    n_top (int): Number of top mega-subreddits to display (default: 15).
    """
    if 'is_mega' not in result_df.columns or result_df['is_mega'].sum() == 0:
        print("  No mega-subreddits identified.")
        return None
    
    mega_df = result_df[result_df['is_mega']].copy()
    print(f"\nAnalyzing {len(mega_df)} mega-subreddits included in clustering...")
    
    # We classify mega-subreddits by dominant role
    mega_df['mega_role'] = mega_df.apply(
        lambda row: 'Mega-Aggressor' if row['src_tox_count_raw'] > row['tgt_tox_count_raw'] * 1.5
        else 'Mega-Victim' if row['tgt_tox_count_raw'] > row['src_tox_count_raw'] * 1.5
        else 'Mega-Hub', axis=1
    )
    
    print(f"\nTop {n_top} Mega-Subreddits (now clustered):")
    display_cols = ['cluster', 'total_tox_count_raw', 'src_tox_count_raw', 'tgt_tox_count_raw', 
                    'net_toxicity_raw', 'mega_role', 'temporal_pattern']
    top_mega = mega_df.nlargest(n_top, 'total_tox_count_raw')[display_cols]
    print(top_mega)
    
    print("\nMega-Subreddit Role Distribution:")
    role_distribution = mega_df['mega_role'].value_counts()
    print(role_distribution)
    for role, count in role_distribution.items():
        pct = 100 * count / len(mega_df)
        print(f"  {role}: {count} ({pct:.1f}%)")
    
    print("\nMega-Subreddit Cluster Distribution:")
    cluster_distribution = mega_df['cluster'].value_counts().sort_index()
    print(cluster_distribution)
    for cluster_id, count in cluster_distribution.items():
        pct = 100 * count / len(mega_df)
        print(f"  Cluster {cluster_id}: {count} mega-subreddits ({pct:.1f}% of all mega-subreddits)")
    
    print("\nMega-Subreddit Summary Statistics:")
    stats = mega_df[['total_tox_count_raw', 'src_tox_count_raw', 'tgt_tox_count_raw', 
                      'net_in_degree_raw', 'net_out_degree_raw']].describe()
    print(stats)
    
    return mega_df

# # mega_summary = analyze_mega_subreddits_in_clusters(result_df, n_top=15)
