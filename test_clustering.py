# Subreddit toxicity clustering (feature build -> scale -> embed -> cluster)
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


# Helper: median gap in days between consecutive timestamps
def median_gap_days(ts: pd.Series) -> float:
    ts_sorted = ts.sort_values()
    if ts_sorted.shape[0] < 2:
        return np.nan
    deltas = ts_sorted.diff().dropna().dt.total_seconds() / 86400
    return deltas.median()


# Source-side aggregates
src_group = interactions_df.groupby("SOURCE_SUBREDDIT")
src_features = pd.DataFrame({
    "tox_mean_src": src_group["TOXICITY_SCORE"].mean(),
    "tox_std_src": src_group["TOXICITY_SCORE"].std(),
    "tox_count_src": src_group.apply(lambda df: (df["TOXICITY_SCORE"] >= TOXICITY_THRESHOLD).sum()),
    "msg_count_src": src_group.size(),
    "sentiment_mean_src": src_group["LINK_SENTIMENT"].mean(),
})
# Temporal cadence: median gap (days) and inverse frequency (events per 30 days)
src_features["median_gap_days_src"] = src_group["TIMESTAMP"].apply(median_gap_days)
src_features["median_gap_freq_src"] = 30 / src_features["median_gap_days_src"].replace(0, np.nan)
src_features["tox_rate_src"] = src_features["tox_count_src"] / src_features["msg_count_src"].replace(0, np.nan)

# Target-side aggregates
tgt_group = interactions_df.groupby("TARGET_SUBREDDIT")
tgt_features = pd.DataFrame({
    "tox_mean_tgt": tgt_group["TOXICITY_SCORE"].mean(),
    "tox_std_tgt": tgt_group["TOXICITY_SCORE"].std(),
    "tox_count_tgt": tgt_group.apply(lambda df: (df["TOXICITY_SCORE"] >= TOXICITY_THRESHOLD).sum()),
    "msg_count_tgt": tgt_group.size(),
    "sentiment_mean_tgt": tgt_group["LINK_SENTIMENT"].mean(),
})
# Temporal cadence: median gap (days) and inverse frequency (events per 30 days)
tgt_features["median_gap_days_tgt"] = tgt_group["TIMESTAMP"].apply(median_gap_days)
tgt_features["median_gap_freq_tgt"] = 30 / tgt_features["median_gap_days_tgt"].replace(0, np.nan)
tgt_features["tox_rate_tgt"] = tgt_features["tox_count_tgt"] / tgt_features["msg_count_tgt"].replace(0, np.nan)

# Combine source + target; fill missing with 0 when no activity
all_subs = src_features.join(tgt_features, how="outer").fillna(0)
all_subs["tox_balance"] = all_subs["tox_count_src"] - all_subs["tox_count_tgt"]
all_subs["msg_balance"] = all_subs["msg_count_src"] - all_subs["msg_count_tgt"]

# Drop constant columns
constant_cols = [c for c in all_subs.columns if all_subs[c].nunique() <= 1]
features = all_subs.drop(columns=constant_cols)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Embed
if umap_available:
    reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=20, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    reducer_name = "UMAP"
else:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    reducer_name = "PCA"

# Cluster
if hdbscan_available:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=5, metric="euclidean")
    cluster_labels = clusterer.fit_predict(embedding)
    cluster_method = "HDBSCAN"
else:
    best_k, best_score, best_labels = None, -1, None
    for k in range(3, min(15, len(features))):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(embedding)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = silhouette_score(embedding, labels) if len(set(labels)) > 1 else -1
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    cluster_labels = best_labels
    cluster_method = f"KMeans (k={best_k}, silhouette={best_score:.3f})"

# Assemble results
result_df = features.copy()
result_df["cluster"] = cluster_labels
result_df["embed_x"] = embedding[:, 0]
result_df["embed_y"] = embedding[:, 1]

print(f"Reducer: {reducer_name} | Clustering: {cluster_method}")
print(result_df.groupby("cluster").size().rename("n_subreddits"))

# Plot embedding colored by cluster
plt.figure(figsize=(9, 7))
palette = sns.color_palette("husl", n_colors=len(set(cluster_labels))) if cluster_labels is not None else None
sns.scatterplot(
    x="embed_x", y="embed_y", hue="cluster", data=result_df,
    palette=palette, s=50, alpha=0.85, edgecolor="white", linewidth=0.6
)
plt.title("Subreddit toxicity clustering (2D embedding)")
plt.xlabel(f"{reducer_name} 1")
plt.ylabel(f"{reducer_name} 2")
plt.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

result_df.head()