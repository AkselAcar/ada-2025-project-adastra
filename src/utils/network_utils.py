# Certain functions in this file were made with the help of AI.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import umap.umap_ as umap
import networkx as nx
import community as community_louvain

from src import *

def filter_active_subreddits(min_interactions):
    """
    Filter subreddits based on a minimum number of interactions.

    Parameters
    ----------
    min_interactions : int
        The minimum number of outgoing interactions required for a subreddit to be considered active.

    Returns
    -------
    active_subs : pandas.Index
        Index of subreddit names that have at least `min_interactions` outgoing interactions.
    filtered_embeddings : pandas.DataFrame
        Subset of `embeddings_subreddits_df` containing only the active subreddits.
    """
    # === Count subreddit activity ===
    activity_counts = interactions_df["SOURCE_SUBREDDIT"].value_counts()

    # Keep only subreddits with at least min_interactions interactions
    active_subs = activity_counts[activity_counts > min_interactions].index
    filtered_embeddings = embeddings_subreddits_df[embeddings_subreddits_df["SUBREDDIT"].isin(active_subs)]

    return active_subs, filtered_embeddings


def reduce_embeddings_2d(embeddings_df):
    """
    Reduce high-dimensional subreddit embeddings to 2D using UMAP.

    Parameters
    ----------
    embeddings_df : pandas.DataFrame
        DataFrame containing subreddit embeddings, with one row per subreddit and
        a column named 'SUBREDDIT'.

    Returns
    -------
    embeddings_df : pandas.DataFrame
        Input DataFrame augmented with two new columns:
        - 'x': UMAP component 1
        - 'y': UMAP component 2
    """
    embedding_cols = [c for c in embeddings_df.columns if c != 'SUBREDDIT']
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_df[embedding_cols])
    embeddings_df['x'] = embedding_2d[:, 0]
    embeddings_df['y'] = embedding_2d[:, 1]
    return embeddings_df


def animate_evolution(active_subs, filtered, freq):
    """
    Create an animated visualization showing the evolution of mean toxicity over time
    for active subreddits projected in 2D UMAP space.

    Parameters
    ----------
    active_subs : list or pandas.Index
        List of subreddit names considered active.
    filtered : pandas.DataFrame
        Embeddings DataFrame containing UMAP coordinates ('x', 'y') for subreddits.
    freq : str
        Frequency of time grouping (e.g., 'W' for weekly, 'M' for monthly).

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        Animation object displaying the temporal evolution of subreddit toxicity.
    """
    # --- Compute period mean outgoing toxicity score per subreddit ---
    period_sentiment = (
        interactions_df[interactions_df['SOURCE_SUBREDDIT'].isin(active_subs)]
        .groupby([pd.Grouper(key='TIMESTAMP', freq=freq), 'SOURCE_SUBREDDIT'])
        ["TOXICITY_SCORE"]
        .mean()
        .reset_index()
    )

    # --- Merge 2D coordinates into period sentiment ---
    period_embeddings = period_sentiment.merge(
        filtered[['SUBREDDIT', 'x', 'y']],
        left_on='SOURCE_SUBREDDIT', right_on='SUBREDDIT', how='left'
    )

    # --- Animate ---
    periods = sorted(period_embeddings['TIMESTAMP'].unique())
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Color limits ---
    vmin, vmax = -0.03, 0.06  # fixed for interpretability

    # Initial scatter
    data0 = period_embeddings[period_embeddings['TIMESTAMP'] == periods[0]]
    sc = ax.scatter(data0['x'], data0['y'], c=data0["TOXICITY_SCORE"], cmap='coolwarm', vmin=vmin, vmax=vmax, s=25, alpha=0.8)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    if freq == 'W':
        title = ax.set_title(f"Week: {periods[0].date()}")
    elif freq == 'M':
        title = ax.set_title(f"Month: {periods[0].date()}")
    else:
        title = ax.set_title(f"Period: {periods[0].date()}")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Mean Toxicity Score')

    def update(frame):
        data = period_embeddings[period_embeddings['TIMESTAMP'] == periods[frame]]
        sc.set_offsets(data[['x', 'y']].values)
        sc.set_array(data["TOXICITY_SCORE"].values)
        if freq == 'W':
            title.set_text(f"Week: {periods[frame].date()}")
        elif freq == 'M':
            title.set_text(f"Month: {periods[frame].date()}")
        else:
            title.set_text(f"Period: {periods[frame].date()}")
        return sc, title

    ani = animation.FuncAnimation(fig, update, frames=len(periods), blit=False, repeat=False)
    plt.close(fig)
    return ani


def louvain_communities(toxic_df):
    """
    Build a directed graph of toxic subreddit interactions and detect communities using
    the Louvain algorithm.

    Parameters
    ----------
    toxic_df : pandas.DataFrame
        DataFrame containing toxic interactions with at least columns:
        ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'].

    Returns
    -------
    community_df : pandas.DataFrame
        DataFrame mapping each subreddit to its detected community ID.
    edges : pandas.DataFrame
        DataFrame of aggregated toxic edges with weights representing interaction counts.
    """
    # Aggregate toxic interactions (number of toxic links between subreddits)
    edges = (
        toxic_df.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'])
        .size()
        .reset_index(name='weight')  # 'weight' = number of toxic interactions
    )

    # Build a directed graph
    G = nx.from_pandas_edgelist(
        edges,
        source='SOURCE_SUBREDDIT',
        target='TARGET_SUBREDDIT',
        edge_attr='weight',
        create_using=nx.DiGraph()
    )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} toxic edges")

    # --- Detect communities using Louvain on the undirected version ---
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected, weight='weight', resolution=0.5)  # use weights to prioritize toxic links

    community_df = pd.DataFrame(list(partition.items()), columns=['SUBREDDIT', 'COMMUNITY_ID'])
    print(f"Detected {community_df['COMMUNITY_ID'].nunique()} communities.")
    return community_df, edges


def get_toxic_communities(community_df, toxicity_threshold):
    """
    Compute toxicity levels per community and flag highly toxic ones.

    Parameters
    ----------
    community_df : pandas.DataFrame
        DataFrame mapping subreddits to community IDs (from `louvain_communities`).
    toxicity_threshold : float
        Minimum fraction of toxic interactions required for a community to be flagged as toxic.

    Returns
    -------
    community_df : pandas.DataFrame
        Input DataFrame with an additional boolean column 'IS_HIGHLY_TOXIC'
        indicating which communities exceed the toxicity threshold.
    """
    # Compute toxicity per community ---
    # Merge community ID with interactions
    interactions_with_comm = interactions_df.merge(
        community_df, left_on='SOURCE_SUBREDDIT', right_on='SUBREDDIT', how='left'
    )

    # Compute fraction of toxic interactions per community
    community_toxicity = (
        interactions_with_comm.groupby('COMMUNITY_ID')['TOXICITY_CAT']
        .apply(lambda x: (x==1).sum() / max(1, len(x)))
        .reset_index()
        .rename(columns={'TOXICITY_CAT': 'TOXIC_FRACTION'})
    )
    # --- Flag highly toxic communities ---
    highly_toxic = community_toxicity[community_toxicity['TOXIC_FRACTION'] > toxicity_threshold]
    
    # --- Add a flag to community_df ---
    community_df['IS_HIGHLY_TOXIC'] = community_df['COMMUNITY_ID'].isin(highly_toxic['COMMUNITY_ID'])
    print(f"Detected {len(highly_toxic)} toxic communities.")
    return community_df


def plot_biggest_toxic_community_network(community_df, edges):
    """
    Plot a weighted directed network graph of the largest highly toxic community.

    Parameters
    ----------
    community_df : pandas.DataFrame
        DataFrame containing subreddit-community assignments and toxicity flags.
    edges : pandas.DataFrame
        DataFrame of directed edges with weights representing toxic interaction counts.

    Returns
    -------
    None
        Displays a matplotlib figure of the community network, where:
        - Node size is constant.
        - Edge width represents the number of toxic interactions (weight).
    """
    # --- Compute community sizes in terms of subreddits ---
    toxic_sizes = (
        community_df[community_df['IS_HIGHLY_TOXIC']]
        .groupby('COMMUNITY_ID')['SUBREDDIT']
        .nunique()  # count unique subreddits
        .reset_index(name='SIZE')
    )

    print(toxic_sizes)

    # --- Select the biggest highly toxic community ---
    biggest_toxic_comm_id = toxic_sizes.sort_values('SIZE', ascending=False)['COMMUNITY_ID'].iloc[0]
    print(f"Biggest highly toxic community ID: {biggest_toxic_comm_id}")

    # Subreddits in this community
    subreddits_in_comm = community_df[
        (community_df['COMMUNITY_ID'] == biggest_toxic_comm_id) &
        (community_df['IS_HIGHLY_TOXIC'])
    ]['SUBREDDIT'].tolist()
    print(f"Number of subreddits in this community: {len(subreddits_in_comm)}")

    # Filter toxic edges to only those within this community
    community_edges = edges[
        edges['SOURCE_SUBREDDIT'].isin(subreddits_in_comm) &
        edges['TARGET_SUBREDDIT'].isin(subreddits_in_comm)
    ]

    # --- Build network graph for the community ---
    G_comm = nx.from_pandas_edgelist(
        community_edges,
        source='SOURCE_SUBREDDIT',
        target='TARGET_SUBREDDIT',
        edge_attr='weight',
        create_using=nx.DiGraph()
    )

    # --- Plot the network ---
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G_comm, seed=42)  # layout

    # Nodes
    nx.draw_networkx_nodes(G_comm, pos, node_size=500, node_color='lightblue')

    # Edges weighted by number of toxic interactions
    edges_list = G_comm.edges(data=True)
    nx.draw_networkx_edges(
        G_comm, pos,
        edgelist=[(u, v) for u, v, d in edges_list],
        width=[d['weight']*0.5 for u, v, d in edges_list],
        alpha=0.7,
        arrowstyle='-|>', arrowsize=12
    )

    # Labels
    nx.draw_networkx_labels(G_comm, pos, font_size=10, font_weight='bold')

    plt.title(f"Weighted Network of the Biggest Highly Toxic Community ({biggest_toxic_comm_id})")
    plt.axis('off')
    plt.show()