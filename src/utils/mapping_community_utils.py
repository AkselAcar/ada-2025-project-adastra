# Certain functions in this file were made with the help of AI.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import umap.umap_ as umap
import networkx as nx
import community as community_louvain
import plotly.express as px
import plotly.colors as pc
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from scipy import linalg

from src import *

def filter_active_subreddits(min_interactions):
    """
    Filter subreddits from the total embeddings file based on a minimum number of interactions.

    Parameters
    ----------
    min_interactions : int
        The minimum number of total interactions (outgoing + incoming) required for a subreddit to be considered active.

    Returns
    -------
    active_subs : pandas.Index
        Index of subreddit names that have at least `min_interactions` total interactions.
    filtered_embeddings : pandas.DataFrame
        Subset of `embeddings_subreddits_df` containing only the active subreddits.
    """
    # === Count subreddit activity ===
    source_counts = interactions_df["SOURCE_SUBREDDIT"].value_counts()
    target_counts = interactions_df["TARGET_SUBREDDIT"].value_counts()
    total_counts = source_counts + target_counts
    embeddings_subreddits_df["interaction_count"] = embeddings_subreddits_df["SUBREDDIT"].map(total_counts).fillna(0).astype(int)

    # Keep only subreddits with at least min_interactions interactions
    active_subs = total_counts[total_counts >= min_interactions].index
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
    embedding_cols = [c for c in embeddings_df.columns if c != 'SUBREDDIT' and c != 'interaction_count']
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding_reduced = embeddings_df.copy()
    embedding_2d = reducer.fit_transform(embeddings_df[embedding_cols])
    embedding_reduced['x'] = embedding_2d[:, 0]
    embedding_reduced['y'] = embedding_2d[:, 1]
    embedding_reduced.drop(columns=embedding_cols, inplace=True)
    return embedding_reduced

def animate_toxicity(active_subs, filtered_2d, freq='ME'):
    """
    Create an animated visualization showing the evolution of mean toxicity over time
    for active subreddits projected in 2D UMAP space.

    Parameters
    ----------
    active_subs : list or pandas.Index
        List of subreddit names considered active.
    filtered_2d : pandas.DataFrame
        Embeddings DataFrame containing UMAP coordinates ('x', 'y') for subreddits.
    freq : str
        Frequency of time grouping (e.g., 'W' for weekly, 'ME' for monthly).

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
        filtered_2d[['SUBREDDIT', 'x', 'y']],
        left_on='SOURCE_SUBREDDIT', right_on='SUBREDDIT', how='left'
    )

    # --- Animate ---
    periods = sorted(period_embeddings['TIMESTAMP'].unique())
    fig, ax = plt.subplots(figsize=(8.8, 6))

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

def plot_biggest_toxic_community_network(community_df, edges, write=False):
    """
    Plot a weighted directed network graph of the largest toxic community.

    Parameters
    ----------
    community_df : pandas.DataFrame
        DataFrame containing subreddit-community assignments and toxicity flags.
    edges : pandas.DataFrame
        DataFrame of directed edges with weights representing toxic interaction counts.
    write : bool, optional
        Whether to save the list of subreddits in the biggest toxic community to a CSV file.

    Returns
    -------
    subreddits_in_comm : list
        List of subreddits in the biggest toxic community.
    displays a network plot of the biggest toxic community.
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

    if write:
        # Save the names of subreddits in network for use in case study
        path="data/CS_networks/network_" + role_of_interest + "_cluster" + str(cluster_of_interest)
        pd.DataFrame(biggest_toxic_community, columns=['subreddit']).to_csv(path)
    return subreddits_in_comm

def subreddit_selector(embeddings_2d):
    """
    Create an interactive app to select subreddits from a 2D embedding scatter plot
    to later assign them to a topic by exporting to CSV.
    """
    # Create map from embeddings 
    fig = px.scatter(
        embeddings_2d,
        x="x", 
        y="y",
        color_continuous_scale="RdBu_r",
        hover_data={
            "SUBREDDIT": True, 
            "interaction_count": True,
            "x": False, 
            "y": False
        },
        title="Subreddits mapped by embeddings",
        labels={
            "x": "UMAP Dimension 1", 
            "y": "UMAP Dimension 2",
            "interaction_count": "Interactions"
        }
    )

    for trace in fig.data:
        trace.update(
            hovertemplate='<b>Name:</b> %{customdata[0]}<br>' +
                        '<b>Interactions:</b> %{customdata[1]}<br>' +
                        '<extra></extra>',
            hoverlabel=dict(
                bgcolor=trace.marker.color,
                font_size=13,
                font_family="Arial",
                font_color="white"
            )
        )

    fig.update_traces(opacity=0.7)
    fig.update_layout(
        autosize=True,
        xaxis=dict(showgrid=True, gridwidth=0.5, range=[-13, 1]),
        yaxis=dict(showgrid=True, gridwidth=0.5, range=[-3, 8]),
        width=800,
        height=600,
        margin=dict(l=0, r=20, t=50, b=0),
        dragmode='lasso'  # Enable selection
    )

    # Create Dash app
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='scatter', figure=fig),
        html.Button('Export Selected to CSV', id='export-btn', n_clicks=0, 
                    style={'margin': '10px', 'padding': '10px', 'font-size': '16px'}),
        html.Div(id='output', style={'margin': '10px', 'font-size': '14px'})
    ])

    @app.callback(
        Output('output', 'children'),
        Input('export-btn', 'n_clicks'),
        State('scatter', 'selectedData')
    )
    def export_selection(n_clicks, selectedData):
        if n_clicks > 0 and selectedData:
            # Extract subreddit names
            selected_subreddits = [point['customdata'][0] for point in selectedData['points']]
            
            # Create DataFrame and export
            df_selected = pd.DataFrame({'SUBREDDIT': selected_subreddits})
            df_selected.to_csv('selected_subreddits.csv', index=False)
            
            return f"✓ Exported {len(selected_subreddits)} subreddits to selected_subreddits.csv"
        return "Select points with lasso/box tool, then click Export"

    # Run the app when called
    app.run(debug=True, port=8050)

def assign_topic(embeddings_df, csv_file, topic_name):
    """
    Assigns a topic to subreddits listed in a CSV file.
    """
    # Read the CSV file
    subreddits_to_label = pd.read_csv(csv_file)
    
    # Assume the first column contains subreddit names
    subreddit_list = subreddits_to_label.iloc[:, 0].tolist()
    
    embeddings_topic = embeddings_df.copy()

    # Create topic column if it doesn't exist
    if 'topic' not in embeddings_topic.columns:
        embeddings_topic['topic'] = None
    
    # Assign topic to matching subreddits
    embeddings_topic.loc[embeddings_topic['SUBREDDIT'].isin(subreddit_list), 'topic'] = topic_name

    return embeddings_topic

def load_topics(embeddings_df):
    """
    Load and assign topics to subreddits in the embeddings dataframe based on predefined CSV files (computed
    with the subreddit_selector function).
    """
    embeddings_with_topics = assign_topic(embeddings_df, './data/topic_assignment/controversial_subreddits.csv', 'Controversial Subjects')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/gaming_subreddits.csv', 'Gaming')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/japan_subreddits.csv', 'Japan')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/football_subreddits.csv', 'Football')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/education_subreddits.csv', 'Education')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/crypto_subreddits.csv', 'Cryptocurrency')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/movies_subreddits.csv', 'Movies')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/us_cities_subreddits.csv', 'US Cities/States')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/us_sports_subreddits.csv', 'US Sports')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/technology_subreddits.csv', 'Technology')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/pornography_subreddits.csv', 'Pornography')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/countries_subreddits.csv', 'Countries')
    embeddings_with_topics = assign_topic(embeddings_with_topics, './data/topic_assignment/music_subreddits.csv', 'Music')
    embeddings_with_topics["topic"] = embeddings_with_topics["topic"].fillna("Popular/Memes")
    return embeddings_with_topics

def fit_ellipse(points, scale=2):
    """
    Fit an ellipse to a set of points.
    scale controls how large the ellipse is (in standard deviations)
    """
    # Calculate center
    center = points.mean(axis=0)
    
    # Calculate covariance matrix
    cov = np.cov(points.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.eigh(cov)
    
    # Sort by eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Calculate ellipse parameters
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Scale by eigenvalues (standard deviations)
    ellipse = np.array([
        scale * np.sqrt(eigenvalues[0]) * np.cos(theta),
        scale * np.sqrt(eigenvalues[1]) * np.sin(theta)
    ])
    
    # Rotate and translate
    R = eigenvectors
    ellipse_rotated = R @ ellipse
    ellipse_points = ellipse_rotated.T + center
    
    return ellipse_points

def plotly_embeddings(embeddings_2d, highlight="topic", write=False):
    # Define a color palette for topics
    topic_palette = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#f032e6',
    '#008080', '#800000', '#808000', '#ff6347', '#ee82ee', '#dc143c', '#00008b']

    # Define a color palette for clusters
    cluster_palette = {
        "-1": "#CCCCCC",
        "0": "#d62728",
        "1": "#1f77b4",
        "2": "#2ca02c",
        "3": "#ff7f0e",
    }

    # To map topics
    if highlight == "topic":
        color = None
        color_discrete_sequence = ["#8A8787"]
        color_discrete_map = None
        category_orders = None
        title = "Subreddits mapped by embeddings with topics"
        labels = {
            "x": "UMAP Dimension 1", 
            "y": "UMAP Dimension 2",
        }
        hover_data = {
        "SUBREDDIT": True, 
        "interaction_count": True,
        "topic": True,
        "x": False,
        "y": False
        }
    
    # To map toxic behaviors
    elif highlight == "cluster":
        color = "cluster"
        color_discrete_sequence = None
        color_discrete_map = cluster_palette
        category_orders = {"cluster": ["-1", "0", "1", "2", "3"]}  # Changed to "cluster"
        title = "Subreddits mapped by embeddings with clusters"
        labels = {
            "x": "UMAP Dimension 1", 
            "y": "UMAP Dimension 2",
            "cluster": "Cluster",
        }
        hover_data = {
        "SUBREDDIT": True, 
        "interaction_count": True,
        "topic": True,
        "cluster": True,
        "cluster_description": True,
        "x": False,
        "y": False
        }

    # To map mean toxicity scores
    elif highlight == "toxicity":
        color = "mean_toxicity_score"
        color_discrete_sequence = None
        color_discrete_map = None
        category_orders = None
        title = "Subreddits mapped by embeddings with toxicity"
        labels = {
            "x": "UMAP Dimension 1", 
            "y": "UMAP Dimension 2",
        }
        hover_data = {
        "SUBREDDIT": True, 
        "interaction_count": True,
        "mean_toxicity_score": True,
        "x": False,
        "y": False
        }
        
        # Map toxicity values to actual RGB colors
        toxicity_values = embeddings_2d['mean_toxicity_score'].values
        min_val = np.nanmin(toxicity_values)
        max_val = np.nanmax(toxicity_values)
        normalized = (toxicity_values - min_val) / (max_val - min_val)
        colorscale = pc.get_colorscale('RdBu_r')
        hover_toxicity = [pc.sample_colorscale(colorscale, [val])[0] for val in normalized]
        
    # Scatter plot using the specific colors/hover info for each highlight mode
    fig = px.scatter(
        embeddings_2d,
        x="x", 
        y="y",
        color=color,
        color_continuous_scale="RdBu_r" if highlight == "toxicity" else None,
        color_discrete_sequence=color_discrete_sequence if highlight != "toxicity" else None,
        color_discrete_map=color_discrete_map,
        hover_data=hover_data,
        category_orders=category_orders,
        title=title,
        labels=labels,
    )
    
    # Add ellipse boundaries for topics
    if highlight == "topic":
        topic_to_color = {}
        for i, topic in enumerate(embeddings_2d['topic'].dropna().unique()):
            topic_to_color[topic] = topic_palette[i % len(topic_palette)] 

        for topic in embeddings_2d['topic'].dropna().unique():
            df_topic = embeddings_2d[embeddings_2d['topic'] == topic]
            
            if len(df_topic) >= 3:
                points = df_topic[['x', 'y']].values
                
                try:
                    ellipse_boundary = fit_ellipse(points, scale=2)
                    
                    fig.add_trace(go.Scattergl(
                        x=ellipse_boundary[:, 0],
                        y=ellipse_boundary[:, 1],
                        mode='lines',
                        line=dict(color=topic_to_color[topic], width=2),
                        hoverinfo='skip',
                        fill='toself',
                        fillcolor=topic_to_color[topic],
                        opacity=0.3,
                        showlegend=True,
                        name=topic
                    ))
                except Exception as e:
                    print(f"Could not create boundary for {topic}: {e}")

    # Customize hover info
    for trace in fig.data:
        if trace.mode == 'markers':
            if highlight == "topic":
                topics = embeddings_2d['topic'].values
                colors = [topic_to_color.get(t, 'gray') for t in topics]
                
                trace.update(
                    hovertemplate='<b>Name:</b> %{customdata[0]}<br>' +
                                '<b>Interactions:</b> %{customdata[1]}<br>' +
                                '<b>Topic:</b> %{customdata[2]}<br>' +
                                '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=colors,
                        font_size=13,
                        font_family="Arial",
                        font_color="white"
                    )
                )
            elif highlight == "cluster":
                trace.update(
                    hovertemplate='<b>Name:</b> %{customdata[0]}<br>' +
                                '<b>Interactions:</b> %{customdata[1]}<br>' +
                                '<b>Topic:</b> %{customdata[2]}<br>' +
                                '<b>Cluster:</b> %{customdata[3]}<br>' +
                                '<b>Description:</b> %{customdata[4]}<br>' +
                                '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=trace.marker.color,  # Use actual colors
                        font_size=13,
                        font_family="Arial",
                        font_color="white"
                    )
                )
            elif highlight == "toxicity":
                trace.update(
                    hovertemplate='<b>Name:</b> %{customdata[0]}<br>' +
                                '<b>Interactions:</b> %{customdata[1]}<br>' +
                                '<b>Mean Toxicity Score:</b> %{customdata[2]:.4f}<br>' +
                                '<extra></extra>',
                    hoverlabel=dict(
                        bgcolor=hover_toxicity,
                        font_size=13,
                        font_family="Arial",
                        font_color="white"
                    )
                )

    fig.update_traces(opacity=0.7, marker=dict(line=dict(width=0)), selector=dict(mode='markers'))
    fig.update_layout(
        autosize=True,
        xaxis=dict(showgrid=True, gridwidth=0.5, fixedrange=False),
        yaxis=dict(showgrid=True, gridwidth=0.5, fixedrange=False),
        height=600,
        dragmode='pan',
    )

    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    fig.show(config=config)

    # Save to HTML to put on the website
    if write:
        filename = f"reddit_map_{highlight}.html"
        fig.write_html(filename, config=config)

def assign_clusters_embeddings(embeddings_2d):
    """
    Assign toxic behavior cluster labels to subreddits based on precomputed cluster assignments.
    """
    # Load clusters
    clusters_df = pd.read_csv("data/subreddit_cluster_assignments.csv")

    embeddings_with_clusters = embeddings_2d.copy()

    # Merge with cluster labels
    embeddings_with_clusters = (
        embeddings_2d
        .merge(
            clusters_df.set_index("subreddit"),
            left_on="SUBREDDIT",
            right_index=True,
            how="left"
        )
        .fillna(-1)
    )

    # Convert float -> int -> string
    embeddings_with_clusters['cluster'] = embeddings_with_clusters['cluster'].astype(int).astype(str)

    # Add cluster descriptive labels
    cluster_labels = {
        "0": "Moderate-volume Victim-leaning - Mixed",
        "1": "Moderate-volume Aggressor - Mixed",
        "2": "High-volume Balanced (Hub) - Bursty",
        "3": "Low-volume Victim - Mixed",
        "-1": "Unassigned (not toxic)"
    }

    embeddings_with_clusters['cluster_description'] = embeddings_with_clusters['cluster'].map(cluster_labels)
    return embeddings_with_clusters

def assign_clusters_interactions(interactions_df):
    """
    Assign toxic behavior cluster labels to the interactions dataframe (source and target)based on precomputed 
    cluster assignments.
    """
    # Load clusters
    clusters_df = pd.read_csv("data/subreddit_cluster_assignments.csv")

    # Merge cluster assignments with interactions_df for SOURCE_SUBREDDIT
    interactions_with_clusters = interactions_df.merge(
        clusters_df[['subreddit', 'cluster']],
        left_on="SOURCE_SUBREDDIT",
        right_on="subreddit",
        how="left"
    ).drop(columns=['subreddit']).fillna(-1).rename(columns={'cluster': 'source_cluster'})

    # Merge cluster assignments with interactions_with_clusters for TARGET_SUBREDDIT
    interactions_with_clusters = interactions_with_clusters.merge(
        clusters_df[['subreddit', 'cluster']],
        left_on="TARGET_SUBREDDIT",
        right_on="subreddit",
        how="left"
    ).drop(columns=['subreddit']).fillna(-1).rename(columns={'cluster': 'target_cluster'})
    
    # Convert float -> int -> string
    interactions_with_clusters['source_cluster'] = interactions_with_clusters['source_cluster'].astype(int).astype(str)
    interactions_with_clusters['target_cluster'] = interactions_with_clusters['target_cluster'].astype(int).astype(str)

    return interactions_with_clusters

def select_cluster_interactions(interactions_with_clusters, cluster=0, role="source"):
    """
    Select interactions where either the source or target subreddit belongs to a specific cluster.
    """
    return interactions_with_clusters[interactions_with_clusters[role + "_cluster"] == str(cluster)]