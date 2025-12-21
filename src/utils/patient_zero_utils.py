import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import copy
from datetime import date
from tigramite.data_processing import DataFrame as TDataFrame
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI

MAX_TIME_WINDOW = pd.to_datetime([date(2013, 12, 31), date(2017, 4, 30)]).date

# Case Studies Variables
CASE_STUDY_NAMES = ["bettersubredditdrama", "science", "drama"]
KEYWORDS = ["bettersubredditdrama", "science", "drama"]
TIME_WINDOWS = [
    [date(2015, 11, 10), date(2015, 12, 25)], 
    [date(2015, 4, 15), date(2015, 5, 5)], 
    [date(2015, 11, 1), date(2016, 1, 1)]
]
THRESHOLDS = [80, 80, 80]
TIME_SERIES_FREQUENCIES = ["12h", "6h", "D"]
MOVING_AVERAGE_WINDOWS = [5, 2, 3]
TOP_INTERACTORS_TO_PLOT = [8, 10, 10]
MAX_LAGS = [6, 4, 6]
ALPHAS = [0.05, 0.05, 0.05]
NB_SUBREDDITS_TO_FOCUS_PCMCI = [10, 10, 10]

def extract_focused_df(df, keyword="", time_window=None):
    """
    Extracts the sub-dataset which contains a specific keyword in its source/target subreddit names,
    and/or only in a certain time window. If neither the keyword nor the time window are given,
    returns a deepcopy of the initial dataframe.

    :param df: dataframe on which to extract the sub-dataset (panda dataframe)
    :param keyword: (OPTIONAL) keyword used to extract the sub-dataset (string)
    :param time_window: (OPTIONAL) time window used to extract the sub-dataset (python list of length 2)

    :return: The focused dataframe corresponding with input parameters (panda dataframe)
    """

    if time_window is None:
        if keyword == "":
            # Return the default focused_df (deepcopy of df)
            print("No keyword nor time window given, returning deepcopy.")
            return copy.deepcopy(df)

        else:
            # If no time_window given, take whole time range available
            time_window = MAX_TIME_WINDOW

    focused_df = df[
        (
            df["SOURCE_SUBREDDIT"].str.contains(keyword, case=False, na=False) |
            df["TARGET_SUBREDDIT"].str.contains(keyword, case=False, na=False)
        ) &
        (pd.to_datetime(df["TIMESTAMP"]).dt.date >= time_window[0]) &
        (pd.to_datetime(df["TIMESTAMP"]).dt.date <= time_window[1])
    ]

    return focused_df


def construct_interaction_count_df(df):
    """
    Constructs the interaction count dataframe based on given dataframe. It includes 3 columns
    (source interaction count, target interaction count and total interaction count) and as many rows as
    the input dataframe, all related to a unique subreddit link.

    :param df: dataframe from which to construct the interaction counts (panda dataframe)

    :return: interaction_counts (panda dataframe)
    """

    # Count the number of source interactions
    source_counts = (
        df["SOURCE_SUBREDDIT"]
        .value_counts()
        .rename("SOURCE_INTERACTIONS")
    )

    # Count the number of target interactions
    target_counts = (
        df["TARGET_SUBREDDIT"]
        .value_counts()
        .rename("TARGET_INTERACTIONS")
    )

    # Create a dataframe with both count numbers
    # Note that in case of missing values (only sourced subreddits or only targeted subreddits),
    # a zero (source or target) interaction count value is attributed.
    interaction_counts = (
        pd.concat([source_counts, target_counts], axis=1)
        .fillna(0)
        .astype(int)
    )

    # Create a third column with total interactions
    interaction_counts["TOTAL_INTERACTIONS"] = (
            interaction_counts["SOURCE_INTERACTIONS"] +
            interaction_counts["TARGET_INTERACTIONS"]
    )

    # Reset index to add 'SUBREDDIT' as the name of the data column
    interaction_counts = interaction_counts.reset_index().rename(columns={"index": "SUBREDDIT"})

    # Sort by number of total interactions
    interaction_counts = interaction_counts.sort_values(by=["TOTAL_INTERACTIONS"], ascending=False)

    return interaction_counts


def construct_daily_interaction_df(df, keyword="", time_window=None, bin_freq="D"):
    """
    Constructs the interaction count dataframe based on given dataframe.
    It includes 3 columns (source interaction count, target interaction count
    and total interaction count) and as many rows as the input dataframe,
    all related to a unique time bin (increasing in time).

    :param df: dataframe from which to construct the interaction counts (panda dataframe)
    :param keyword: (OPTIONAL) keyword used to extract the sub-dataset (string)
    :param time_window: (OPTIONAL) time window used to extract the sub-dataset (python list of length 2)
    :param bin_freq: (OPTIONAL) frequency of the time bins (e.g. "D", "4H", "1H")

    :return: daily_interactions (panda dataframe)
    """

    # If no time_window given, take whole time range available
    if time_window is None:
        time_window = MAX_TIME_WINDOW

    # Filter dataset by time window
    mask = (
        (pd.to_datetime(df["TIMESTAMP"]).dt.date >= time_window[0]) &
        (pd.to_datetime(df["TIMESTAMP"]).dt.date <= time_window[1])
    )
    df = df.loc[mask]

    # Define time bins according to chosen frequency
    time_bins = pd.to_datetime(df["TIMESTAMP"]).dt.floor(bin_freq)

    # Count the number of source interactions per time bin
    daily_interactions_source = (
        df[df["SOURCE_SUBREDDIT"].str.contains(keyword, case=False, na=False)]
        .groupby(time_bins)
        .size()
        .rename("DAILY_TOTAL_SOURCE_INTERACTIONS")
    )

    # Count the number of target interactions per time bin
    daily_interactions_target = (
        df[df["TARGET_SUBREDDIT"].str.contains(keyword, case=False, na=False)]
        .groupby(time_bins)
        .size()
        .rename("DAILY_TOTAL_TARGET_INTERACTIONS")
    )

    # Create the full time range with chosen frequency
    time_range = pd.date_range(
        start=pd.to_datetime(time_window[0]),
        end=pd.to_datetime(time_window[1]),
        freq=bin_freq
    )

    # Reindex to include missing time bins
    daily_interactions_source = daily_interactions_source.reindex(time_range, fill_value=0)
    daily_interactions_target = daily_interactions_target.reindex(time_range, fill_value=0)

    # Concatenate into one dataframe
    daily_interactions = pd.concat(
        [daily_interactions_source, daily_interactions_target],
        axis=1
    )

    # Add total column
    daily_interactions["DAILY_TOTAL_INTERACTIONS"] = (
        daily_interactions["DAILY_TOTAL_SOURCE_INTERACTIONS"] +
        daily_interactions["DAILY_TOTAL_TARGET_INTERACTIONS"]
    )

    # Ensure datetime index
    daily_interactions.index = pd.to_datetime(daily_interactions.index)
    daily_interactions.index.name = "TIME"

    return daily_interactions


def construct_neg_interaction_df(
    df,
    time_window=None,
    criteria=None,
    threshold=None,
    upper=None,
    groupby_type="DAY",
    bin_freq="D"
):
    """
    Constructs the negative interaction count (series or) dataframe based on given dataframe, given criterion,
    given thresholds and groupby type (daily, per subreddit, ...).
    It includes as many columns as criteria and as many rows as the input dataframe groupby_type has (number
    of time bins, number of subreddits, ...).

    :param df: dataframe from which to construct the interaction counts (panda dataframe)
    :param time_window: (OPTIONAL) time window used to extract the sub-dataset (python list of length 2)
    :param criteria: (OPTIONAL) list of criteria on which perform the counts (python list)
    :param threshold: (OPTIONAL) list of threshold for each criterion, upper or lower (python list)
    :param upper: (OPTIONAL) whether if considered thresholds are upper (True) or lower (False) bounds
    :param groupby_type: (OPTIONAL) type of grouping ("DAY", "SOURCE_SUBREDDIT", ...)
    :param bin_freq: (OPTIONAL) time bin frequency ("D", "4h", "1h", ...)

    :return: neg_interaction_df (panda dataframe)
    """

    if criteria is None:
        criteria = ["LINK_SENTIMENT"]

    if threshold is None:
        threshold = [0]

    if upper is None:
        upper = [True for _ in range(len(threshold))]

    if not (len(criteria) == len(threshold) and len(criteria) == len(upper)):
        print("Error: Number of criteria doesn't match number of thresholds or upper/lower flags.")
        return None

    if time_window is None:
        time_window = MAX_TIME_WINDOW

    df = df.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    # Filter by time window
    mask = (
        (df["TIMESTAMP"] >= pd.to_datetime(time_window[0])) &
        (df["TIMESTAMP"] <= pd.to_datetime(time_window[1]))
    )
    df = df.loc[mask]

    # --- Define grouping ---
    if groupby_type == "DAY":
        groupby_key = pd.Grouper(key="TIMESTAMP", freq=bin_freq)
        time_range = pd.date_range(
            start=time_window[0],
            end=time_window[1],
            freq=bin_freq
        )
        all_keys = None
    else:
        groupby_key = groupby_type
        all_keys = df[groupby_key].dropna().unique()
        time_range = None

    neg_interaction_df = pd.DataFrame()

    # --- Compute counts for each criterion ---
    for i in range(len(criteria)):
        upper_sign = 1 if upper[i] else -1
        column_name = f"{criteria[i]}_{threshold[i]}_{'UPPER' if upper[i] else 'LOWER'}"

        series = (
            df[df[criteria[i]] * upper_sign <= upper_sign * threshold[i]]
            .groupby(groupby_key)[criteria[i]]
            .count()
        )

        # Reindex to ensure consistent shape
        if time_range is not None:
            series = series.reindex(time_range, fill_value=0)
        elif all_keys is not None:
            series = series.reindex(all_keys, fill_value=0)

        neg_interaction_df[column_name] = series

    # Final index formatting
    if time_range is not None:
        neg_interaction_df.index = pd.to_datetime(neg_interaction_df.index)
        neg_interaction_df.index.name = "TIME"
    else:
        neg_interaction_df.index.name = groupby_type

    return neg_interaction_df


def smart_df_add(df1, df2):
    """
    Adds two dataframes in a robust way, by ensuring indexes and columns union and filling missing values with 0s.

    :param df1: first dataframe to add (panda dataframe)
    :param df2: second dataframe to add (panda dataframe)

    :return: added_df (panda dataframe)
    """
    # Compute union index & union columns
    all_index = df1.index.union(df2.index)
    all_cols = df1.columns.union(df2.columns)

    # Reindex both frames to the union and fill missing with 0
    left = df1.reindex(index=all_index, columns=all_cols, fill_value=0)
    right = df2.reindex(index=all_index, columns=all_cols, fill_value=0)

    # Ensure numeric dtype
    left = left.apply(pd.to_numeric, errors='coerce').fillna(0)
    right = right.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Add element-wise
    added_df = left.add(right, fill_value=0)

    return added_df


def plot_daily_interactions(dataframe, bin_freq="D",
                            specifier="", threshold=None, moving_avg_window=None,
                            html_filename="CS9_daily_interactions.html"):
    """
    Plot daily interactions using Plotly with filled areas under curves.
    
    :param dataframe: DataFrame with datetime index and interaction columns
    :param bin_freq: Frequency for binning (e.g., "D" for day, "H" for hour)
    :param bin_interval: Interval for tick marks
    :param specifier: Label specifier (e.g., "Reddit", "Twitter")
    :param threshold: Optional threshold value (as percentage of max)
    :param moving_avg_window: Optional window size for moving average
    :param html_filename: Output HTML filename
    """
    
    dataframe = dataframe.copy()
    
    # Convert index to timestamp if needed
    if isinstance(dataframe.index, pd.PeriodIndex):
        dataframe.index = dataframe.index.to_timestamp()
    
    dataframe.index = pd.to_datetime(dataframe.index)
    
    # Create figure
    fig = go.Figure()
    
    # Sort columns by their maximum values (descending) to layer properly
    col_max_values = [(col, dataframe[col].max()) for col in dataframe.columns]
    col_max_values.sort(key=lambda x: x[1], reverse=True)
    sorted_cols = [col for col, _ in col_max_values]
    
    # Add traces for each column (largest first, so smallest appears on top)
    for col in sorted_cols:
        fig.add_trace(go.Scatter(
            x=dataframe.index,
            y=dataframe[col],
            mode='lines',
            name=col,
            fill='tozeroy',
            line=dict(width=2),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y}<extra></extra>'
        ))
    
    # Add moving average if specified
    if moving_avg_window is not None:
        if bin_freq == 'D':
            temp = str(moving_avg_window) + '-day'
        else:
            temp = int(bin_freq[:-1]) * moving_avg_window
            temp = str(temp) + 'h'

        # Calculate total interactions across all columns
        total_interactions = dataframe.sum(axis=1)
        moving_avg = total_interactions.rolling(window=moving_avg_window, center=True).mean()
        
        fig.add_trace(go.Scatter(
            x=dataframe.index,
            y=moving_avg,
            mode='lines',
            name=f'{temp} Moving Average',
            line=dict(color='red', width=2),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Count: %{y:.1f}<extra></extra>'
        ))
    
    # Add threshold line if specified
    if threshold is not None:
        threshold_value = threshold / 100 * dataframe.max().max()
        fig.add_trace(go.Scatter(
            x=dataframe.index,
            y=[threshold_value] * len(dataframe),
            mode='lines',
            name=f'{threshold}% of max value',
            line=dict(color='red', dash='dash', width=2),
            hoverinfo='skip'
        ))
    
    # Update layout
    x_label = "Time" if "h" in bin_freq.lower() else "Day"

    current_case_study = html_filename[15]
    
    fig.update_layout(
        title=f"Total {specifier} Interactions over time (Case Study {current_case_study}: '{CASE_STUDY_NAMES[int(current_case_study)-1]}')",
        xaxis_title=x_label,
        yaxis_title=f"Number of {specifier} Interactions",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            griddash='dash'
        ),
        height=400,
        autosize=True,
    )
    
    config = {
        'scrollZoom': False,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    fig.show(config=config)
    fig.write_html(html_filename, config=config)


def plot_top_interactors(interaction_count_df,
                         x_col, y_cols,
                         legend,
                         x_ticklabels,
                         plot_top=10,
                         y_axis_scale='linear',
                         html_filename="top_interactors.html"):
    """
    Plot top interactors using Plotly.
    
    :param interaction_count_df: DataFrame with interaction counts
    :param x_col: Column name for x-axis (or None to use index)
    :param y_cols: List of column names for y-axis (bar groups)
    :param legend: List of legend labels
    :param x_ticklabels: List of labels for x-axis ticks
    :param plot_top: Number of top items to plot
    """
    
    # Get top N rows
    df_top = interaction_count_df.head(plot_top)
    
    # Create figure
    fig = go.Figure()
    
    # Determine x values - use provided labels or extract from data
    if x_ticklabels is not None:
        x_labels = list(x_ticklabels[:len(df_top)])
    elif x_col is None:
        x_labels = list(df_top.index)
    else:
        x_labels = list(df_top[x_col])
    
    # Convert y_cols to list if it's not already
    y_cols_list = list(y_cols) if hasattr(y_cols, '__iter__') and not isinstance(y_cols, str) else [y_cols]
    legend_list = list(legend) if hasattr(legend, '__iter__') and not isinstance(legend, str) else [legend]
    
    # Add bar traces for each y column
    for i, col in enumerate(y_cols_list):
        fig.add_trace(go.Bar(
            x=x_labels,
            y=df_top[col],
            name=str(legend_list[i]) if i < len(legend_list) else str(col),
            hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>Count: %{y}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Top {plot_top} Subreddits by Number of Interactions",
        xaxis_title="Subreddit",
        yaxis_title="Number of Interactions",
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.80,
            y=0.99,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        legend_title_text="Interaction Type",
        xaxis=dict(
            showgrid=False,
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            type=y_axis_scale,
            gridcolor='lightgray',
            griddash='dash'
        ),
        barmode='stack',
        height=400,
        autosize=True,
        #width=900,
        margin=dict(t=40, b=20, l=10, r=10),
    )

    config = {
        'scrollZoom': False,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    fig.show(config=config)
    fig.write_html(html_filename, config=config)


def construct_time_series_df(df, subreddit_names, time_window=None, bin_freq="D"):
    """
    Constructs a time series dataframe of interactions for given subreddits.
    Each value corresponds to the number of interactions within a time bin.

    :param df: interactions dataframe
    :param subreddit_names: list of subreddit names
    :param time_window: time window [start, end]
    :param bin_freq: time bin frequency ("D", "4H", "1H", ...)
    """

    if time_window is None:
        time_window = MAX_TIME_WINDOW

    df = df.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

    # Create time bins
    df["TIME_BIN"] = df["TIMESTAMP"].dt.floor(bin_freq)

    # Filter by subreddits and time window
    focused_df = df[
        (
            df["SOURCE_SUBREDDIT"].isin(subreddit_names) |
            df["TARGET_SUBREDDIT"].isin(subreddit_names)
        ) &
        (df["TIME_BIN"] >= pd.to_datetime(time_window[0])) &
        (df["TIME_BIN"] <= pd.to_datetime(time_window[1]))
    ]

    # Long format: source + target
    long_df = pd.concat(
        [
            focused_df[["TIME_BIN", "SOURCE_SUBREDDIT"]]
            .rename(columns={"SOURCE_SUBREDDIT": "subreddit"}),

            focused_df[["TIME_BIN", "TARGET_SUBREDDIT"]]
            .rename(columns={"TARGET_SUBREDDIT": "subreddit"})
        ],
        axis=0
    )

    # Aggregate counts
    time_series_df = (
        long_df
        .groupby(["TIME_BIN", "subreddit"])
        .size()
        .unstack(fill_value=0)
    )

    # Full time index with chosen binning
    full_time_index = pd.date_range(
        start=pd.to_datetime(time_window[0]),
        end=pd.to_datetime(time_window[1]),
        freq=bin_freq
    )

    # Reindex to ensure complete coverage
    time_series_df = (
        time_series_df
        .reindex(index=full_time_index, fill_value=0)
        .reindex(columns=subreddit_names, fill_value=0)
    )

    time_series_df.index.name = "TIME"

    return time_series_df


def compute_PCMCI(time_series_df, max_lag=5, alpha=0.05, show=False, identifier='(Non-Toxic on Toxic)', case_study_id_py=0):
    """
    Function performing PCMCI causal discovery on the given time series dataframe using tigramite library. 
    Returns the p-value matrix, effect value matrix and list of significant edges discovered.
    Optionally visualizes the discovered causal graph using networkx.
    
    :param time_series_df: Groupement of timeseries to perform PCMCI on (panda dataframe, T x N)
    :param max_lag: Maximum time lag to consider (int)
    :param alpha: Significance level used on p-value for link extraction (float)
    :param show: Whether to visualize the discovered causal graph (boolean)

    :return: p_matrix (N x N x max_lag+1), val_matrix (N x N x max_lag+1), edges (list of dicts)
    """

    # Data transformation to tigramite format
    data_arr = time_series_df.values
    var_names = list(time_series_df.columns)
    tdata = TDataFrame(data_arr, var_names=var_names)

    # PCMCI from tigramite
    parcorr = RobustParCorr(significance='analytic') # CI type: partial correlation test
    pcmci = PCMCI(dataframe=tdata, cond_ind_test=parcorr)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None)

    # Extracting results
    p_matrix = results['p_matrix']     # (p-values) N_vars x N_vars x max_lag+1
    val_matrix = results['val_matrix'] # (effect values e.g. partial correlations) N_vars x N_vars x max_lag+1

    # Extracting significant links based on provided alpha
    edges = []
    N = len(var_names)
    for src in range(N):
        for tgt in range(N):
            for lag in range(1, max_lag+1): # skipping zero-lag
                pval = p_matrix[src, tgt, lag]
                if pval is not None and (pval < alpha):
                    effect = val_matrix[src, tgt, lag]
                    edges.append({
                        'source': var_names[src],
                        'target': var_names[tgt],
                        'lag': lag,
                        'pval': float(pval),
                        'effect': float(effect)
                    })
    edges = sorted(edges, key=lambda e: (e['pval'], -abs(e['effect'])))

    # Create NX digraph if show is True
    if show:
        G = nx.DiGraph()
        G.add_nodes_from(var_names)

        # Add edges with weight and label
        for e in edges:
            src, tgt, lag, eff = e['source'], e['target'], e['lag'], e['effect']
            lbl = f"lag={lag}\n{eff:.2f}"
            # if multiple edges between same nodes with different lag, we aggregate by writing both labels (simple approach)
            if G.has_edge(src, tgt):
                prev = G[src][tgt].get('label', '')
                G[src][tgt]['label'] = prev + ' | ' + lbl
                G[src][tgt]['weight'] += abs(eff)
            else:
                G.add_edge(
                    src, tgt,
                    label=lbl,
                    weight=abs(eff),
                    effect=eff   # ← store signed effect
                )

        # Plot (for readability limit nodes if huge)
        plt.figure(figsize=(12, 9))

        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except:
            pos = nx.kamada_kawai_layout(G)

        # --- Force PROTAGONIST node to the center ---
        if KEYWORDS[case_study_id_py] in pos:
            pos[KEYWORDS[case_study_id_py]] = np.array([0.0, 0.0])

        node_colors = [
            "gold" if n == KEYWORDS[case_study_id_py] else "lightblue"
            for n in G.nodes()
        ]

        node_sizes = [
            1300 if n == KEYWORDS[case_study_id_py] else 900
            for n in G.nodes()
        ]

        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="black"
        )

        nx.draw_networkx_labels(
            G, pos,
            font_size=10
        )

        edge_colors = []
        edge_widths = []

        for u, v in G.edges():
            edge_widths.append(max(1.5, G[u][v]['weight'] * 3))

            if u == KEYWORDS[case_study_id_py] or v == KEYWORDS[case_study_id_py]:
                eff = G[u][v].get("effect", 0.0)
                if eff > 0:
                    edge_colors.append("crimson")  # amplifying
                else:
                    edge_colors.append("green")    # suppressing
            else:
                edge_colors.append("gray")


        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            arrowstyle="->",
            arrowsize=16,
            alpha=0.85
        )


        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8
        )

        plt.title(f"PCMCI causal graph {identifier}")
        plt.axis("off")
        plt.show()

    return p_matrix, val_matrix, edges


def plot_pcmci_interactive_html(edges, var_names, case_study_id_py, identifier='(Non-Toxic on Toxic)', output_html="pcmci.html"):
    """
    Interactive PCMCI graph with lag slider.
    Nodes stay fixed, edges accumulate as lag increases.

    :param edges: list of edge dicts from compute_PCMCI
    :param var_names: list of variable names
    :param output_html: output HTML file
    """

    # --- Build graph ---
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    for e in edges:
        G.add_edge(
            e["source"],
            e["target"],
            lag=e["lag"],
            effect=e["effect"]
        )

    # --- Fixed layout ---
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        pos = nx.kamada_kawai_layout(G)

    if KEYWORDS[case_study_id_py] in pos:
        pos[KEYWORDS[case_study_id_py]] = np.array([0.0, 0.0])

    # --- Edge traces per lag (CREATE EDGES FIRST) ---
    max_lag = max(e["lag"] for e in edges)
    edge_traces_by_lag = {lag: [] for lag in range(1, max_lag + 1)}

    for e in edges:
        src, tgt = e["source"], e["target"]
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        color = (
            "red" if (tgt == KEYWORDS[case_study_id_py] and e["effect"] > 0) or (src == KEYWORDS[case_study_id_py] and e["effect"] > 0)
            else "green" if (tgt == KEYWORDS[case_study_id_py]) or (src == KEYWORDS[case_study_id_py])
            else "gray"
        )

        # Create edge line
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(width=2, color=color),
            hoverinfo="skip",  # Skip hover on the line itself
            showlegend=False,
            visible=(e["lag"] == 1)  # Only lag=1 visible initially
        )
        
        # Create invisible marker at midpoint for hover
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Set hover text color based on edge color
        if color == "gray":
            hover_text = f"<span style='color:white'><b>{src} → {tgt}</b><br>Lag: {e['lag']}<br>Correlation: {e['effect']:.4f}</span>"
        else:
            hover_text = f"<b>{src} → {tgt}</b><br>Lag: {e['lag']}<br>Correlation: {e['effect']:.4f}"
        
        hover_trace = go.Scatter(
            x=[mid_x],
            y=[mid_y],
            mode="markers",
            marker=dict(size=15, color=color, opacity=0.01),  # Nearly invisible
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=False,
            visible=(e["lag"] == 1)  # Only lag=1 visible initially
        )

        edge_traces_by_lag[e["lag"]].extend([edge_trace, hover_trace])

    # --- Node trace (CREATE NODES LAST so they appear on top) ---
    node_x, node_y, node_text = [], [], []
    node_hover = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)
        node_hover.append(f"<b>{n}</b>")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=[30 if n == KEYWORDS[case_study_id_py] else 20 for n in G.nodes()],
            color=["gold" if n == KEYWORDS[case_study_id_py] else "lightblue" for n in G.nodes()],
            opacity=1.0,
            line=dict(width=1, color="black")
        ),
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False
    )

    # --- Collect all traces (EDGES FIRST, THEN NODES) ---
    all_traces = []
    for lag in range(1, max_lag + 1):
        all_traces.extend(edge_traces_by_lag[lag])
    all_traces.append(node_trace)  # Add nodes last so they're on top

    # --- Create slider steps ---
    steps = []
    for selected_lag in range(1, max_lag + 1):
        # Create visibility list for this lag
        visible = []
        
        for lag in range(1, max_lag + 1):
            for _ in edge_traces_by_lag[lag]:
                visible.append(lag <= selected_lag)
        
        visible.append(True)  # Node trace always visible (now at the end)
        
        if selected_lag == 1:
            step = dict(
                method="update",
                args=[{"visible": visible}],
                label=f"Lag = {selected_lag}"
            )
        else:
            step = dict(
                method="update",
                args=[{"visible": visible}],
                label=f"Lag ≤ {selected_lag}"
            )
        steps.append(step)

    sliders = [dict(
        active=0,
        steps=steps
    )]

    # --- Figure ---
    fig = go.Figure(data=all_traces)

    fig.update_layout(
        title=f"PCMCI Causal Graph {identifier}",
        showlegend=False,
        sliders=sliders,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        hovermode='closest',
        height=600,
        autosize=True,
        #width=900,
    )

    config = {
        'scrollZoom': False,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    fig.show(config=config)
    fig.write_html(output_html, config=config)

def average_toxicity_traits(df, groupby_col='SOURCE_SUBREDDIT'):
    """
    Calculate average toxicity traits from interaction dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns: SOURCE_SUBREDDIT, TARGET_SUBREDDIT, 
        POST_ID, TIMESTAMP, TOXICITY_CAT
    groupby_col : str, default='SOURCE_SUBREDDIT'
        Column to group by for duration calculation. Options:
        'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', or 'BOTH'
    
    Returns:
    --------
    dict with keys:
        - 'avg_daily_toxicity': float, average proportion of toxic interactions per day
        - 'avg_toxic_duration_hours': float, average time until next interaction after a toxic one
        - 'num_toxic_interactions': int, number of toxic interactions with measurable duration
    """
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure TIMESTAMP is datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    # 1. Calculate average daily toxicity
    df['DATE'] = df['TIMESTAMP'].dt.date
    daily_toxicity = df.groupby('DATE')['TOXICITY_CAT'].mean()
    avg_daily_toxicity = daily_toxicity.mean()
    
    # 2. Calculate average duration of toxic interactions
    toxic_durations = []
    
    # Determine grouping
    if groupby_col == 'BOTH':
        group_cols = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']
    else:
        group_cols = [groupby_col]
    
    # Process each subreddit (or subreddit pair)
    for group_key, group_df in df.groupby(group_cols):
        # Sort by timestamp
        group_df = group_df.sort_values('TIMESTAMP').reset_index(drop=True)
        
        # For each toxic interaction, measure time until next interaction
        for idx in range(len(group_df) - 1):
            if group_df.iloc[idx]['TOXICITY_CAT'] == 1:
                current_time = group_df.iloc[idx]['TIMESTAMP']
                next_time = group_df.iloc[idx + 1]['TIMESTAMP']
                duration = (next_time - current_time).total_seconds() / 3600  # Convert to hours
                toxic_durations.append(duration)
    
    # Calculate average duration
    if toxic_durations:
        avg_toxic_duration = np.mean(toxic_durations)
    else:
        avg_toxic_duration = 0.0

    print(f"Average daily toxicity: {avg_daily_toxicity:.3f}")
    print(f"Average toxic duration: {avg_toxic_duration:.2f} hours")


def run_time_window_identification(interactions_df, case_study_id):
    # Get python ID
    case_study_id_py = case_study_id - 1

    # Subreddit keyword to focus on
    KEYWORD = KEYWORDS[case_study_id_py]

    # Extract focused dadaframe
    temp_focused_df = extract_focused_df(interactions_df, KEYWORD)

    # Construct daily interaction dataframe (without time window restriction)
    daily_interaction_df = construct_daily_interaction_df(temp_focused_df, KEYWORD, bin_freq='D')
    daily_neg_interaction_df = construct_daily_interaction_df(temp_focused_df[temp_focused_df["TOXICITY_CAT"] == 1], KEYWORD, bin_freq='D')

    # html filenames
    html_filename_1 = f"Results_html/CS{case_study_id}/CS{case_study_id}_1_daily_interactions_overall.html"
    html_filename_2 = f"Results_html/CS{case_study_id}/CS{case_study_id}_2_daily_interactions_negative.html"

    # Plot daily interactions
    plot_daily_interactions(daily_interaction_df[["DAILY_TOTAL_TARGET_INTERACTIONS",
                                                "DAILY_TOTAL_SOURCE_INTERACTIONS"]], 
                                                threshold=THRESHOLDS[case_study_id_py],
                                                html_filename=html_filename_1)
    plot_daily_interactions(pd.DataFrame(daily_neg_interaction_df["DAILY_TOTAL_INTERACTIONS"].rename("DAILY_TOTAL_TOXIC_INTERACTIONS")), 
                                                threshold=THRESHOLDS[case_study_id_py], 
                                                specifier="Toxic",
                                                html_filename=html_filename_2)

def run_init_toxic_event_assessment(interactions_df, case_study_id):

    # Get python ID
    case_study_id_py = case_study_id - 1
    
    # Parameters to consider
    TIME_WINDOW = TIME_WINDOWS[case_study_id_py]
    KEYWORD = KEYWORDS[case_study_id_py]

    # Extract focused dadaframe
    focused_df = extract_focused_df(interactions_df, KEYWORD, TIME_WINDOW)

    # Extract focused subreddits
    interacting_subreddits = pd.unique(focused_df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].values.ravel()).tolist()
    if case_study_id_py == 0:
        neighboring_subreddits = pd.read_csv("Data/CS_networks/network_target_cluster0")["subreddit"].tolist()
    elif case_study_id_py == 1:
        neighboring_subreddits = pd.read_csv("Data/CS_networks/network_source_cluster1")["subreddit"].tolist()
    elif case_study_id_py == 2:
        neighboring_subreddits = pd.read_csv("Data/CS_networks/network_source_cluster2")["subreddit"].tolist() + pd.read_csv("Data/CS_networks/network_target_cluster2")["subreddit"].tolist()
    else:
        print("Error: Case study ID not recognized for neighboring subreddits.")
        neighboring_subreddits = []
    focused_subreddits = list(set(interacting_subreddits).union(set(neighboring_subreddits)))
    print("-------------------------------------------")
    print("Case Study", case_study_id, ":")
    print(len(interacting_subreddits), "interacting subreddits found.")
    print(len(focused_subreddits) - len(interacting_subreddits), "additional neighboring subreddits found. (", 
          (len(focused_subreddits) - len(interacting_subreddits)) / len(neighboring_subreddits) * 100, 
          "% of neighboring subreddits included.)")
    print("-------------------------------------------")

    return focused_df, focused_subreddits

def run_daily_interactions(focused_df, case_study_id):
    
    # Get python ID
    case_study_id_py = case_study_id - 1
    
    # Parameters to consider
    TIME_WINDOW = TIME_WINDOWS[case_study_id_py]
    KEYWORD = KEYWORDS[case_study_id_py]
    TIME_SERIES_FREQ = TIME_SERIES_FREQUENCIES[case_study_id_py]

    # Count the number of daily interactions across all considered subreddits
    daily_interaction_df = construct_daily_interaction_df(focused_df, KEYWORD, TIME_WINDOW, bin_freq=TIME_SERIES_FREQ)
    plot_daily_interactions(daily_interaction_df[["DAILY_TOTAL_SOURCE_INTERACTIONS",
                                                "DAILY_TOTAL_TARGET_INTERACTIONS",
                                                "DAILY_TOTAL_INTERACTIONS"]],
                                                html_filename=f"Results_html/CS{case_study_id}/CS{case_study_id}_3_daily_interactions_windowed.html",
                                                bin_freq=TIME_SERIES_FREQ)
    
def run_top_interactors(focused_df, case_study_id):

    # Get python ID
    case_study_id_py = case_study_id - 1

    # Count the number of interaction as source and as target for each subreddit
    sr_interaction_count_df = construct_interaction_count_df(focused_df)

    # Choose the number of subreddits to plot
    NB_TO_PLOT = TOP_INTERACTORS_TO_PLOT[case_study_id_py]

    plot_top_interactors(sr_interaction_count_df,
                        x_col = "SUBREDDIT",
                        y_cols = ["SOURCE_INTERACTIONS", "TARGET_INTERACTIONS"],
                        legend = ["As Source", "As Target", "Total"],
                        x_ticklabels=sr_interaction_count_df.head(NB_TO_PLOT)["SUBREDDIT"],
                        plot_top = NB_TO_PLOT,
                        html_filename=f"Results_html/CS{case_study_id}/CS{case_study_id}_4_top_interactors_windowed.html")
    return sr_interaction_count_df

def run_toxic_interactions_over_time(focused_df, case_study_id):

    # Get python ID
    case_study_id_py = case_study_id - 1

    # Parameters to consider
    TIME_WINDOW = TIME_WINDOWS[case_study_id_py]
    TIME_SERIES_FREQ = TIME_SERIES_FREQUENCIES[case_study_id_py]
    CRITERION = ["TOXICITY_CAT"]
    THRESHOLDS_toxic = [0.5]
    UPPER = [False]

    daily_neg_interaction_df = construct_neg_interaction_df(focused_df, 
                                                            time_window=TIME_WINDOW, 
                                                            criteria=CRITERION, 
                                                            threshold=THRESHOLDS_toxic, 
                                                            upper=UPPER,
                                                            groupby_type="DAY",
                                                            bin_freq=TIME_SERIES_FREQ)

    plot_daily_interactions(daily_neg_interaction_df, 
                            specifier="Toxic", 
                            bin_freq=TIME_SERIES_FREQ, 
                            moving_avg_window=MOVING_AVERAGE_WINDOWS[case_study_id_py],
                            html_filename=f"Results_html/CS{case_study_id}/CS{case_study_id}_5_daily_negative_interactions_windowed.html")
    
    return daily_neg_interaction_df 

def run_top_toxic_interactors(focused_df, case_study_id):

    # Get python ID
    case_study_id_py = case_study_id - 1

    # Parameters to consider
    TIME_WINDOW = TIME_WINDOWS[case_study_id_py]
    CRITERION = ["TOXICITY_CAT"]
    THRESHOLDS_toxic = [0.5]
    UPPER = [False]
    NB_TO_PLOT = TOP_INTERACTORS_TO_PLOT[case_study_id_py]

    # Compute negative interactions as source
    sr_neg_source_interaction_df = construct_neg_interaction_df(focused_df, 
                                                            time_window=TIME_WINDOW, 
                                                            criteria=CRITERION, 
                                                            threshold=THRESHOLDS_toxic, 
                                                            upper=UPPER,
                                                            groupby_type="SOURCE_SUBREDDIT")

    # Compute negative interactions as target
    sr_neg_target_interaction_df = construct_neg_interaction_df(focused_df, 
                                                            time_window=TIME_WINDOW, 
                                                            criteria=CRITERION, 
                                                            threshold=THRESHOLDS_toxic, 
                                                            upper=UPPER,
                                                            groupby_type="TARGET_SUBREDDIT")

    # Add interactions and sort them by total
    sr_neg_interaction_df = smart_df_add(sr_neg_source_interaction_df, sr_neg_target_interaction_df)
    sr_neg_interaction_df["TOTAL_NEGATIVE_INTERACTIONS"] = sr_neg_interaction_df.sum(axis=1)
    sr_neg_interaction_df = sr_neg_interaction_df.sort_values(by="TOXICITY_CAT_0.5_LOWER", ascending=False)

    plot_top_interactors(sr_neg_interaction_df,
                        x_col = None,
                        y_cols = sr_neg_interaction_df.columns[:-1],
                        legend = sr_neg_interaction_df.columns,
                        x_ticklabels = sr_neg_interaction_df.head(NB_TO_PLOT).index,
                        plot_top = NB_TO_PLOT,
                        y_axis_scale='log',
                        html_filename=f"Results_html/CS{case_study_id}/CS{case_study_id}_6_top_negative_interactors_windowed.html")
    
def run_pcmci_analysis(daily_neg_interaction_df, focused_subreddits, sr_interaction_count_df, interactions_df, case_study_id):

    # Get python ID
    case_study_id_py = case_study_id - 1

    # Parameters to consider
    TIME_WINDOW = TIME_WINDOWS[case_study_id_py]
    MAX_LAG = MAX_LAGS[case_study_id_py]
    ALPHA = ALPHAS[case_study_id_py]
    NB_SUBREDDITS_TO_FOCUS = NB_SUBREDDITS_TO_FOCUS_PCMCI[case_study_id_py]

    finer_focused_subreddits = list(sr_interaction_count_df.sort_values(by="TOTAL_INTERACTIONS", ascending=False).head(NB_SUBREDDITS_TO_FOCUS)["SUBREDDIT"])

    time_series_neg_df = construct_time_series_df(interactions_df[interactions_df["TOXICITY_CAT"] == 1], subreddit_names=finer_focused_subreddits, time_window=TIME_WINDOW)
    time_series_df = construct_time_series_df(interactions_df[interactions_df["TOXICITY_CAT"] == 0], subreddit_names=finer_focused_subreddits, time_window=TIME_WINDOW)

    p_matrix, val_matrix, edges = compute_PCMCI(time_series_df, max_lag=MAX_LAG, alpha=ALPHA, show=False, case_study_id_py=case_study_id_py)
    p_matrix_neg, val_matrix_neg, edges_neg = compute_PCMCI(time_series_neg_df, max_lag=MAX_LAG, alpha=ALPHA, show=False, identifier='(Toxic on Toxic)', case_study_id_py=case_study_id_py)

    return edges, edges_neg, finer_focused_subreddits