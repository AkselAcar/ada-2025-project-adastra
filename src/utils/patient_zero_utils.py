import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import networkx as nx
import matplotlib.pyplot as plt
import copy
from datetime import date
from tigramite.data_processing import DataFrame as TDataFrame
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI

MAX_TIME_WINDOW = pd.to_datetime([date(2013, 12, 31), date(2017, 4, 30)])

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


def construct_daily_interaction_df(df, keyword="", time_window=None):
    """
    Constructs the daily interaction count dataframe based on given dataframe. It includes 3 columns
    (daily source interaction count, daily target interaction count and daily total interaction count)
    and as many rows as the input dataframe, all related to a unique day (increasing in time).

    :param df: dataframe from which to construct the interaction counts (panda dataframe)
    :param keyword: (OPTIONAL) keyword used to extract the sub-dataset (string)
    :param time_window: (OPTIONAL) time window used to extract the sub-dataset (python list of length 2)

    :return: daily_interactions (panda dataframe)
    """

    # If no time_window given, take whole time range available
    if time_window is None:
        time_window = MAX_TIME_WINDOW

    # Filter dataset by time window
    mask = ((pd.to_datetime(df["TIMESTAMP"]).dt.date >= time_window[0]) &
            (pd.to_datetime(df["TIMESTAMP"]).dt.date <= time_window[1]))
    df = df.loc[mask]

    # Count the number of source interactions per day
    daily_interactions_source = (
        df[df["SOURCE_SUBREDDIT"].str.contains(keyword, case=False, na=False)]
        .groupby(pd.to_datetime(df["TIMESTAMP"]).dt.date)
        .size()
        .rename("DAILY_SOURCE_INTERACTIONS")
    )

    # Count the number of target interactions per day
    daily_interactions_target = (
        df[df["TARGET_SUBREDDIT"].str.contains(keyword, case=False, na=False)]
        .groupby(pd.to_datetime(df["TIMESTAMP"]).dt.date)
        .size()
        .rename("DAILY_TARGET_INTERACTIONS")
    )

    # Create the full date range (daily frequency)
    time_range = pd.date_range(start=time_window[0], end=time_window[1], freq="D")

    # Reindex to include missing days
    daily_interactions_source = daily_interactions_source.reindex(time_range.date, fill_value=0)
    daily_interactions_target = daily_interactions_target.reindex(time_range.date, fill_value=0)

    # Concatenate into one dataframe
    daily_interactions = pd.concat([daily_interactions_source, daily_interactions_target], axis=1)

    # Add total column
    daily_interactions["DAILY_TOTAL_INTERACTIONS"] = (
        daily_interactions["DAILY_SOURCE_INTERACTIONS"] + daily_interactions["DAILY_TARGET_INTERACTIONS"]
    )

    # Convert index to datetime
    daily_interactions.index = pd.to_datetime(daily_interactions.index)
    daily_interactions.index.name = "DAY"

    return daily_interactions

def construct_neg_interaction_df(df, time_window=None, criteria=None, threshold = None, upper=None, groupby_type="DAY"):
    """
    Constructs the negative interaction count (series or) dataframe based on given dataframe, given criterion,
    given thresholds and groupby type (daily, per subreddit, ...).
    It includes as many columns as criteria and as many rows as the input dataframe groupby_type has (number
    of days, number of subreddits, ...). By default, a single criterion is considered
    (LINK_SENTIMENT <= 0) and the whole time window is chosen to construct the dataframe grouped by day.

    :param df: dataframe from which to construct the interaction counts (panda dataframe)
    :param time_window: (OPTIONAL) time window used to extract the sub-dataset (python list of length 2)
    :param criteria: (OPTIONAL) list of criteria on which perform the counts (python list)
    :param threshold: (OPTIONAL) list of threshold for each criterion, upper or lower (python list)
    :param upper: (OPTIONAL) whether if considered thresholds are upper (True) or lower (False) bounds (python list of booleans)
    :param groupby_type: (OPTIONAL) type of grouping (typically "DAY", "SOURCE_SUBREDDIT" or "TARGET_SUBREDDIT", string)

    :return: neg_interaction_df (panda dataframe)
    """

    if criteria is None:
        criteria = ['LINK_SENTIMENT']

    if threshold is None:
        threshold = [0]

    if upper is None:
        upper = [True for _ in range(len(threshold))]

    if not (len(criteria) == len(threshold) and len(criteria) == len(upper)):
        print("Error: Number of criteria doesn't match number of thresholds or upper/lower flags.")
        return None

    if time_window is None:
        time_window = MAX_TIME_WINDOW

    if groupby_type == "DAY":
        groupby_key = pd.to_datetime(df["TIMESTAMP"]).dt.date
        time_range = pd.date_range(start=time_window[0], end=time_window[1], freq="D")
        all_keys = None
    else:
        groupby_key = groupby_type
        all_keys = df[groupby_key].dropna().unique()
        time_range = None

    neg_interaction_df = pd.DataFrame()

    for i in range(len(criteria)):
        upper_sign = 1 if upper[i] else -1
        column_name = f"{criteria[i]}_{threshold[i]}_{'UPPER' if upper[i] else 'LOWER'}"

        series = (
            df[df[criteria[i]] * upper_sign <= upper_sign * threshold[i]]
            .groupby(groupby_key)[criteria[i]]
            .count()
        )

        if time_range is not None:
            series.index = pd.to_datetime(series.index).normalize()
            series = series.reindex(time_range, fill_value=0)
        elif all_keys is not None:
            series = series.reindex(all_keys, fill_value=0)

        neg_interaction_df[column_name] = series

    if time_range is not None:
        neg_interaction_df.index = pd.to_datetime(neg_interaction_df.index)
        neg_interaction_df.index.name = "DAY"
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


def plot_daily_interactions(dataframe, day_interval = None, specifier = ""):
    
    fig, ax = plt.subplots(figsize=(10,4))

    dataframe.plot(ax = ax)

    if(day_interval != None):
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Day")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_title(f"{specifier} Interactions over time")

    ax.set_ylabel(f"Number of {specifier} Interactions")
    plt.tight_layout()
    plt.show()


def plot_top_interactors(interaction_count_df,
                         x_col, y_cols,
                         legend,
                         x_ticklabels,
                         plot_top = 10):
    
    fig, ax = plt.subplots(figsize=(8, 5))

    if(x_col == None):
        interaction_count_df.head(plot_top).plot(
            y=y_cols,
            kind="bar",
            width=0.8,
            ax=ax
        )
    else:
        interaction_count_df.head(plot_top).plot(
            x=x_col,
            y=y_cols,
            kind="bar",
            width=0.8,
            ax=ax
        )

    ax.set_title("Top " + str(plot_top) + " Subreddits by Number of Interactions", fontsize=14)
    ax.set_xlabel("Subreddit", fontsize=12)
    ax.set_ylabel("Number of Interactions", fontsize=12)
    ax.set_xticklabels(x_ticklabels, rotation=45, ha="right")
    ax.legend(legend, title="Interaction Type")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def construct_time_series_df(df, subreddit_names, time_window=None):

    if time_window is None:
        time_window = MAX_TIME_WINDOW

    df = df.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df["DATE"] = df["TIMESTAMP"].dt.date

    focused_df = df[
        (
            df["SOURCE_SUBREDDIT"].isin(subreddit_names) |
            df["TARGET_SUBREDDIT"].isin(subreddit_names)
        ) &
        (df["DATE"] >= time_window[0]) &
        (df["DATE"] <= time_window[1])
    ]

    long_df = pd.concat(
        [
            focused_df[["DATE", "SOURCE_SUBREDDIT"]]
            .rename(columns={"SOURCE_SUBREDDIT": "subreddit"}),

            focused_df[["DATE", "TARGET_SUBREDDIT"]]
            .rename(columns={"TARGET_SUBREDDIT": "subreddit"})
        ],
        axis=0
    )
    long_df = long_df[long_df["subreddit"].isin(subreddit_names)]

    time_series_df = (
        long_df
        .groupby(["DATE", "subreddit"])
        .size()
        .unstack(fill_value=0)
    )

    full_time_index = pd.date_range(
        start=time_window[0],
        end=time_window[1],
        freq="D"
    ).date

    time_series_df = (
        time_series_df
        .reindex(index=full_time_index, fill_value=0)
        .reindex(columns=subreddit_names, fill_value=0)
    )

    return time_series_df


def compute_PCMCI(time_series_df, max_lag=5, alpha=0.05, show=False):
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

    print("Data transformation to tigramite format...")
    # Data transformation to tigramite format
    data_arr = time_series_df.values
    var_names = list(time_series_df.columns)
    tdata = TDataFrame(data_arr, var_names=var_names)

    print("PCMCI from tigramite...")
    # PCMCI from tigramite
    parcorr = RobustParCorr(significance='analytic') # CI type: partial correlation test
    pcmci = PCMCI(dataframe=tdata, cond_ind_test=parcorr)
    results = pcmci.run_pcmci(tau_max=max_lag, pc_alpha=None)  # pc_alpha=None => we will control significance manually

    print("Extracting results...")
    # Extracting results
    p_matrix = results['p_matrix']     # (p-values) N_vars x N_vars x max_lag+1
    val_matrix = results['val_matrix'] # (effect values e.g. partial correlations) N_vars x N_vars x max_lag+1

    print("Extracting significant links...")
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

        # for layout clarity, you may want to only visualize a subset (e.g., protagonist and top-K parents)
        # Optionally filter edges to those involving protagonist:
        # protagonist = 'protagonist'  # replace
        # edges = [e for e in edges if (e['source']==protagonist or e['target']==protagonist)]

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
                G.add_edge(src, tgt, label=lbl, weight=abs(eff))

        # Plot (for readability limit nodes if huge)
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.6, seed=42)  # positions

        # draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, font_size=9)

        # draw edges; use width scaled by 'weight'
        edge_widths = [G[u][v]['weight']*2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=12)

        # draw edge labels (lag + effect)
        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title("PCMCI-discovered causal edges (lagged). Edge labels: lag / effect (ParCorr)")
        plt.axis('off')
        plt.show()

    return p_matrix, val_matrix, edges