# Cyp
# Define functions for computing toxcity metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

property_columns = [
    "NUMBER_OF_CHARACTERS",
    "NUMBER_OF_CHARACTERS_WITHOUT_COUNTING_WHITE_SPACE",
    "FRACTION_OF_ALPHABETICAL_CHARACTERS",
    "FRACTION_OF_DIGITS",
    "FRACTION_OF_UPPERCASE_CHARACTERS",
    "FRACTION_OF_WHITE_SPACES",
    "FRACTION_OF_SPECIAL_CHARACTERS_SUCH_AS_COMMA_EXCLAMATION_MARK_ETC",
    "NUMBER_OF_WORDS",
    "NUMBER_OF_UNIQUE_WORDS",
    "NUMBER_OF_LONG_WORDS_AT_LEAST_6_CHARACTERS",
    "AVERAGE_WORD_LENGTH",
    "NUMBER_OF_UNIQUE_STOPWORDS",
    "FRACTION_OF_STOPWORDS",
    "NUMBER_OF_SENTENCES",
    "NUMBER_OF_LONG_SENTENCES_AT_LEAST_10_WORDS",
    "AVERAGE_NUMBER_OF_CHARACTERS_PER_SENTENCE",
    "AVERAGE_NUMBER_OF_WORDS_PER_SENTENCE",
    "AUTOMATED_READABILITY_INDEX",
    "POSITIVE_SENTIMENT_CALCULATED_BY_VADER",
    "NEGATIVE_SENTIMENT_CALCULATED_BY_VADER",
    "COMPOUND_SENTIMENT_CALCULATED_BY_VADER",
    "LIWC_FUNCT",
    "LIWC_PRONOUN",
    "LIWC_PPRON",
    "LIWC_I",
    "LIWC_WE",
    "LIWC_YOU",
    "LIWC_SHEHE",
    "LIWC_THEY",
    "LIWC_IPRON",
    "LIWC_ARTICLE",
    "LIWC_VERBS",
    "LIWC_AUXVB",
    "LIWC_PAST",
    "LIWC_PRESENT",
    "LIWC_FUTURE",
    "LIWC_ADVERBS",
    "LIWC_PREP",
    "LIWC_CONJ",
    "LIWC_NEGATE",
    "LIWC_QUANT",
    "LIWC_NUMBERS",
    "LIWC_SWEAR",
    "LIWC_SOCIAL",
    "LIWC_FAMILY",
    "LIWC_FRIENDS",
    "LIWC_HUMANS",
    "LIWC_AFFECT",
    "LIWC_POSEMO",
    "LIWC_NEGEMO",
    "LIWC_ANX",
    "LIWC_ANGER",
    "LIWC_SAD",
    "LIWC_COGMECH",
    "LIWC_INSIGHT",
    "LIWC_CAUSE",
    "LIWC_DISCREP",
    "LIWC_TENTAT",
    "LIWC_CERTAIN",
    "LIWC_INHIB",
    "LIWC_INCL",
    "LIWC_EXCL",
    "LIWC_PERCEPT",
    "LIWC_SEE",
    "LIWC_HEAR",
    "LIWC_FEEL",
    "LIWC_BIO",
    "LIWC_BODY",
    "LIWC_HEALTH",
    "LIWC_SEXUAL",
    "LIWC_INGEST",
    "LIWC_RELATIV",
    "LIWC_MOTION",
    "LIWC_SPACE",
    "LIWC_TIME",
    "LIWC_WORK",
    "LIWC_ACHIEV",
    "LIWC_LEISURE",
    "LIWC_HOME",
    "LIWC_MONEY",
    "LIWC_RELIG",
    "LIWC_DEATH",
    "LIWC_ASSENT",
    "LIWC_DISSENT",
    "LIWC_NONFLU",
    "LIWC_FILLER"
]

#############################################################################
# Bar-plot of normlaized mean differences (interactive)
#############################################################################
def evaluate_props_norm_diffs(interactions_df, threshold):

    props_only = interactions_df[property_columns]
    props_mean = props_only.mean()
    props_std = props_only.std()
    props_z = (props_only - props_mean) / props_std

    df_z = pd.concat([interactions_df["LINK_SENTIMENT"], props_z], axis=1)

    avg_z_by_sentiment = df_z.groupby("LINK_SENTIMENT").mean(numeric_only=True).T
    avg_z_by_sentiment.columns = ["negative", "positive"]

    avg_z_diffs = avg_z_by_sentiment["positive"] - avg_z_by_sentiment["negative"]

    # ---- Build labels ----
    short_labels = [f"P{i+1}" for i in range(len(avg_z_diffs))]
    full_labels = avg_z_diffs.index.tolist()
    values = avg_z_diffs.values

    colors = [
        "green" if v > 0 else "darkred"
        for v in values
    ]

    opacity = [
        1.0 if abs(v) > threshold else 0.5
        for v in values
    ]

    # ---- Plotly figure ----
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=short_labels,
        y=values,
        marker=dict(color=colors, opacity=opacity),
        customdata=full_labels,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Property: %{customdata}<br>"
            "Δ z-score: %{y:.3f}<extra></extra>"
        )
    ))

    # Threshold & zero lines
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.add_hline(y=threshold, line_dash="dash", line_color="blue")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue")

    fig.update_layout(
        title="Normalized Difference of Property Means (Positive vs Negative Links)",
        xaxis_title="Property",
        yaxis_title="Δ (z-score difference: Positive − Negative)",
        xaxis_tickangle=-90,
        height=500,
        margin=dict(l=20, r=20, t=40, b=40),
    )

    fig.show()
    fig.write_html("Data/Results_html/property_diffs.html")

    return avg_z_diffs


########################################################################################
# Slice-plot showing absolute weights of retained proprieties (interactive)
########################################################################################
def plot_toxicity_camembert(tox_props):
    # ---- Property -> P_i mapping ----
    prop_to_pi = {
        prop: f"P{i+1}"
        for i, prop in enumerate(property_columns)
    }
    df = tox_props.copy()
    df["P_i"] = df["name"].map(prop_to_pi)
    df["abs_weight"] = df["weight"].abs()
    # ---- Sort: greens first, reds second ----
    df = df.sort_values("weight")
    # ---- Percentage contribution ----
    total_abs = df["abs_weight"].sum()
    df["pct"] = 100 * df["abs_weight"] / total_abs
    # ---- Color mapping ----
    max_abs = df["abs_weight"].max()
    def color_from_weight(w):
        intensity = abs(w) / max_abs
        alpha = 0.4 + 0.6 * intensity
        if w < 0:
            return f"rgba(0, 140, 0, {alpha})"
        else:
            return f"rgba(180, 0, 0, {alpha})"
    colors = df["weight"].apply(color_from_weight)
    # ---- Slice labels ----
    labels = [
        f"{pi}<br>{pct:.1f}%"
        for pi, pct in zip(df["P_i"], df["pct"])
    ]
    # ---- Pie ----
    fig = go.Figure(
        go.Pie(
            values=df["abs_weight"],
            labels=labels,
            marker=dict(
                colors=colors,
                line=dict(color="black", width=1)
            ),
            customdata=np.stack(
                [df["name"], df["weight"]],
                axis=-1
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Weight norm: %{value}<br>"
                "<extra></extra>"
            ),
            hole=0.15,
            sort=False,
            direction="clockwise",
            textinfo="label"
        )
    )
    # ---- Compact layout without title ----
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=10),  # Marges équilibrées
        height=400,  # Même hauteur que le conteneur
        width=550,   # Largeur ajustée
        paper_bgcolor="#fef9e7",  # Fond transparent
        plot_bgcolor="#fef9e7",
        title="Slice plot of absolute values of property weights",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text="Negative weights<br>(positive msgs)",
                x=0.88,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="green", size=11),
                align="center"
            ),
            dict(
                text="Positive weights<br>(negative msgs)",
                x=0.12,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="darkred", size=11),
                align="center"
            )
        ]
    )
    
    # Config pour bloquer toute interaction
    config = {
        'displayModeBar': False, 
        'responsive': False,
        'scrollZoom': False
    }
    
    # Générer le HTML avec style inline
    html_string = fig.to_html(
        config=config,
        include_plotlyjs='cdn'
    )
    
    # Injecter du CSS pour centrer et bloquer overflow
    html_string = html_string.replace(
        '</head>',
        '<style>body, html { margin: 0; padding: 0; overflow: hidden !important; height: 400px; display: flex; align-items: center; justify-content: center; }</style></head>'
    )
    
    with open("weights_slices.html", "w") as f:
        f.write("Data/Results_html/"+html_string)
    fig.show()

########################################################################################
# Plot empirical distributions
########################################################################################
def plot_empirical_distribution(series, name):
    """
    Plot and summarize the empirical distribution of a numeric variable.

    --> This function provides a statistical summary and a visual representation of the
    empirical distribution of a given numeric series. It prints summary statistics, then
    creates two side-by-side histograms — one with a linear y-axis and one with a logarithmic
    y-axis.

    Args:
        series (pd.Series): Numeric data to analyze and visualize.
        name (str): Label for the variable (used for axis titles).

    Returns:
        None
    """

    print(series.describe())

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    fig.suptitle("Empirical distribution of TOXICITY_SCORE", fontsize=13)

    # --- Linear y-axis ---
    ax[0].hist(series, bins=50, density=True,
               alpha=0.6, edgecolor='black')
    ax[0].set_title("Linear y-scale")
    ax[0].set_xlabel(name)
    ax[0].set_ylabel("Density")
    ax[0].grid(axis='y', linestyle='--', alpha=0.6)

    # --- Log y-axis ---
    ax[1].hist(series, bins=50, density=True,
               alpha=0.6, edgecolor='black')
    ax[1].set_title("Log y-scale")
    ax[1].set_xlabel(name)
    ax[1].set_ylabel("Density")
    ax[1].set_yscale("log")
    ax[1].grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.show()



color_nontox = "#7CC0C6"
color_tox = "#C185B6"


########################################################################################
# Cross-count bar-plots
########################################################################################
def cross_count_SENTIMENT_TOXICITY(interactions_df):
    """
    Visualize and analyze the relationship between LINK_SENTIMENT and TOXICITY_CAT.

    This function creates
      1. A grouped bar chart displaying the raw cross-tabulated counts.
      2. A stacked bar chart showing the relative proportions (percentages)
         of toxic vs non-toxic links within each sentiment category.

    Args:
        interactions_df (pd.DataFrame): DataFrame containing the columns:
            - "LINK_SENTIMENT": sentiment labels (e.g., -1 for negative, 1 for positive)
            - "TOXICITY_CAT": binary toxicity indicator (e.g., 0 = non-toxic, 1 = toxic)

    """
    
    cross_counts = pd.crosstab(interactions_df["LINK_SENTIMENT"], interactions_df["TOXICITY_CAT"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [3, 1]})

    colors = [color_nontox, color_tox]
    cross_counts.plot(
        kind='bar',
        width=0.7,
        color=colors,
        edgecolor='black',
        ax=ax[0]
    )

    for container in ax[0].containers:
        ax[0].bar_label(container, label_type='edge', fontsize=10, color='black', padding=3)

    ax[0].set_xlabel("LINK_SENTIMENT")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Cross-counts of LINK_SENTIMENT vs TOXICITY_CAT")
    ax[0].legend(["NON-TOXIC", "TOXIC"])
    ax[0].set_ylim(0, 820000)
    ax[0].set_xticklabels([-1, 1], rotation=0, ha='right')
    ax[0].grid(axis='y')


    negnontox, negtox, posnontox, postox = cross_counts.values.flatten()
    n = negnontox + negtox
    p = posnontox + postox

    nontoxic_percents = [negnontox/n, posnontox/p]
    toxic_percents = [negtox/n, postox/p]

    # Plot

    bars1 = ax[1].bar([-1, 1], nontoxic_percents, color=color_nontox, edgecolor = 'black', width = 1)
    bars2 = ax[1].bar([-1, 1], toxic_percents, bottom=nontoxic_percents, color=color_tox, edgecolor = 'black', width = 1)

    epsilon=1e-6
    for bar, value in zip(bars1, nontoxic_percents):
        ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f'{value*100: .2f}%', 
                ha='center', va='center', color='white', fontsize=10)

    for bar, value, offset in zip(bars2, toxic_percents, nontoxic_percents):
        ax[1].text(bar.get_x() + bar.get_width() / 2, offset + value / 2, f'{value*100: .2f}%', 
                ha='center', va='center', color='white', fontsize=10)

    ax[1].set_xticks([-1, 1])
    ax[1].set_title("Proportion of TOXICITY_CAT")
    ax[1].set_xlabel("LINK_SENTIMENT")
    plt.tight_layout()
    plt.show()

########################################################################################
# Hyperlink activity over time
########################################################################################

def hyperlink_activity_over_time(interactions_df):
    interactions_df["year_month"] = interactions_df["TIMESTAMP"].dt.to_period("M").dt.to_timestamp()

    links_over_time_sent = (
        interactions_df.groupby(["year_month", "TOXICITY_CAT"])
        .size()
        .reset_index(name="n_links")
    )

    links_over_time_pivot = links_over_time_sent.pivot(
        index="year_month", columns="TOXICITY_CAT", values="n_links"
    ).fillna(0)

    links_over_time_pivot["total"] = links_over_time_pivot.sum(axis=1)
    toxicity_counts = interactions_df["TOXICITY_CAT"].value_counts()

    plt.figure(figsize=(12, 6))
    plt.plot(links_over_time_pivot.index, links_over_time_pivot[0], label=f"Non-toxic links \n(total:{toxicity_counts[0]})", color=color_nontox)
    plt.plot(links_over_time_pivot.index, links_over_time_pivot[1], label=f"Toxic links \n(total:{toxicity_counts[1]})", color=color_tox)
    plt.plot(links_over_time_pivot.index, links_over_time_pivot["total"], label="Total links", color="black", alpha = 0.7, linestyle = '--')

    plt.title("Hyperlink Activity Over Time - log-scale y")
    plt.xlabel("Month")
    plt.ylabel("Nb of links")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()