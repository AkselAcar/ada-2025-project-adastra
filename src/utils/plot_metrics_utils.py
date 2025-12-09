# Cyp
# Define functions for computing toxcity metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO: property_columns was already defined in data so not good to define twice
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

def evaluate_props_norm_diffs(interactions_df, threshold):
    """
    Evaluate and visualize normalized property differences by link sentiment.

    --> Compares a set of numeric property columns between positive and negative links
    (as indicated by the `LINK_SENTIMENT` column). Each property is z-score normalized across
    all links, and then the mean z-score is computed separately for positive and negative
    sentiment groups. The function plots the difference between these two means and highlights
    properties whose normalized difference exceeds a specified threshold.

    Args:
        interactions_df (pd.DataFrame): Input dataframe containing `LINK_SENTIMENT` and property columns.
        threshold (float): Z-score difference threshold to flag significant properties.

    Returns:
        pd.Series: A Series of normalized mean differences (positive − negative) for each property.
    """

    # Normalize each property across all links
    props_only = interactions_df[property_columns]

    # Select numeric columns
    num_df = props_only.select_dtypes(include='number')

    props_stats = pd.DataFrame({
        'min': num_df.min(),
        'median': num_df.median(),
        'max': num_df.max()
    }).T  # transpose so rows are min/median/max

    # Display all columns without truncation
    display(props_stats)

    props_mean = props_only.mean()
    props_std = props_only.std()
    props_z = (props_only - props_mean) / props_std

    df_z = pd.concat([interactions_df["LINK_SENTIMENT"], props_z], axis=1)

    # Compute mean z-score per sentiment
    avg_z_by_sentiment = df_z.groupby("LINK_SENTIMENT").mean(numeric_only=True).T
    avg_z_by_sentiment.columns = ["negative", "positive"]

    avg_z_diffs = avg_z_by_sentiment["positive"] - avg_z_by_sentiment["negative"]

   # Plot normalized differences

    plt.figure(figsize=(18, 6))

    for i, (prop, diff) in enumerate(avg_z_diffs.items()):
        plt.bar(
            prop,
            diff,
            color = "green" if diff > 0 else "darkred",
            alpha = 1 if abs(diff) > threshold else 0.5
        )
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(threshold, color="blue", linestyle = "--")
    plt.axhline(-threshold, color="blue", linestyle = "--")
    plt.xticks(np.arange(len(avg_z_diffs)), ["P" + str(i + 1) for i in np.arange(len(avg_z_diffs))])
    plt.xticks(rotation=90)
    plt.title("Normalized Difference of Property Means (Positive vs Negative Links)", fontsize = 14)
    plt.ylabel("Δ (z-score difference: Positive − Negative)")
    plt.xlabel("Property")
    plt.tight_layout()
    plt.grid()
    plt.show()

    
    # Print properties with high differences

    high_props = avg_z_diffs[avg_z_diffs > threshold].index.tolist()
    low_props = avg_z_diffs[avg_z_diffs < -threshold].index.tolist()

    print(f"Found {len(high_props)} properties higher for positive links (zΔ > {threshold}):")
    for i in high_props:
        print(f"- (z_diff = {round(avg_z_diffs[i], 3)}) \t {i}")

    print(f"\n Found {len(low_props)} properties higher for negative links (zΔ < -{threshold}):")
    for i in low_props:
        print(f"- (z_diff = {round(avg_z_diffs[i], 3)}) \t {i}")

    return avg_z_diffs



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
