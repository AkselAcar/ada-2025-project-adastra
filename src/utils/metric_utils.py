# Cyp
# Define functions for computing toxcity metrics

import pandas as pd


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

def evaluate_props_norm_diffs(interactions_df):

    # Normalize each property across all links
    props_only = interactions_df[property_columns]
    props_mean = props_only.mean()
    props_std = props_only.std()

    # Compute z-scores
    props_z = (props_only - props_mean) / props_std

    # Reattach sentiment
    df_z = pd.concat([interactions_df["LINK_SENTIMENT"], props_z], axis=1)

    # Compute mean z-score per sentiment
    avg_z_by_sentiment = df_z.groupby("LINK_SENTIMENT").mean(numeric_only=True).T
    avg_z_by_sentiment.columns = ["negative", "positive"]

    # Compute difference (positive - negative)
    avg_z_by_sentiment["diff"] = avg_z_by_sentiment["positive"] - avg_z_by_sentiment["negative"]

    return avg_z_by_sentiment["diff"]