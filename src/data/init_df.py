# Import panda
import pandas as pd


# =========================== Load datasets
DATA_FOLDER = '../Data/'

HYPERLINKS_BODY_DATASET = DATA_FOLDER+"soc-redditHyperlinks-body.tsv"
HYPERLINKS_TITLE_DATASET = DATA_FOLDER+"soc-redditHyperlinks-title.tsv"
EMBEDDINGS_SUBREDDITS_DATASET = DATA_FOLDER+"web-redditEmbeddings-subreddits.csv"
EMBEDDINGS_USERS_DATASET = DATA_FOLDER+"web-redditEmbeddings-users.csv"

body_df = pd.read_csv(HYPERLINKS_BODY_DATASET, sep="\t")
title_df = pd.read_csv(HYPERLINKS_TITLE_DATASET, sep="\t")
embeddings_subreddits_df = pd.read_csv(EMBEDDINGS_SUBREDDITS_DATASET, header=None)
embeddings_users_df = pd.read_csv(EMBEDDINGS_USERS_DATASET, header=None)


# =========================== Expand properties into columns (use UPPER_SNAKE_CASE convention)
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
def expand_properties(df):
    props_expanded = df["PROPERTIES"].str.split(",", expand=True)
    props_expanded.columns = property_columns
    props_expanded = props_expanded.apply(pd.to_numeric, errors='coerce')
    df = pd.concat([df.drop(columns=["PROPERTIES"]), props_expanded], axis=1)
    return df

body_df = expand_properties(body_df)
title_df = expand_properties(title_df)


# =========================== Concatenate body and title
body_df["TYPE"] = "body"
title_df["TYPE"] = "title"
interactions_df = pd.concat([body_df, title_df], ignore_index=True)


# =========================== Standardize Subreddit names
interactions_df["SOURCE_SUBREDDIT"] = interactions_df["SOURCE_SUBREDDIT"].astype(str).str.lower()
interactions_df["TARGET_SUBREDDIT"] = interactions_df["TARGET_SUBREDDIT"].astype(str).str.lower()


# =========================== Expand timestamp into day, week, month and year
interactions_df["TIMESTAMP"] = pd.to_datetime(interactions_df["TIMESTAMP"])

# Reorder the interactions_df chronologically
interactions_df = interactions_df.sort_values('TIMESTAMP', ascending=True).reset_index(drop=True)

#interactions_df["YEAR"] = interactions_df["TIMESTAMP"].dt.year
#interactions_df["WEEK"] = interactions_df["TIMESTAMP"].dt.isocalendar().week
#interactions_df["MONTH"] = interactions_df["TIMESTAMP"].dt.month
#interactions_df["DAY"] = interactions_df["TIMESTAMP"].dt.day


# =========================== Formate embeddings dataset and number them
embeddings_subreddits_df.columns = ["SUBREDDIT"] + [f"emb_{i}" for i in range(embeddings_subreddits_df.shape[1] - 1)]
embeddings_users_df.columns = ["USER"] + [f"emb_{i}" for i in range(embeddings_users_df.shape[1] - 1)]


# =========================== Check data quality (missing values, timestamp parsing, duplicate rows, overlap)
def check_quality(df):
    print(f"\n=== QUALITY REPORT ===")
    print(f"Shape: {df.shape}")
    print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))
    print("\nDuplicate rows:", df.duplicated().sum())
    #if "TIMESTAMP" in df.columns:
        #df["DATETIME"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        #print("Unparseable timestamps:", df["DATETIME"].isna().sum())


def function_test_autoreload():
    print("Autoload test successful.")