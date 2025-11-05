import pandas as pd
import numpy as np

# ===== CONSTANT DEFINITIONS =====

TOXICITY_THRESHOLD = 0.2

TOX_PROPS = pd.DataFrame([
    [-3, "FRACTION_OF_DIGITS"],
    [0, "LINK_SENTIMENT_ADJUST"],
    [1, "FRACTION_OF_STOPWORDS"],
    [5, "NEGATIVE_SENTIMENT_CALCULATED_BY_VADER"],
    [-5, "LIWC_CAUSE"],
    [-5, "LIWC_INSIGHT"],
    [5, "LIWC_SWEAR"],
    [1, "LIWC_SOCIAL"],
    [1, "LIWC_HUMANS"],
    [0, "LIWC_AFFECT"],
    [1, "LIWC_NEGEMO"],
    [5, "LIWC_ANGER"],
    [1, "LIWC_SEXUAL"]
    ],
    columns = ["weight", "name"])

# Normalize weights
TOX_PROPS["weight (normd)"] = TOX_PROPS["weight"] / TOX_PROPS["weight"].abs().sum()

# ===== DATA VERIFICATION FUNCTION =====


def verify_data_format(data, time_window):
    """
    Verify that the input DataFrame contains all necessary columns for toxicity computations.
    If it is not the case, we format the DataFrame accordingly.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame to be verified.

    Return:
    data (pd.DataFrame): The input DataFrame with all necessary columns.
    """
    # First, we ensure that TOXICITY_SCORE exists
    if "TOXICITY_SCORE" not in data.columns:
        data = compute_toxicity_score(data)
    # Then, we ensure that TIMESTAMP is datetime if present
    if "TIMESTAMP" in data.columns:
        data = data.copy()
        data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
    else:
        # If no timestamp available, only overall makes sense
        if time_window != "overall":
            raise ValueError("TIMESTAMP column not found in data, so only 'overall' time_window is supported")
    
    return data


# ===== INTENSITY METRICS =====


def compute_toxicity_score(data):
    """
    This function computes the toxicity_intesity metric for each entry (row) in the provided 
    DataFrame 'data'.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Hyperlinks.

    Returns:
    data (pd.DataFrame): The input DataFrame with an additional column 'TOXICITY_SCORE' 
    representing the computed toxicity intensity for each entry.
    """    
    data["LINK_SENTIMENT_ADJUST"] = (1 - (data["LINK_SENTIMENT"] + 1)/2)/8
    data["TOXICITY_SCORE"] = (data[TOX_PROPS["name"].to_list()].values * \
                              TOX_PROPS["weight (normd)"].to_list()).sum(axis = 1)
    
    return data


def compute_base_toxicity_offset(data, time_window="overall"):
    """
    This function computes the base toxicity offset metric (mean toxixicity_inteisity over 
    the entire dataset over the entire timeperiod).

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interations Hyperlinks.

    Returns:
    float or series(pd.Series): The computed base toxicity offset values.
    """
    # First, we ensure the data contains all the necessary columns
    verify_data_format(data, time_window)

    # Dispatch on time_window
    # - 'overall' -> return a single float (global mean)
    # - 'monthly' -> return a Series indexed by month (PeriodIndex converted to Timestamp)
    # - 'yearly' -> return a Series indexed by year
    # - any pandas offset string like '30D','90D','7D' -> return a rolling mean Series (indexed by timestamp)
    tw = str(time_window).lower()
    match tw:
        case "overall":
            return data["TOXICITY_SCORE"].mean()

        case "monthly" | "m":
            # the following line groups the data per month and computes the mean toxicity score for each month
            series = data.set_index("TIMESTAMP").resample("M")["TOXICITY_SCORE"].mean()
            return series

        case "yearly" | "y" | "annual":
            # the following line groups the data per year and computes the mean toxicity score for each year
            series = data.set_index("TIMESTAMP").resample("Y")["TOXICITY_SCORE"].mean()
            return series
        
        case _:
            pass

    # If user provided a pandas offset string for a rolling window (e.g. '30D', '90D')
    # perform a time-based rolling mean. Data must be indexed by TIMESTAMP.
    # Accept offsets that end with common frequency letters.
    if isinstance(time_window, str) and time_window.upper().endswith(("D", "W", "M", "Y")):
        df = data.sort_values("TIMESTAMP").set_index("TIMESTAMP")
        # the following line computes the rolling mean toxicity score over the specified time window
        rolling_mean = df["TOXICITY_SCORE"].rolling(time_window).mean()
        return rolling_mean

    raise ValueError(f"Unsupported time_window: {time_window}")


def compute_toxicity_std(data, time_window="overall"):
    """
    This function computes the base toxicity standard deviation (std toxixicity_inteisity over 
    the entire dataset over the entire timeperiod).

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.

    Returns:
    float or pd.Series: The computed base toxicity offset values.
    """
    # First, we ensure the data contains all the necessary columns
    verify_data_format(data, time_window)

    # Dispatch on time_window
    # - 'overall' -> return a single float (global std)
    # - 'monthly' -> return a Series indexed by month (PeriodIndex converted to Timestamp)
    # - 'yearly' -> return a Series indexed by year
    # - any pandas offset string like '30D','90D','7D' -> return a rolling std Series (indexed by timestamp)
    tw = str(time_window).lower()
    match tw:
        case "overall":
            return data["TOXICITY_SCORE"].std()

        case "monthly" | "m":
            # the following line groups the data per month and computes the std toxicity score for each month
            series = data.set_index("TIMESTAMP").resample("M")["TOXICITY_SCORE"].std()
            return series
        case "yearly" | "y" | "annual":
            # the following line groups the data per year and computes the std toxicity score for each year
            series = data.set_index("TIMESTAMP").resample("Y")["TOXICITY_SCORE"].std()
            return series
    
        case _:
            pass

    # If user provided a pandas offset string for a rolling window (e.g. '30D', '90D')
    # perform a time-based rolling std. Data must be indexed by TIMESTAMP.
    # Accept offsets that end with common frequency letters.
    if isinstance(time_window, str) and time_window.upper().endswith(("D", "W", "M", "Y")):
        df = data.sort_values("TIMESTAMP").set_index("TIMESTAMP")
        # the following line computes the rolling std toxicity score over the specified time window
        rolling_std = df["TOXICITY_SCORE"].rolling(time_window).std()
        return rolling_std

    raise ValueError(f"Unsupported time_window: {time_window}")


def compute_toxicity_context_zscore(data, time_window="overall"):
    """
    This function computes the toxicity z-score metric within a given time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for computation (note that here we can only use 'overall', 'monthly', 'yearly')

    Returns:
    pd.Series: The computed toxicity z-score values for each row in the dataset.
    """
    verify_data_format(data, time_window)
    
    ti_series = data["TOXICITY_SCORE"].astype(float)

    base_toxicity_offset = compute_base_toxicity_offset(data, time_window)
    toxicity_std = compute_toxicity_std(data, time_window)

    # Handle the case where mean/std are scalars
    if pd.api.types.is_scalar(base_toxicity_offset) and pd.api.types.is_scalar(toxicity_std):
        # To handle the case of zero standard deviation
        if toxicity_std == 0:
            return pd.Series(np.nan, index=ti_series.index)
        result = (ti_series - base_toxicity_offset) / toxicity_std
        return result
    
    # We need to align each row's timestamp to the corresponding period mean/std
    if "TIMESTAMP" in data.columns:
        timestamps = pd.to_datetime(data["TIMESTAMP"], errors="coerce")
        tw = str(time_window).lower()
        
        # Map each row to its period (month or year)
        match tw:
            case "monthly" | "m":
                # Convert timestamps to month-end periods for alignment
                # First we convert each datetime to a period object representing the month
                # (e.g., 2023-06-15 -> 2023-06) then we convert it back to a timestamp at month-end
                # (e.g., 2023-06 -> 2023-06-30)
                period_idx = timestamps.dt.to_period('M').dt.to_timestamp('M')
            case "yearly" | "y" | "annual":
                # Convert timestamps to year-end periods for alignment
                period_idx = timestamps.dt.to_period('Y').dt.to_timestamp('Y')
            case _:
                raise ValueError(f"Unsupported time_window for z-score with Series: {time_window}")
        
        # We align the mean and std Series to each row using the period index
        # This creates an array where each entry corresponds to the mean/std of the period
        # (e.g., for monthly, all rows in June 2023 get the mean/std of June 2023)
        mean_aligned = base_toxicity_offset.reindex(period_idx).values
        std_aligned = toxicity_std.reindex(period_idx).values
        
        std_aligned = np.where(std_aligned == 0, np.nan, std_aligned)
        
        # Finally, we compute z-scores element-wise
        result_vals = (ti_series.values - mean_aligned) / std_aligned
        return pd.Series(result_vals, index=ti_series.index)
    
    raise ValueError("TIMESTAMP column required for non-overall time_window")
    

# ===== COUNT METRICS =====


def compute_toxic_count(data, time_window="overall"):
    """
    This function counts messages classified as toxic within a given time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting

    Returns:
    int or pd.Series: Number of toxic messages for the requested scope.
    """
    # First, we ensure the data contains all the necessary columns
    verify_data_format(data, time_window)

    tw = str(time_window).lower()
    match tw:
        case "overall":
            return int((data["TOXICITY_SCORE"] >= TOXICITY_THRESHOLD).sum())

        case "monthly" | "m":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].gt(TOXICITY_THRESHOLD).resample("M").sum()
            return series.astype(int)

        case "yearly" | "y" | "annual":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].gt(TOXICITY_THRESHOLD).resample("Y").sum()
            return series.astype(int)

        case _:
            pass

    if isinstance(time_window, str) and time_window.upper().endswith(("D", "W", "M", "Y")):
        df = data.sort_values("TIMESTAMP").set_index("TIMESTAMP")
        rolling_counts = df["TOXICITY_SCORE"].gt(TOXICITY_THRESHOLD).rolling(time_window).sum()
        return rolling_counts.fillna(0).astype(int)

    raise ValueError(f"Unsupported time_window: {time_window}")


def compute_non_toxic_count(data, time_window="overall"):
    """
    This function counts messages classified as NON toxic within a given time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting

    Returns:
    int or pd.Series: Number of NON toxic messages for the requested scope.
    """
    # First, we ensure the data contains all the necessary columns
    verify_data_format(data, time_window)

    tw = str(time_window).tolower()
    match tw:
        case "overall": 
            return int((data["TOXICITY_SCORE"] < TOXICITY_THRESHOLD).sum())
        
        case "monthly" | "m":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].lt(TOXICITY_THRESHOLD).resample("M").sum()
            return series
        
        case "yearly" | "y" | "annual":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].lt(TOXICITY_THRESHOLD).resample("Y").sum()
            return series
        
        case _:
            pass

    if isinstance(time_window, str) and time_window.upper().endswith(("D", "W", "M", "Y")):
        df = data.sort_values("TIMESTAMP").set_index("TIMESTAMP")
        rolling_counts = df["TOXICITY_SCORE"].gt(TOXICITY_THRESHOLD).rolling(time_window).sum()
        return rolling_counts.fillna(0).astype(int)

    raise ValueError(f"Unsupported time_window: {time_window}")


def compute_total_count(data, time_window):
    """
    This function counts the total number of messages within a given time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting

    Returns:
    int or pd.Series: Number of messages for the requested scope.
    """
    # First, we ensure the data contains all the necessary columns
    verify_data_format(data, time_window)

    tw = str(time_window).tolower()
    match tw:
        case "overall":
            return int(data["TOXICITY_SCORE"].sum())
        
        case "monthly" | "m":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].resample("M").sum()
            return series
        
        case "yearly" | "y" | "annual":
            series = data.set_index("TIMESTAMP")["TOXICITY_SCORE"].resample("Y").sum()
            return series
        
        case _:
            pass

    if isinstance(time_window, str) and time_window.toupper().endswith(("D", "W", "M", "Y")):
        df = data.sort_values["TIMESTAMP"].set_index["TIMESTAMP"]
        rolling_counts = df["TOXICITY_SCORE"].rolling(time_window).sum()
        return rolling_counts.fillna(0).astype(int)
    
    raise ValueError(f"Unsupported time_window: {time_window}")


# ===== RATE-BASED METRICS =====


def compute_toxicity_rate(data, time_window):
    """
    This function computes the proportion of toxic source messages in a time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for computation

    Returns:
    int or pd.Series: Proportion of toxic source messages.
    """
    toxic_count = compute_toxic_count(data, time_window).copy()
    total_count = compute_total_count(data, time_window)    
    
    # Here we handle the case where total_count is zero or contains zeroes
    if isinstance(total_count, int):
        if total_count == 0:
            total_count = np.nan
    if isinstance(total_count, pd.Series):        
            total_count = total_count.replace(0, np.nan)
    
    return toxic_count / total_count


def compute_toxicity_odds(data, time_window):
    """
    This function computes the odds of toxic source messages in a time scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for computation

    Returns:
    int or pd.Series: Odds of toxic source messages.
    """
    toxic_count = compute_toxic_count(data, time_window).copy()
    non_toxic_count = compute_non_toxic_count(data, time_window).copy()
    
    # Here we handle the case where total_count is zero or contains zeroes
    if isinstance(non_toxic_count, int):
        if non_toxic_count == 0:
            non_toxic_count = np.nan
    if isinstance(non_toxic_count, pd.Series):        
            non_toxic_count = non_toxic_count.replace(0, np.nan)
    
    return toxic_count / non_toxic_count


# ===== ADVANCED LINGUISTIC METRICS =====


def compute_emotional_volatility():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


def compute_personal_attack_indicators():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


def compute_rational_discourse_score():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


def compute_complexity_metrics():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


# ===== DIRECTINAL METRICS =====


def compute_toxicity_out(data, time_window, subreddit_name):
    """
    This function counts the number of toxic links sent by a specfic subreddit in a time
    scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting
    subreddit_name(str): Name of the subreddit sending the toxic links

    Returns:
    int or pd.Series: Count of toxic links sent in a time scope.
    """
    verify_data_format(data, time_window)

    # We make sure subreddit_name is a string
    if isinstance(subreddit_name, str):
        specific_subreddit_data = data[data["SOURCE_SUBREDDIT"] == subreddit_name]
        toxic_links_count = compute_toxic_count(specific_subreddit_data, time_window)
        return toxic_links_count
    else:
        raise ValueError("subreddit_name must be a string representing the subreddit name")



def compute_toxicity_in(data, time_window, subreddit_name):
    """
    This function counts the number of toxic links received by a specfic subreddit in a time
    scope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting
    subreddit_name(str): Name of the subreddit receiving the toxic links

    Returns:
    int or pd.Series: Count of toxic links received in a time scope.
    """
    verify_data_format(data, time_window)

    # We make sure subreddit_name is a string
    if isinstance(subreddit_name, str):
        specific_subreddit_data = data[data["TARGET_SUBREDDIT"] == subreddit_name]
        toxic_links_count = compute_toxic_count(specific_subreddit_data, time_window)
        return toxic_links_count
    else:
        raise ValueError("subreddit_name must be a string representing the subreddit name")


def compute_toxicity_balance(data, time_window, subreddit_name):
    """
    This function computes the net difference between the number of sent and received by a specfic
    subreddit toxic links in a timescope.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting
    subreddit_name(str): Name of the subreddit for which to compute the toxicity balance

    Returns:
    int or pd.Series: Count of toxic balances
    """
    verify_data_format(data, time_window)

    # We make sure subreddit_name is a string
    if isinstance(subreddit_name, str):
        sent_toxic_count = compute_toxicity_out(data, time_window, subreddit_name)
        received_toxic_count = compute_toxicity_in(data, time_window, subreddit_name)
        toxicity_balance = sent_toxic_count - received_toxic_count
        return toxicity_balance
    else:
        raise ValueError("subreddit_name must be a string representing the subreddit name")


def compute_reciprocal_toxicity(data, time_window):
    """
    This function counts the number of bidirectional (reciprocal) edges between subreddit pairs
    in the toxic interaction graph within a time scope.
    N.B.: A bidirectional edge exists when both A→B and B→A toxic links exist between two subreddits.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the Reddit Interactions Hyperlinks.
    time_window(str): Scope for counting

    Returns:
    int or pd.Series: Count of bidirectional edges in the specified time scope.
    """
    verify_data_format(data, time_window)

    def _count_bidirectional_edges(df):
        """
        Helper function to count bidirectional edges in a DataFrame.
        
        Returns the number of subreddit pairs with mutual toxic links.
        """
        toxic_df = df[df["TOXICITY_SCORE"] >= TOXICITY_THRESHOLD]
        
        # Create directed edges as (source, target) tuples
        edges = set(
            (row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"])
            for _, row in toxic_df.iterrows()
        )
        
        # Count pairs where both (A,B) and (B,A) exist
        bidirectional = sum(
            1 for (src, tgt) in edges 
            if (tgt, src) in edges and src < tgt  # src < tgt avoids double counting
        )
        
        return bidirectional

    tw = str(time_window).lower()
    match tw:
        case "overall":
            return _count_bidirectional_edges(data)
        
        case "monthly" | "m":
            series = data.set_index("TIMESTAMP").resample("M").apply(_count_bidirectional_edges)
            return series.astype(int)
        
        case "yearly" | "y" | "annual":
            series = data.set_index("TIMESTAMP").resample("Y").apply(_count_bidirectional_edges)
            return series.astype(int)
        
        case _:
            pass

    raise ValueError(f"Unsupported time_window: {time_window}")


# ===== TEMPORAL METRICS =====

def compute_toxicity_velocity():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


def compute_toxicity_acceleration():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0


def compute_toxicity_persistence():
    # TO BE IMPLEMETED FOR MILESTONE 3
    return 0