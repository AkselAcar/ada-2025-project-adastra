# ADA CS-(402) COURSE PROJECT P2 - Team ADAstra
# Treating Toxicity as an Infectious Disease

**Team Members: Aksel Acar, Emilien Coudurier, Nathan Tabet, Cyprien Tordo, Cedric Zanou**

## PROJECT OVERVIEW
While networks of the web are not physical, it turns out they can closely resemble physical ones, and from their hyper-connected nature, they tend to share their spreading behaviour with diseases. We aim to conduct an epidemiological analysis of toxicity on Reddit in this project, treating toxic behavior as an infectious disease that spreads through social networks.
Recent related works have been conducted, especially on identifying toxicity in social media [1][2] and intercommunity interactions [3], and we intend to analyze how toxicity thrives based on their contribution. From birth to death and through propagation, we treat it like a living organism as a way to highlight temporal patterns and potentially predict its outbreaks, among different communities.

## CORE RESEARCH ANGLES

Among others, we aim to address the following research questions:
### What conditions need to be met for toxicity to appear?
- Do specific time periods stand out for breakouts? 
- Do patient-zero communities exist?
### How does toxicity spread through the network? 
- Do specific contagion patterns stand out (cascades, toxic chains)?
- Does toxicity mutate (i.e., changes in form)?
- What key actions manage to restrain or stop propagation?
### Does toxicity follow cyclical patterns?
- Do global rhythms and trends stand out?
- Can we classify trajectory patterns?
- Can we predict toxicity outcomes from early behavioral signals?


## METHODS

### Preprocessing
We have performed the following preprocessing steps in order to prepare our data for analysis: 
We began by loading the TSV files containing the subreddit interactions and expanded the PROPERTIES column into 86 individual LIWC and sentiment features. This enabled us to have more comprehensive data and to perform analysis more easily. We also created a unified dataset by merging title and body links, adding a `TYPE` column to distinguish between them. In our case, distinguishing both is not crucial for the analyses we intend to perform, thus we opted for simplicity by merging them. We standardized subreddit names (lowercase, whitespace removal) and converted timestamps to datetime format for temporal analysis. Finally, we performed data quality checks for missing values, timestamp parsing errors, and duplicate entries, documenting any dataset overlap between title and body links.

### 1. Toxicity Metric Definition and Computation
Since the dataset does not provide a direct toxicity score, we constructed a composite `toxicity_intensity` metric by combining multiple linguistic and sentiment features available in the data. The formula weights five key components: `toxicity_intensity = α×LIWC_Anger + β×LIWC_Swear + γ×LIWC_Negemo + δ×Negative_VADER + ε×LINK_SENTIMENT`, where coefficients (α, β, γ, δ, ε) are optimized through exploratory analysis to maximize discriminative power between clearly toxic and non-toxic interactions. We computed platform-wide statistics to establish `base_toxicity_offset` (mean intensity) and `toxicity_std` (standard deviation), allowing us to standardize scores as z-scores: `toxicity_zscore = (toxicity_intensity - base_toxicity_offset) / toxicity_std`. Individual messages are classified as toxic when `toxicity_intensity > base_toxicity_offset + k×toxicity_std` (with k=1 as the default threshold, capturing approximately 16% most toxic messages). We then aggregated these binary classifications at multiple scopes (subreddit-level, subreddit-pair level, and temporal windows) to compute derived metrics including `toxicity_rate` (proportion of toxic messages), directional metrics (`toxicity_out`, `toxicity_in`, `toxicity_balance`), and temporal dynamics (`toxicity_velocity`, `toxicity_acceleration`). Additional linguistic metrics such as `emotional_volatility` (standard deviation of toxicity intensity), `personal_attack_indicators` (LIWC_You + LIWC_Negemo combinations), and `rational_discourse_score` (LIWC_CogMech + LIWC_Cause) were computed to characterize the nature of toxic discourse in different communities.

### 2. Temporal Network Construction and Community Detection
For each time window (monthly granularity, as finer temporal resolution proved unreasonable given the data sparsity), we constructed directed weighted graphs where nodes represent subreddits and edges represent interactions, with edge weights determined by interaction frequency and average toxicity intensity. We computed standard network centrality measures (degree, betweenness, closeness, eigenvector centrality) for each node to characterize subreddit positions within the interaction network. To identify communities of subreddits that interact frequently, we will apply the Louvain modularity maximization algorithm to each temporal snapshot, allowing us to track how community structure evolves over time. For each detected community, we will compute baseline toxicity profiles (mean toxicity_rate, intensity distributions) and linguistic fingerprints (dominant LIWC categories) to characterize their typical behavior. We will also measure semantic similarity between subreddits using cosine distances in the provided embedding space to understand whether toxic interactions preferentially occur between semantically similar or dissimilar communities.

### 3. Outbreak Detection and Cascade Tracing
We define a toxicity outbreak as occurring when a subreddit's `toxicity_rate` exceeds its historical baseline by more than 2 standard deviations for at least 2 consecutive time periods, representing a ≥50% increase from baseline behavior. To identify "patient zero" communities that initiate cascades of toxicity, we implement a temporal backtracking algorithm that traces toxic interactions backwards from outbreak peaks, scoring candidate sources by: (1) temporal priority (earlier onset receives higher scores with exponential decay), (2) network position (high toxicity_out/toxicity_in ratio), and (3) correlation between the candidate's activity and downstream toxicity spikes. For cascade identification, we define a toxic cascade as a temporally ordered sequence of directed toxic links C: s₁ →(t₁) s₂ →(t₂) s₃ →(t₃) ... →(tₙ) sₙ, where each target subreddit exhibits increased toxicity within Δt time (1-4 weeks) after receiving a toxic link from the source. We recursively trace these chains until no further propagation is detected, computing cascade metrics including length, duration, branching factor, and semantic drift (embedding distance between first and last nodes). To quantify "mutations" in toxic behavior as it propagates, we will compare LIWC profiles at each cascade step and measure vocabulary shifts between communities.

### 4. Temporal Pattern Analysis and Trajectory Classification
To detect periodicities in toxicity, we will apply Fast Fourier Transform (FFT) to subreddit-level toxicity_rate time series, identifying dominant frequencies and converting them to period estimates (weekly, monthly, annual cycles). Statistical significance of detected periodicities will be assessed by comparing observed power spectra to those generated from randomly permuted time series. We will correlate toxicity spikes (z-score > 3) with a curated timeline of major real-world events (2013-2017) to test whether platform-wide or community-specific toxicity increases follow external triggers. For life cycle analysis, we will extract smoothed toxicity trajectories for each subreddit and apply hierarchical clustering with Dynamic Time Warping (DTW) distance to group similar evolution patterns. We anticipate identifying archetypal trajectories such as chronic toxicity (sustained high levels), acute outbreaks (sharp spikes with recovery), gradual radicalization (slow increase), and intermittent episodes. To build an early warning system, we will extract features from the first N weeks of each subreddit's activity (early toxicity_velocity, network centrality, linguistic markers, embedding-based similarity to known toxic communities) and train logistic regression or random forest classifiers to predict which communities will develop chronic toxicity, validating predictions on a temporal holdout set (train on 2013-2015, test on 2016-2017).

## REFERENCES

[1] VAIDYA, Aatman, NAGAR, Seema, et NANAVATI, Amit A. Analysing the Spread of Toxicity on Twitter. In : Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD). 2024. p. 118-126.
[2] SHETH, Amit, SHALIN, Valerie L., et KURSUNCU, Ugur. Defining and detecting toxicity on social media: context and knowledge are key. Neurocomputing, 2022, vol. 490, p. 312-318.
[3] KUMAR, Srijan, HAMILTON, William L., LESKOVEC, Jure, et al. Community interaction and conflict on the web. In : Proceedings of the 2018 world wide web conference. 2018. p. 933-943.