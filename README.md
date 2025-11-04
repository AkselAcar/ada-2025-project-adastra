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

## ADDITIONAL DATASETS 

## METHODS

### Preprocessing
We have performed the following preprocessing steps in order to prepare our data for analysis: we began by loading the TSV files containing the subreddit interactions and expanded the PROPERTIES column into 86 individual LIWC and sentiment features. This enabled us to have more comprehensive data and to perform analysis more easily. Also, for simplicity, we created a unified dataset by merging title and body links, adding a `TYPE` column to distinguish them. We standardized subreddit names (lowercase, whitespace removal) and converted timestamps to datetime format for temporal analysis. Finally, we performed data quality checks for missing values, timestamp parsing errors, and duplicate entries, documenting any dataset overlap between title and body links.

### 1. Toxicity Metric Definition and Computation
Since the dataset does not provide a direct toxicity score but rather a negativity score, we constructed a composite `toxicity_intensity` metric by combining multiple linguistic and sentiment features available in the data. The formula weights five key components: `toxicity_intensity = α×LIWC_Anger + β×LIWC_Swear + γ×LIWC_Negemo + δ×Negative_VADER + ε×LINK_SENTIMENT`, where coefficients (α, β, γ, δ, ε) are optimized through exploratory analysis to maximize discriminative power between clearly toxic and non-toxic interactions. From this metric, we derive several other detailed in results.ipynb.

### 2. Temporal Network Construction and Community Detection
For each monthly time window (finer resolution proved impractical given data sparsity), we constructed directed weighted graphs with subreddits as nodes and interactions as edges, weighted by interaction frequency and average toxicity intensity. We computed standard centrality measures to characterize subreddit network positions. To identify community structures, we will apply the Louvain algorithm to each temporal snapshot and compute baseline toxicity profiles and linguistic fingerprints (dominant LIWC categories) for each detected community. We will also measure semantic similarity between subreddits using cosine distances in the embedding space to determine whether toxic interactions occur preferentially between semantically similar or dissimilar communities.

### 3. Outbreak Detection and Cascade Tracing
We detect outbreaks when a subreddit's toxicity rate jumps above its normal baseline for a prefixed number of consecutive time periods. To find likely "patient zero" communities that initiate these outbreaks, we trace toxic interactions backward from outbreak peaks and prioritize candidate sources that acted earlier: subreddits that sent toxic links before the outbreak receive higher timing scores. We also look at whether they send more toxic links than they receive, and how their activity aligns with downstream spikes. A toxic cascade is a time-ordered chain of toxic links where targets show increased toxicity shortly after exposure. We follow each chain until propagation stops and summarize cascades by length, duration, branching, and changes in language.

### 4. Temporal Pattern Analysis and Trajectory Classification
To detect periodicities in toxicity, we will apply FFT to subreddit-level toxicity_rate time series, identifying dominant frequencies (monthly, annual cycles) and assessing statistical significance against randomly permuted time series. We will correlate toxicity spikes with a timeline of major 2013-2017 events to test whether platform-wide or community-specific increases follow external triggers. For life cycle analysis, we will cluster subreddit toxicity trajectories to identify common patterns such as: chronic toxicity (sustained high levels), acute outbreaks (sharp spikes with recovery), gradual radicalization, and intermittent episodes. To build an early warning system, we will use features from each subreddit's first N weeks activity (initial toxicity trends, network position, linguistic markers) and train logistic regression or random forest classifiers to predict chronic toxicity development, validating on a testing split.

## PROPOSED TIMELINE & ORGANIZATION WITHIN THE TEAM

## REFERENCES
[1] VAIDYA, Aatman, NAGAR, Seema, et NANAVATI, Amit A. Analysing the Spread of Toxicity on Twitter. In : Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD). 2024. p. 118-126.
[2] SHETH, Amit, SHALIN, Valerie L., et KURSUNCU, Ugur. Defining and detecting toxicity on social media: context and knowledge are key. Neurocomputing, 2022, vol. 490, p. 312-318.
[3] KUMAR, Srijan, HAMILTON, William L., LESKOVEC, Jure, et al. Community interaction and conflict on the web. In : Proceedings of the 2018 world wide web conference. 2018. p. 933-943.