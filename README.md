# ADA CS-(402) COURSE PROJECT P2 - Team ADAstra
# Epidemiology of Toxicity on Reddit Hyperlink Network

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
We chose to use the following additional dataset: 
- web-redditEmbeddings-subreddits.csv 

<!-- The two first datasets contain subreddit links with their associated properties, the first one being only based on body links and the second on title links. They are independent datasets and since we have no purpose in treating them separately, we merge them and add a flag telling from which dataset the link is from. We benefit from this merging as it reduces the number of separate analyses, and forms a denser, more relevant dataset for our purpose.  -->
This additional dataset contains the embeddings for subreddits, which will allow us to have a visual representation of the communities, from clustering in instance. They are interesting in our project as they offer the possibility of a spatial analysis of groups, especially relevant for patient-0 community identification and propagation study among others.

## METHODS

### Preprocessing
We performed the following preprocessing steps in order to prepare our data for analysis: we start by loading the TSV files containing the subreddit interactions and expand the PROPERTIES column into 86 individual LIWC and sentiment features. This enables us to have more comprehensive data and to perform analysis more easily. Also, for simplicity, we create a unified dataset by merging title and body links, adding a `TYPE` column to distinguish them. We standardize subreddit names (lowercase, whitespace removal) and convert timestamps to datetime format for temporal analysis. Finally, we perform data quality checks for missing values, timestamp parsing errors, and duplicate entries, documenting any dataset overlap between title and body links.

<!-- ### 1. Toxicity Metric Definition and Computation
Since the dataset does not provide a direct toxicity score but rather a negativity score, we construct a composite `toxicity_intensity` metric by combining multiple linguistic and sentiment features available in the data. The formula weights five key components: `toxicity_intensity = α×LIWC_Anger + β×LIWC_Swear + γ×LIWC_Negemo + δ×Negative_VADER + ε×LINK_SENTIMENT`, where coefficients (α, β, γ, δ, ε) are optimized through exploratory analysis to maximize discriminative power between clearly toxic and non-toxic interactions. From this metric, we further investigate other key relevant metrics and aspects detailed in results.ipynb notebook. -->

### 1. Toxicity Metric Definition and Computation
A core challenge of our project is to define a metric of toxicity. By combining a qualitative choice of properties defining toxicity (cf. results_P2.ipynb section 1.2.1) with a constraint to focus on negative-sentiment posts, we constructed a composite TOXICITY_METRIC metric as a weighted average of multiple linguistic and sentiment features available in the data such as FRACTION_OF_DIGITS, LIWC_Swear or LIWC_Negemo. Weights are optimized through exploratory analysis to maximize discriminative power between clearly toxic and non-toxic interactions. From this metric, we derive several others detailed, in results.ipynb.
 
### 2. Temporal Network Construction and Community Detection
We use the embeddings to create a map that represents the proximity between subreddits and link it to their toxicity scores, allowing us to observe how toxicity is distributed across the Reddit hyperlink network. For each monthly or weekly time window, we construct directed weighted graphs with subreddits as nodes and interactions as edges, weighted by interaction frequency and toxicity intensity. We compute standard centrality measures to characterize subreddit network positions. To identify community structures, we will apply the Louvain algorithm to each temporal snapshot and compute baseline toxicity profiles and linguistic fingerprints (dominant LIWC categories) for each detected community. We will also measure semantic similarity between subreddits using cosine distances in the embedding space to determine whether toxic interactions occur preferentially between semantically similar or dissimilar communities.

### 3. Outbreak Detection and Cascade Tracing
We detect outbreaks when a subreddit's toxicity rate jumps above its normal baseline for a prefixed number of consecutive time periods. To find likely "patient zero" communities that initiate these outbreaks, we trace toxic interactions backward from outbreak peaks and prioritize candidate sources that acted earlier: subreddits that sent toxic links before the outbreak receive higher timing scores. We also look at whether they send more toxic links than they receive, and how their activity aligns with downstream spikes. A toxic cascade is a time-ordered chain of toxic links where targets show increased toxicity shortly after exposure. We follow each chain until propagation stops and summarize cascades by length, duration, branching, and changes in language. To establish the actual causal effect of one toxic event on the rest of the chain, we will apply causality tests to assess whether upstream toxic interactions significantly predict increased toxicity in downstream communities rather than it being just the consequences of shared confounders.

<!-- Granger causality & time lagged regression & permutation based null models -->

### 4. Temporal Pattern Analysis and Trajectory Classification
To detect periodicities in toxicity, we will apply FFT to subreddit-level `toxicity_rate` time series, identifying dominant frequencies (monthly, annual cycles) and assessing statistical significance against randomly permuted time series. We will correlate toxicity spikes with a timeline of major 2013-2017 events to test whether platform-wide or community-specific increases follow external triggers. For life cycle analysis, we will cluster subreddit toxicity trajectories to identify common patterns such as: chronic toxicity (sustained high levels), acute outbreaks (sharp spikes with recovery), gradual radicalization, and intermittent episodes. To build an early warning system, we will use features from each subreddit's first N weeks activity (initial toxicity trends, network position, linguistic markers) and train logistic regression or random forest classifiers to predict chronic toxicity development, validating on a testing split.

## PROPOSED TIMELINE & ORGANIZATION WITHIN THE TEAM
### Members:
Aksel (A) \
Emilien (E) \
Nathan (N) \
Cyprien (Cy) \
Cédric (Cé)

W9:   Find communities (N, Cy) + Identify Patient 0 (A, E, Cé) + H2 \
W10: Find communities (N, Cy) + Identify Patient 0 (A, E, Cé) + H2 \
W11: Trace cascades (E, Cy) + Study Immunity (N, A, Cé)      + H2 \
W12: Trace cascades (E, Cy) + Study Immunity (N, A) + Create visualization (Cé) \
W13: Find Cycles & Events (E, N) + Track evolution (Cy) + Create visualization (A, Cé) \
W14: Find Cycles & Events (E, N) + Track evolution (Cy) + Create visualization (A, Cé)

## REFERENCES
[1] VAIDYA, Aatman, NAGAR, Seema, et NANAVATI, Amit A. Analysing the Spread of Toxicity on Twitter. In : Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD). 2024. p. 118-126.

[2] SHETH, Amit, SHALIN, Valerie L., et KURSUNCU, Ugur. Defining and detecting toxicity on social media: context and knowledge are key. Neurocomputing, 2022, vol. 490, p. 312-318.

[3] KUMAR, Srijan, HAMILTON, William L., LESKOVEC, Jure, et al. Community interaction and conflict on the web. In : Proceedings of the 2018 world wide web conference. 2018. p. 933-943.