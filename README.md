# ADA CS-(402) COURSE PROJECT - Team ADAstra
# Epidemiology of Toxicity on Reddit Hyperlink Network

**Team Members: Aksel Acar, Emilien Coudurier, Nathan Tabet, Cyprien Tordo, Cedric Zanou**

## PROJECT OVERVIEW
Online communities form large interaction networks where behaviors can spread in ways that closely resemble contagion processes. In this project, we study toxicity on Reddit as a virus-like phenomenon, treating toxic behavior as something that can emerge, propagate, mutate, and eventually die out across interconnected subreddits. The full datastory can be found on the **project website**:

https://ctordo.github.io/reddit-hyperlink-network-project-draft/

Rather than considering toxicity as isolated messages or users, we focus on its community-level dynamics, drawing parallels with epidemiology: some communities act as sources of infection, others as vulnerable hosts or resistant populations. Recent related works have been conducted, especially on identifying toxicity in social media [1][2] and intercommunity interactions [3], and we intend to analyze how toxicity thrives based on their contribution. By following toxicity through time and across interactions, we aim to characterize its lifecycles, identify recurring propagation patterns, and understand the conditions under which toxic behavior amplifies or is contained within the Reddit ecosystem.


## CORE RESEARCH ANGLES

Among others, we aim to address the following research questions:
### How does toxicity emerge within communities?
- Do some subreddits consistently act as early sources of toxic content?
- Are there temporal signatures preceding toxic outbreaks?
- Can we distinguish episodic toxicity from sustained toxic behavior?
### How does toxicity propagate across the Reddit network?
- Do distinct propagation patterns appear between source and target subreddits?
- How do interaction frequency, regularity, and network position affect spread?
- Can we identify clusters of communities with similar toxic dynamics?
### What temporal patterns characterize toxic behavior?
- Are there identifiable lifecycles for toxic events (onset, escalation, decay)?
- Do subreddits exhibit chronic versus burst-driven toxicity patterns?
- Can recent activity predict future toxic targeting or aggression?

## ADDITIONAL DATASETS 
We chose to use the following additional dataset: 
- web-redditEmbeddings-subreddits.csv 

This additional dataset contains the embeddings for subreddits, which will allow us to have a visual representation of the communities, from clustering in instance. They are interesting in our project as they offer the possibility of a spatial analysis of groups, especially relevant for patient-0 community identification and propagation study among others.

## METHODS

### Preprocessing
We performed the following preprocessing steps in order to prepare our data for analysis: we start by loading the TSV files containing the subreddit interactions and expand the PROPERTIES column into 86 individual LIWC and sentiment features. This enables us to have more comprehensive data and to perform analysis more easily. Also, for simplicity, we create a unified dataset by merging title and body links, adding a `TYPE` column to distinguish them. We standardize subreddit names (lowercase, whitespace removal) and convert timestamps to datetime format for temporal analysis. Finally, we perform data quality checks for missing values, timestamp parsing errors, and duplicate entries, documenting any dataset overlap between title and body links.


### 1. Toxicity Metric Definition and Computation
A core challenge of our project is to define a metric that captures *toxicity* of a subreddit post. By combining a qualitative choice of properties defining toxicity with a constraint to focus on negative-sentiment posts, we constructed a composite `TOXICITY_METRIC` metric as a weighted average of multiple linguistic and sentiment features available in the data such as FRACTION_OF_DIGITS, LIWC_Swear or LIWC_Negemo. Weights are optimized through exploratory analysis to maximize discriminative power between clearly toxic and non-toxic interactions. Passing a threshold on this scoring, we also derive a binary classification `TOXICITY_CAT`, allowing to distinguish between non-toxic and toxic posts.
 
### 2. Hyperlink Network Mapping

To better understand how toxicity is spreading throughout reddit, we map the hyperlink network environment and investigate how toxic interactions manifest and evolve spatially. In this section, we use the subreddit embeddings additionnal dataset which represents them and encapsulates their proximities in a 300-dimensional space. From these embeddings, we build a map of Reddit and use it to determine:
1. How different topics are reprented spatially in the Hyperlink Network;
2. How toxicity is distributed across the map and evolving through time.

### 3. Temporal Toxicity Clustering Analysis

In this part, we group subreddits into **behavioral archetypes** of toxicity, based on when they are active, whether they act more as aggressors or victims, and how they are positioned in the hyperlink network.

We first build a **feature space** tailored to propagation-style behavior. For each subreddit, we compute:
- **Role-aware temporal features** for source and target roles (how often they send/receive toxicity, how long they are active, how regular or bursty their activity is, and how recent their toxic events are).
- **Network features** (in-degree, out-degree, reciprocity) to describe whether a community is isolated, mostly attacked, mostly attacking, or a hub in mutual conflicts.
- A **volume signal** to identify “mega-subreddits” that operate at much higher toxic activity levels than typical communities.

To obtain reliable clusters, we apply a preprocessing pipeline: we filter out very low-activity subreddits, reduce redundant features (VIF), normalize skewed distributions (log transform and winsorization), detect and flag mega-subreddits on raw counts, and finally standardize all features so they contribute comparably.

We then use **Principal Component Analysis (PCA)** to compress this cleaned feature set into a smaller number of principal components that retain most of the variance, giving each subreddit a compact “behavioral fingerprint”.

On this reduced space, we run **K-Means clustering** and select the number of clusters using internal quality metrics (silhouette score, cluster size balance), stability checks (bootstrap-based Adjusted Rand Index), and standard indices (e.g., Davies–Bouldin, Calinski–Harabasz, Kruskal–Wallis tests). This yields a set of robust, reproducible clusters that summarize the main **propagation styles of toxicity** across Reddit.

### 4. Finding toxic communities

Once the clusters of toxic behaviors are computed, we look at their distribution across the Reddit map to observe whether certain topics tend to have a certain behavior, or if agressors and victims are evenly distributed. We then isolate networks of toxic interactions illustrating the detected behaviors using the Louvain algorithm and represent them as interaction networks, with subreddits as nodes and the number of toxic interactions between them as weighted edges. 


### 5. Case Studies
We aim to deepen our comprehension of the toxic events through case studies. That is, by performing an analysis on a specific time period and for a carefully chosen set of subreddits, we expect to highlight previously-discussed behaviors, and discover additional ones. 

Before anything, we select an interesting group of subreddits.
A typical subreddit part of a clustered behavior (high-volume balanced (hub) - bursty, low-volume victim - mixt, ...) is chosen and its associated habitual neighbors are identified. To this set of neighbors, we add all subreddits interacting with the initially chosen subreddit. Note that the union is kept, since some of its neighbors may interact with it. This selected subreddit is called the protagonist.

#### 5.1 Time Window Identification
We choose a reduced time window, from daily interactions over the whole time period (January 2014 to April 2017). We assess peaks and compare them with daily negative interactions over the same period. A period of interest is qualitatively picked.

#### 5.2 Toxic Event Assessment
We then assess if the event is toxic or not. The event is defined as the time series in the selected window of the total negative interactions of all subreddits within the chosen group. Its nature is assessed from the daily activity of the group, and most active subreddits (overall and toxic active) are highlighted. From the trend of negative interactions, we decide if this event is a potential toxic one.

#### 5.3 Life Cycle Analysis
Finally, we analyze the life cycle of the event. For this, we perform a causal inference analysis using the PCMCI algorithm, reconstructing a causal graph between the most active subreddits time series and the protagonist. This allows us to identify catalysts and true actors in the event, as well as attenuators. Furthermore, death of the event is assigned to passive or active.

## Group Contributions
### Members:

- Aksel Acar: Propagation-style clustering, Repo management, Datastory
- Emilien Coudurier: Study cases (structure and analysis)
- Nathan Tabet: Reddit embeddings mappings, community exploration, toxicity evolution
- Cyprien Tordo: Metrics (toxicity scoring and binary classification), Website Design, Datastory
- Cedric Zanou: Propagation clustering, Repo management, Datastory

## REFERENCES
[1] VAIDYA, Aatman, NAGAR, Seema, et NANAVATI, Amit A. Analysing the Spread of Toxicity on Twitter. In : Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD). 2024. p. 118-126.

[2] SHETH, Amit, SHALIN, Valerie L., et KURSUNCU, Ugur. Defining and detecting toxicity on social media: context and knowledge are key. Neurocomputing, 2022, vol. 490, p. 312-318.

[3] KUMAR, Srijan, HAMILTON, William L., LESKOVEC, Jure, et al. Community interaction and conflict on the web. In : Proceedings of the 2018 world wide web conference. 2018. p. 933-943.

[4] Causal Discovery with Multivariate Time Series Data (https://medium.com/causality-in-data-science/causal-discovery-with-multivariate-time-series-data-a3f7ffc16747)

[5] Tigramite Documentation (https://jakobrunge.github.io/tigramite/index.html#)

[6] Krich, C., Runge, J., Miralles, D. G., Migliavacca, M., Perez-Priego, O., El-Madany, T., ... & Mahecha, M. D. (2020). Estimating causal networks in biosphere–atmosphere interaction with the PCMCI approach. Biogeosciences, 17(4), 1033-1061.