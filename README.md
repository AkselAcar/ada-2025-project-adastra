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

## PROJECT PIPELINE

### STEP 1: Preprocessing & Cleaning
#### Data Quality Checks, Unification & Standardization

1. Import TSV files and expand PROPERTIES column into 86 features
2. Create unified view with `link_type` column for combined analyses (single merged dataset between title and body)
3. Standardize subreddit names (lowercase, trim whitespace)
4. Convert timestamps to datetime and split into year, month, and day columns. 
5. Fromate embeddings datasets (one subreddit per row, one column for each embedding dimension)
6. Check data quality (missing values, timestamp parsing)
7. Check for duplicates and document dataset overlap


### STEP : Key Metric Definition

**Aggregation Levels:**
All metrics computed at multiple scopes:
- Subreddit level (aggregated over time windows)
- Subreddit-pair level (source → target relationships)
- Community cluster level (aggregated within detected communities)

#### Toxicity Metric Definitions
Define and compute core toxicity metrics that will be used throughout the analysis:

**Intensity Metrics:**
To capture severity of toxic behavior: 
- `toxicity_intensity`: Weighted score combining LINK_SENTIMENT with linguistic features
  - Formula: `α×LIWC_Anger + β×LIWC_Swear + γ×LIWC_Negemo + δ×Negative_VADER + ε×LINK_SENTIMENT`

- `base_toxicity_offset`: A baseline offset computed by averaging toxicity_intensity scores across all links to establish a reference point. 
This would also be used as a threshold for defining toxic vs non-toxic messages.

- `toxicity_std`: The standard deviation of toxicity_intensity scores across all links.

- `toxicity_zscore = (toxicity_intensity - base_toxicity_offset) / toxicity_std`: this formula would standardise toxicity_intensity scores to allow meaningful comparisons across different links. 

**Count Metrics:**
Foundation for all rate-based calculations. Counts are computed within a specific scope (not the entire dataset):
- `toxic_count`: Number of messages (links) classified as toxic within the given scope. A message is toxic if `toxicity_intensity > toxicity_threshold` (threshold defined below).
- `non_toxic_count`: Number of messages classified as non-toxic (`toxicity_intensity ≤ toxicity_threshold`) within the scope.
- `total_count`: Total number of messages within the scope = `toxic_count + non_toxic_count`.


**Rate-Based Metrics:**
To quantify frequency and ratios of toxic behavior:
- `toxicity_rate`: 
Proportion of toxic source messages in a time window, `toxic_count / (toxic_count + non_toxic_count)`. Range: 0–1.
- An **optional** metric derived from this one would be the `toxicity_odds = (toxic_count / non_toxic_count) = toxicity_rate / (1 − toxicity_rate)` as it could be used to model odds and log-odds for logistic models in the following steps of the project.

**Advanced Linguistic Metrics:**

These metrics could be useful and interesting when analysing the propagation of the negativity among subreddits. 
- `emotional_volatility`: Standard deviation of `toxicity_intensity`
- `personal_attack_indicators`: LIWC_You + LIWC_Negemo combinations
- `rational_discourse_score`: LIWC_CogMech + LIWC_Cause : this captures logical reasoning in comments since LIWC_CogMech relates to cognitive mechanisms and LIWC_Cause relates to causal reasoning (e.g., for both: LIWC_CogMech words include "cause", "know", "ought", while LIWC_Cause words include "because", "effect", "hence"). 
- `complexity_metrics`: Readability + word length + vocabulary richness.

**Directional Metrics:**
- `toxicity_out`: Toxic links sent by a subreddit (aggressor behavior)
- `toxicity_in`: Toxic links received by a subreddit (victim/target behavior)
- `toxicity_balance`: Net toxicity = `toxicity_out - toxicity_in`
- `reciprocal_toxicity`: Bidirectional toxic relationships between subreddit pairs which corresponds to the number of bidirectinal edges in the toxic interaction graph.

**Temporal Metrics:**
- `toxicity_velocity`: Rate of change in toxicity over time (Δtoxicity_rate / Δtime)
- `toxicity_acceleration`: Second derivative (change in velocity)
- `toxicity_persistence`: Duration of sustained toxic behavior (consecutive toxic periods)

**Threshold Definitions:**
- `toxicity_threshold`: Threshold for classifying individual messages as toxic = `base_toxicity_offset + k×toxicity_std`, where k is a sensitivity parameter (recommended: k=1, capturing ~16% most toxic messages; alternative: k=0 for above-mean, or k=2 for top ~2.5%).
- `toxic_onset`: First occurrence of toxic behavior (based on first toxicity_intensity score above toxicity_threshold)
- `chronic_toxicity`: Sustained toxicity_rate > 50% for X+ consecutive weeks/ consecutive links
- `outbreak_threshold`: toxicity_rate > (platform_mean + 2×platform_std)
- `recovery_threshold`: A subreddit is considered recovered after a toxic period when the weekly toxicity_rate falls below 10% for X consecutive weeks. 


### STEP 3: Network Construction

1. Build temporal network snapshots (monthly/yearly windows as shorter windows seem unreasonable given the data)  
2. Create directed graphs for each time period
3. Compute node edge attributes: degrees, toxicity metrics, network centrality measures
4. Store as adjacency lists or graph objects for analysis

### STEP 4: Derived Datasets

#### 4.1 Subreddit-Level Longitudinal Dataset
**One row per subreddit per time period**
- Columns: source_subreddit, time_period, toxic_out/in_count, toxicity_rate, network_centrality, linguistic_profiles, outbreak_status, community_cluster_id
- Use: Time series analysis, outbreak detection

#### 4.2 Edge-Level Interaction Dataset  
**One row per source → target pair per time period**
- Columns: source, target, time_period, interaction_count, toxic_count, toxicity_rate, avg_linguistic_features, cascade_position, temporal_delta, relationship_type
- Use: Propagation cascade analysis, relationship evolution

#### 4.3 Cascade Dataset
**One row per identified cascade event**
- Columns: cascade_id, origin_subreddit, start_timestamp, cascade_size, cascade_depth, duration, velocity, originating_toxicity_type, affected_communities
- Use: Epidemic modeling, contagion dynamics

#### 4.4 Temporal Aggregation Dataset
**One row per time period (platform-wide)**
- Columns: date, toxicity_rate_platform, active_subreddit_count, outbreak_count, network_density, clustering_coefficient, event_markers
- Use: Seasonal analysis, platform trends, event correlation

#### 4.5 Community Cluster Dataset
**Subreddit community memberships over time**
- Columns: source_subreddit, time_period, community_id, community_size, community_toxicity_rate, cohesion, linguistic_profile
- Use: Community mutation analysis, Communiaty affiliation

#### 4.6 Subreddit Biography Dataset
**One row per subreddit (longitudinal summary)**
- Columns: source_subreddit, lifespan, trajectory_pattern, critical_timestamps (first_toxic, onset, peak, recovery), early_warning_scores, toxicity_velocity, outcome_classification
- Use: Trajectory clustering, prediction model training, pattern analysis

#### 4.7 Time-Windowed Snapshot Dataset
**One row per subreddit per window (for prediction)**
- Columns: source_subreddit, window_date, days_since_appearance, current_toxicity_rate, delta_toxicity, developmental_stage
- Use: Time series forecasting, early warning systems

#### 4.8 Transition Event Dataset
**One row per significant state transition**
- Columns: source_subreddit, transition_date, transition_type, pre/post_context (30d windows), behavioral_changes, trigger_identification
- Use: Event detection, trigger analysis, intervention points

#### 4.9 Cohort Analysis Dataset
**Subreddits grouped by inception period**
- Columns: cohort_id, cohort_size, days_since_inception, activity_statistics, trajectory_distributions
- Use: Generational effects

### STEP 5: Analysis-Ready Computations

#### 5.1 Network Metrics (per time window)
- Centrality: Degree, betweenness, closeness, eigenvector, PageRank
- Community detection: Louvain, modularity optimization
- Temporal reachability: k-step neighborhoods
- Influence propagation probabilities

#### 5.2 Baseline Distributions
- Null toxicity rate distributions by subreddit size/activity
- Expected cascade sizes under random interaction model
- Seasonal baseline ranges
- Community stability baselines

#### 5.3 Indexing & Optimization
- Time-series indices for temporal queries
- Spatial indices for network neighborhoods
- Materialized views for common aggregations

#### 5.4 Life Cycle-Specific Computations
- **Trajectory clustering**: Dynamic Time Warping, functional k-means
- **Supervised classification**: Early-stage features → trajectory type prediction
- **Survival analysis**: Time-to-toxicity models, Kaplan-Meier curves, Cox proportional hazards
- **Change point detection**: Statistical break detection, regime shift identification
- **Early warning calibration**: ROC curves, optimal thresholds, sensitivity/specificity at different lead times
- **Causal inference prep**: Propensity score matching, difference-in-differences, instrumental variables

### STEP 6: Validation & Quality Control

#### 6.1-6.3 General Quality Checks
- Missing data summaries
- Outlier detection (linguistic features)
- Distribution plots and correlation matrices
- Propagation temporal ordering validation
- Network coherence checks

#### 6.4 Life Cycle Validation
- Trajectory temporal coherence checks
- Minimum observation requirements
- Outlier trajectory detection
- Prediction model validation (temporal train-test splits, rolling origin forecasting)
- Calibration assessment
- Life cycle visualizations (spaghetti plots, heatmaps, Sankey diagrams, survival curves, phase portraits)

### STEP 7: External Data Integration

#### 7.1 Event Timeline Framework
- External events: Political events, Reddit policy changes, major news, elections

#### 7.2 Subreddit Metadata Enrichment
- Creation dates, Subscriber counts, Ban/quarantine records

#### 7.3 Control Variables
- Platform growth metrics, policy change timeline




# PROJECT PIPELINE OVERVIEW

## **PHASE 1: DATA PREPARATION** (Milestone P2)

### Step 1: Clean & Unify Data
- Load TSV files
- Merge title/body datasets
- Standardize subreddit names
- Parse timestamps
- Check data quality

### Step 2: Define Core Metrics
- **Toxicity Score**: Combine sentiment + LIWC features
- **Toxicity Rate**: % of toxic messages per subreddit/time
- **Directional Metrics**: Toxicity sent vs received
- **Temporal Metrics**: How toxicity changes over time

### Step 3: Build Networks
- Create monthly network snapshots
- Build directed graphs (subreddit → subreddit)
- Calculate network statistics (degree, centrality)

---

## **PHASE 2: OUTBREAK ANALYSIS** (After P2)

### Step 4: Find Communities
- Detect subreddit clusters (who interacts with whom?)
- Characterize each community's baseline behavior
- Track community changes over time

### Step 5: Identify "Patient Zero"
- **Detect outbreaks**: When toxicity suddenly spikes
- **Find origins**: Which subreddit started it?
- **Measure impact**: How big did the outbreak get?

---

## **PHASE 3: PROPAGATION ANALYSIS** (After P2)

### Step 6: Trace Cascades
- **Find chains**: A→B→C toxic spreading patterns
- **Analyze mutations**: How toxicity changes as it spreads
- **Model contagion**: What makes toxicity spread?

### Step 7: Study Immunity
- **Positive alliances**: Do friendly connections protect communities?
- **Resistance factors**: What makes communities immune?

---

## **PHASE 4: TEMPORAL PATTERNS** (After P2)

### Step 8: Find Cycles & Events
- **Periodicity**: Does toxicity follow seasonal patterns?
- **Event correlation**: Do real-world events trigger spikes?
- **Fourier analysis**: Detect hidden rhythms

---

## **PHASE 5: LIFE CYCLES** (After P2)

### Step 9: Track Evolution
- **Classify trajectories**: How do toxic communities evolve?
  - Gradual radicalization
  - Sudden outbreak then recovery
  - Chronic toxicity
- **Early warning**: Predict which communities will become toxic

---

## **PHASE 6: TESTING & VALIDATION** (After P2)

### Step 11: Create Visualizations
- Animated networks showing toxicity spread
- Epidemic curves (like flu charts)
- Interactive dashboards

### Step 12: Check Robustness
- Sensitivity tests
- Validate on different time periods
- Rule out alternative explanations

## REFERENCES

[1] VAIDYA, Aatman, NAGAR, Seema, et NANAVATI, Amit A. Analysing the Spread of Toxicity on Twitter. In : Proceedings of the 7th Joint International Conference on Data Science & Management of Data (11th ACM IKDD CODS and 29th COMAD). 2024. p. 118-126.
[2] SHETH, Amit, SHALIN, Valerie L., et KURSUNCU, Ugur. Defining and detecting toxicity on social media: context and knowledge are key. Neurocomputing, 2022, vol. 490, p. 312-318.
[3] KUMAR, Srijan, HAMILTON, William L., LESKOVEC, Jure, et al. Community interaction and conflict on the web. In : Proceedings of the 2018 world wide web conference. 2018. p. 933-943.