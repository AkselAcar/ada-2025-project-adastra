# ADA CS-(402) COURSE PROJECT P2 - Team ADAstra
# Treating Toxicity as an Infectious Disease

**Team Members: Aksel Acar, Emilien Coudurier, Nathan Tabet, Cyprien Tordo, Cedric Zanou**

## PROJECT OVERVIEW
This project analyzes toxicity on Reddit using an epidemiological framework, treating toxic behavior as an "infectious disease" that spreads through social networks. The analysis spans 2013-2017 Reddit data with ~858K interaction records between subreddits.

## CORE RESEARCH ANGLES

### 1. OUTBREAK ANALYSIS
How toxicity appears and spreads initially
- Identifying "breakouts" of toxic behavior
- Detecting "patient zero" communities for cascade of toxic interactions

### 2. PROPAGATION MECHANISMS  
How toxicity spreads through the network
- Analyzing cascade patterns (A→B→C toxic chains)
- Identifying community-specific "mutations" in toxic behavior (e.g., different forms of toxicity in different toxic subreddits)
- Studying potential "immunity" through positive alliances
- Understanding contagion mechanisms and resistance factors

### 3. TEMPORAL PATTERNS
How toxicity follows seasonal patterns
- Identifying cyclical patterns (like flu seasonality)
- Correlating toxicity spikes with major events
- Understanding platform-wide rhythms and trends

### 4. LIFE CYCLE OF TOXICITY
How toxic communities evolve over time
- Mapping developmental trajectories from birth to maturity
- Identifying early warning signs that predict future toxicity
- Classifying trajectory patterns (gradual decline, rapid radicalization, recovery, etc.)
- Predicting community outcomes from early behavioral signals

## PROJECT PIPELINE

### STEP 1: Preprocessing & Cleaning
#### Data Quality Checks & Normalization

1. Import TSV files and expand PROPERTIES column into 86 features
2. Check data quality (missing values, timestamp parsing)
3. Check for duplicates and document dataset overlap
4. Convert timestamps to datetime and split into year, month, and day columns. 
5. Normalize subreddit names (lowercase, trim whitespace)

#### Dataset Organization
Decision made: 
Create unified view with `link_type` column for combined analyses (single merged dataset between title and body)

### STEP 2: Key Metric Definition

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

### PHASE 5: Derived Datasets

#### 5.1 Subreddit-Level Longitudinal Dataset
**One row per subreddit per time period**
- Columns: subreddit_id, time_period, toxic_out/in_count, toxicity_rates, network_centrality, linguistic_profiles, outbreak_status, superspreader_flag, community_cluster_id
- Use: Time series analysis, outbreak detection, superspreader identification

#### 5.2 Edge-Level Interaction Dataset  
**One row per source→target pair per time period**
- Columns: source, target, time_period, interaction_count, toxic_count, toxicity_rate, avg_linguistic_features, cascade_position, temporal_delta, relationship_type
- Use: Propagation cascade analysis, relationship evolution

#### 5.3 Cascade Dataset
**One row per identified cascade event**
- Columns: cascade_id, origin_subreddit, start_timestamp, cascade_size, cascade_depth, duration, velocity, originating_toxicity_type, affected_communities
- Use: Epidemic modeling, contagion dynamics

#### 5.4 Temporal Aggregation Dataset
**One row per time period (platform-wide)**
- Columns: date, toxicity_rate_platform, active_subreddit_count, outbreak_count, superspreader_count, network_density, clustering_coefficient, event_markers
- Use: Seasonal analysis, platform trends, event correlation

#### 5.5 Community Cluster Dataset
**Subreddit community memberships over time**
- Columns: subreddit_id, time_period, community_id, community_size, community_toxicity_rate, cohesion, boundary_strength, linguistic_profile
- Use: Community mutation analysis, echo chamber effects

#### 5.6 Subreddit Biography Dataset
**One row per subreddit (longitudinal summary)**
- Columns: subreddit_id, lifespan, trajectory_pattern, critical_timestamps (first_toxic, onset, peak, recovery), early_warning_scores, escalation_velocity, outcome_classification
- Use: Trajectory clustering, prediction model training, pattern analysis

#### 5.7 Time-Windowed Snapshot Dataset
**One row per subreddit per window (for prediction)**
- Columns: subreddit_id, window_date, days_since_appearance, current_toxicity_rate, lagged_features (lag1/2/3), delta_toxicity, developmental_stage, risk_scores, future_labels (will_turn_toxic_30d/90d)
- Use: Time series forecasting, early warning systems

#### 5.8 Transition Event Dataset
**One row per significant state transition**
- Columns: subreddit_id, transition_date, transition_type, pre/post_context (30d windows), behavioral_changes, trigger_identification
- Use: Event detection, trigger analysis, intervention points

#### 5.9 Cohort Analysis Dataset
**Subreddits grouped by inception period**
- Columns: cohort_id, cohort_size, days_since_inception, survival_statistics, trajectory_distributions, environmental_context
- Use: Generational effects, platform evolution impact
