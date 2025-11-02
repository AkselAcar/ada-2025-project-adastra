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

### PHASE 1: Preprocessing & Cleaning
#### Data Quality Checks

1. Import TSV files and expand PROPERTIES column into 86 features
2. Check data quality (missing values, timestamp parsing)
3. Check for duplicates and document dataset overlap

#### Data Normalization
1. Normalize subreddit names (lowercase, trim whitespace)
2. Create subreddit ID mappings for memory efficiency

#### Temporal Feature Extraction
Convert timestamps to datetime and extract temporal features (year, month, day)

#### Toxicity Metric Definitions
Define and compute core toxicity metrics that will be used throughout the analysis:

**Rate-Based Metrics:**
To quantify prevalence and ratios of toxic behavior:
- `toxicity_rate`: Proportion of toxic source messages (toxic_count / total_count) total and for a fixed time window
- `toxicity_ratio`: Ratio of toxic to non-toxic source messages (toxic_count / non_toxic_count)

**Intensity Metrics:**
To capture severity of toxic behavior :
- `toxicity_intensity`: Weighted score combining LINK_SENTIMENT with linguistic features
  - Formula: `base_toxicity_offset + α×LIWC_Anger + β×LIWC_Swear + γ×LIWC_Negemo + δ×Negative_VADER` + standardization

**Advanced Linguistic Metrics:**
- `emotional_volatility`: Standard deviation of `toxicity_intensity`
- `personal_attack_indicators`: LIWC_You + LIWC_Negemo combinations
- `rational_discourse_score`: LIWC_CogMech + LIWC_Cause : this captures logical reasoning in comments since LIWC_CogMech relates to cognitive mechanisms and LIWC_Cause relates to causal reasoning (e.g., for both: LIWC_CogMech words include "cause", "know", "ought", while LIWC_Cause words include "because", "effect", "hence"). 
- `complexity_metrics`: Readability + word length + vocabulary richness -> Could be interesting but complicqted to do, to discuss further.

**Directional Metrics:**
- `toxicity_out`: Toxic links sent by a subreddit (aggressor behavior)
- `toxicity_in`: Toxic links received by a subreddit (victim/target behavior)
- `toxicity_balance`: Net toxicity = toxicity_out - toxicity_in
- `reciprocal_toxicity`: Bidirectional toxic relationships between subreddit pairs = # of bidirectinal edges in the toxic interaction graph

**Temporal Metrics:**
- `toxicity_velocity`: Rate of change in toxicity over time (Δtoxicity_rate / Δtime)
- `toxicity_acceleration`: Second derivative (change in velocity)
- `toxicity_persistence`: Duration of sustained toxic behavior (consecutive toxic periods)

**Network-Adjusted Metrics:**
Used to identify unusually high toxic behavior and periods:
- `normalized_toxicity`: Toxicity rate adjusted for subreddit activity level
- `relative_toxicity`: Toxicity compared to platform baseline for same time period

**Threshold Definitions:**
- `toxic_onset`: First occurrence of toxic behavior (based on first toxicity_intensity score above threshold to be set)
- `chronic_toxicity`: Sustained toxicity_rate > 50% for X+ consecutive weeks/ consecutive links
- `outbreak_threshold`: toxicity_rate > (platform_mean + 2×platform_std)
- `recovery_threshold`: toxicity_rate < 10% for X+ consecutive weeks after toxic period

**Aggregation Levels:**
All metrics computed at multiple granularities:
- Subreddit level (aggregated over time windows)
- Subreddit-pair level (source → target relationships)
- Community cluster level (aggregated within detected communities)

#### Dataset Organization
Decision to make: 
Create unified view with `link_type` column for combined analyses (single merged dataset between title and body)


### PHASE 3: Network Construction

1. Build temporal network snapshots (monthly/yearly windows as shorter windows seem unreasonable given the data)  
2. Create directed graphs for each time period
3. Compute node edge attributes: degrees, toxicity metrics, network centrality measures
4. Store as adjacency lists or graph objects for analysis

### PHASE 4: Feature Engineering



#### 4.2 Propagation Mechanism Features  
- **Cascade chains**: Track A→B→C propagation sequences
- **Cascade metrics**: Depth, breadth, time-to-infection
- **Linguistic shifts**: LIWC feature changes before/after toxic exposure
- **Toxicity style transfer**: Cluster subreddits by toxic linguistic profiles
- **Immunity indicators**: Subreddits exposed to toxicity without becoming toxic
- **Alliance networks**: Positive connection clusters
- **Recovery patterns**: Toxic→non-toxic transitions

#### 4.3 Temporal Pattern Features ->>>> TO REMOVE
- **Seasonal decomposition**: Trend, seasonal, residual components
- **Event correlation windows**: Align with external events
- **Temporal dynamics**: Response times, cooling-off periods, contagion velocity
- **Activity rhythms**: Hour-of-day, day-of-week patterns
- **Platform trends**: Long-term toxicity trajectory



#### 4.5 Life Cycle-Specific Features
- **Developmental stage**: Days since first appearance, activity trajectory, maturity phase
- **Toxicity evolution**: Onset timing, ramp-up velocity, escalation pattern, plateau identification
- **Early warning indicators**:
  - Pre-toxic linguistic precursors (rising anger/negativity before first toxic link)
  - Network precursors (degree changes, clique formation, reciprocity decline)
  - Activity precursors (posting volatility, sudden connection influx)
- **Critical transitions**: First toxic link sent/received, threshold crossings, reciprocal toxicity
- **Trajectory patterns**: Innocent→decline, born toxic, rapid radicalization, oscillating, recovery, chronic
- **Trajectory shape**: Peak toxicity, time-to-peak, duration, volatility, trend slope, curvature
- **Risk scores**: Composite toxicity risk score (TRS), time-to-toxicity predictors, stability scores


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
