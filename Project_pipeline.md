## PROJECT PIPELINE
### **PHASE 1: DATA PREPARATION** (Milestone P2)

### STEP 1: Preprocessing & Cleaning
#### Data Quality Checks, Unification & Standardization

1. Import TSV files and expand PROPERTIES column into 86 features
2. Create unified view with `link_type` column for combined analyses (single merged dataset between title and body)
3. Standardize subreddit names (lowercase, trim whitespace)
4. Convert timestamps to datetime and split into year, month, and day columns. 
5. Fromate embeddings datasets (one subreddit per row, one column for each embedding dimension)
6. Check data quality (missing values, timestamp parsing)
7. Check for duplicates and document dataset overlap


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
2. Build directed graphs for each time period
3. Compute node edge attributes: degrees, toxicity metrics, network centrality measures, eigenvector centrality
4. Store as adjacency lists or graph objects for analysis
---

### **PHASE 2: OUTBREAK ANALYSIS** (After P2)

### Step 4: Find Communities
- Detect subreddit clusters (who interacts with whom?)
- Characterize each community's baseline behavior
- Track community changes over time

### Step 5: Identify "Patient Zero"
- **Detect outbreaks**: When toxicity suddenly spikes
- **Find origins**: Which subreddit started it?
- **Measure impact**: How big did the outbreak get?

---
### **PHASE 3: PROPAGATION ANALYSIS** (After P2)

### Step 6: Trace Cascades
- **Find chains**: A→B→C toxic spreading patterns
- **Analyze mutations**: How toxicity changes as it spreads
- **Model contagion**: What makes toxicity spread?

### Step 7: Study Immunity
- **Positive alliances**: Do friendly connections protect communities?
- **Resistance factors**: What makes communities immune?

---

### **PHASE 4: LIFE CYCLES & EVOLUTION** (After P2)

### Step 8: Find Cycles & Events
- **Periodicity**: Does toxicity follow seasonal patterns? (Fourier analysis)
- **Event correlation**: Do real-world events trigger spikes?

### Step 9: Track Evolution
- **Classify trajectories**: How do toxic communities evolve?: Is there a gradual radicalization? Or sudden outbreaks? Do we observe chronic toxicity?
- **Early warning**: Try to predict which communities will become toxic
---
### **PHASE 5: CONCLUSION & RESULT VALIDATION** (After P2)

### Step 10: Create Visualizations
We could create various visualizations to effectively communicate our findings. Examples such as :  Animated networks showing toxicity spread, Epidemic curves (like flu charts), etc.

