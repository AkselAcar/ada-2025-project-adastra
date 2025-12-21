### ?. Case Studies
We aim to deepen our comprehension of the toxic events through case studies. That is, by performing an analysis on a specific time period and for a carefully chosen set of subreddits, we expect to highlight previously-discussed behaviors, and discover additional ones. 

Before anything, we select an interesting group of subreddits.
A typical subreddit part of a clustered behavior (high-volume balanced (hub) - bursty, low-volume victim - mixt, ...) is chosen and its associated habitual neighbors are identified. To this set of neighbors, we add all subreddits interacting with the initially chosen subreddit. Note that the union is kept, since some of its neighbors may interact with it. This selected subreddit is called the protagonist.

#### ?.1 Time Window Identification
We choose a reduced time window, from daily interactions over the whole time period (January 2014 to April 2017). We assess peaks and compare them with daily negative interactions over the same period. A period of interest is qualitatively picked.

#### ?.2 Toxic Event Assessment
We then assess if the event is toxic or not. The event is defined as the time series in the selected window of the total negative interactions of all subreddits within the chosen group. Its nature is assessed from the daily activity of the group, and most active subreddits (overall and toxic active) are highlighted. From the trend of negative interactions, we decide if this event is a potential toxic one.

#### ?.3 Life Cycle Analysis
Finally, we analyze the life cycle of the event. For this, we perform a causal inference analysis using the PCMCI algorithm, reconstructing a causal graph between the most active subreddits time series and the protagonist. This allows us to identify catalysts and true actors in the event, as well as attenuators. Furthermore, death of the event is assigned to passive or active.


## REFERENCES
[?] Causal Discovery with Multivariate Time Series Data (https://medium.com/causality-in-data-science/causal-discovery-with-multivariate-time-series-data-a3f7ffc16747)

[?] Tigramite Documentation (https://jakobrunge.github.io/tigramite/index.html#)

[?] Krich, C., Runge, J., Miralles, D. G., Migliavacca, M., Perez-Priego, O., El-Madany, T., ... & Mahecha, M. D. (2020). Estimating causal networks in biosphere–atmosphere interaction with the PCMCI approach. Biogeosciences, 17(4), 1033-1061.