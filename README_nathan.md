### 2. Hyperlink Network Mapping

To better understand how toxicity is spreading throughout reddit, we map the hyperlink network environment and investigate how toxic interactions manifest and evolve spatially. In this section, we use the subreddit embeddings additionnal dataset which represents them and encapsulates their proximities in a 300-dimensional space. From these embeddings, we build a map of Reddit and use it to determine:
1. How different topics are reprented spatially in the Hyperlink Network;
2. How toxicity is distributed across the map and evolving through time.



### 4. Finding toxic communities

Once the clusters of toxic behaviors are computed, we look at their distribution across the Reddit map to observe whether certain topics tend to have a certain behavior, or if agressors and victims are evenly distributed. We then isolate networks of toxic interactions illustrating the detected behaviors using the Louvain algorithm and represent them as interaction networks, with subreddits as nodes and the number of toxic interactions between them as weighted edges. 