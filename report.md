(Part A)
The search algorithms used for evaluation were google, bing, duckduckgo and yahoo. Google
was used as a baseline for comparison. The search algorithms were evaluated based on the
query 'information retrieval evaluation'. We retrieved 7 documents from google, 4 from bing,
20 from duckduckgo and 6 from yahoo. 

(Part B)

A brief report comparing the evaluation of the three search algorithms. For each metric briefly describe the metric, the scores/plots obtained on the tree search techniques and a comparative evaluation of these scores

Google was used as a baseline for comparison. 
According to the results we can see that the other 3 search engines where not as precise as the baseline.

Precision: The fraction of retrieved documents that are relevant
Recall Fraction of relevant documents that are retrieved


The results for the precision are as follow:
Baseline Google:    1.0
Bing:               0.4
Duckduckgo:         0.2
Yahoo:              0.6
In this result we se that 2nd place for Precision is Yahoo, which is 0.4 behind google for precision.

The results for the recall are as follow:
Baseline Google:    1.0
Bing:               0.5714285714285714
Duckduckgo:         0.5714285714285714
Yahoo:              0.42857142857142855
But in the resutlt for recall we se that yahoo falls behind and duckduckog and bing have the same recall number.


google - Precision at 11 standard recall levels: ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

bing - Precision at 11 standard recall levels: ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.6, 0.5, 0.5714285714285714])

duckduckgo - Precision at 11 standard recall levels: ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.6, 0.5, 0.5714285714285714])

yahoo - Precision at 11 standard recall levels: ([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.6, 0.6, 0.6])

Single Valued Summaries:
These metrics provide an indication of which algorithm that might be preferable for the user.
The F-measure here is the harmonic mean precision and recall
if the f = 0 that would indicate that there are no relevant documents that was retrieved.
if the f = 1 that would indicate that the set of relevant documents equal to the set of retrieved documents.
If the F-measure is high, this means that both the precision and recall are high.
Therefore we want a high as possible F-measure.


google - P@5: 1.0, P@10: 1.0, F-measure: 1.0

bing - P@5: 0.6, P@10: 0.4, F-measure: 0.48

duckduckgo - P@5: 0.6, P@10: 0.4, F-measure: 0.48

yahoo - P@5: 0.6, P@10: 0.6, F-measure: 0.6
